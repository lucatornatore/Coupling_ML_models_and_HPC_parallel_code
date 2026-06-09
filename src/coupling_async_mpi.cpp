// ---------------------------------------------------------------------
//  src/coupling_async_mpi.cpp -- shared non-blocking MPI transport for
//  service-style ML coupling cases.
// ---------------------------------------------------------------------

#include "mlhpc/detail/coupling_async_mpi.hpp"

#include "mlhpc/detail/coupling_wire.hpp"
#include "mlhpc/inference.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlhpc::detail
{

namespace
{

// ---------------------------------------------------------------------
//  checked_mpi_count -- protect MPI count arguments from silent
//  size_t -> int truncation.
//
//  MPI-3 still uses int counts for the routines used here.  The higher
//  level transport chunks by particle count, but optional cooling fields
//  can still make a payload larger than INT_MAX Real values if the user
//  asks for enormous chunks.  Throwing is preferable to wrapping the
//  count and corrupting the query stream.
// ---------------------------------------------------------------------
int
checked_mpi_count (std::size_t   n,
                   const char   *what)
{
  const std::size_t  max_count =
      static_cast<std::size_t> (std::numeric_limits<int>::max ());
  if (n > max_count)
    {
      throw std::length_error (std::string (what)
                               + " exceeds MPI int count range");
    }
  return static_cast<int> (n);
}

// ---------------------------------------------------------------------
//  checked_nonnegative_particles -- validate particle counts received
//  from the wire before using them as std::size_t values.
// ---------------------------------------------------------------------
std::size_t
checked_nonnegative_particles (const QueryHeader &hdr)
{
  if (hdr.n < 0)
    {
      throw std::runtime_error ("negative particle count in QueryHeader");
    }
  return static_cast<std::size_t> (hdr.n);
}

}  // namespace

// ---------------------------------------------------------------------
//  Private request records.
//
//  ClientChunk owns every buffer whose address is handed to a non-
//  blocking client-side MPI call: the header object, the packed payload,
//  and the request handles.  The caller owns the output buffer, and the
//  public API documents that it must remain live until completion.
//
//  ServerReceive owns an in-flight payload receive.  ServerSend owns the
//  answer buffer until MPI confirms the non-blocking send has completed.
// ---------------------------------------------------------------------
struct AsyncMpiTransport::ClientChunk
{
                  CouplingRequest       parent;
                  QueryHeader           hdr;
                  std::vector<Real>     wire;
                  Real                 *out              = nullptr;
                  int                   target_rank      = -1;
                  MPI_Request           header_request   = MPI_REQUEST_NULL;
                  MPI_Request           data_request     = MPI_REQUEST_NULL;
                  MPI_Request           answer_request   = MPI_REQUEST_NULL;
                  bool                  header_done      = false;
                  bool                  data_done        = false;
                  bool                  answer_done      = false;
                  double                submit_start     = 0.0;
                  double                submit_done      = 0.0;
};

struct AsyncMpiTransport::ParentRequest
{
                  std::size_t           id        = 0;
                  std::size_t           remaining = 0;
};

struct AsyncMpiTransport::ServerReceive
{
                  QueryHeader           hdr;
                  std::vector<Real>     wire;
                  int                   source_rank  = -1;
                  MPI_Request           data_request = MPI_REQUEST_NULL;
                  bool                  data_done    = false;
                  double                iter_start   = 0.0;
};

struct AsyncMpiTransport::ServerSend
{
                  std::vector<Real>     out;
                  int                   target_rank  = -1;
                  MPI_Request           send_request = MPI_REQUEST_NULL;
                  double                iter_start   = 0.0;
                  double                send_start   = 0.0;
};

// ---------------------------------------------------------------------
//  AsyncMpiTransport constructor.
//
//  The communicator is borrowed from the owning Coupling object.  The
//  transport does not duplicate or free it, which keeps communicator
//  ownership in one place and avoids extra MPI handle churn.
// ---------------------------------------------------------------------
AsyncMpiTransport::AsyncMpiTransport (MPI_Comm                world,
                                      const CouplingConfig   &cfg) noexcept
  : m_world                 (world),
    m_max_batch_particles   (cfg.max_batch_particles),
    m_max_inflight_requests (cfg.max_inflight_requests == 0
                             ? 1u : cfg.max_inflight_requests),
    m_next_request          (0),
    m_timings               ()
{
}

AsyncMpiTransport::~AsyncMpiTransport () = default;

// ---------------------------------------------------------------------
//  chunk_size -- choose how many particles to place in the next wire
//  message.
// ---------------------------------------------------------------------
std::size_t
AsyncMpiTransport::chunk_size (std::size_t remaining) const noexcept
{
  if (m_max_batch_particles == 0 || m_max_batch_particles > remaining)
    {
      return remaining;
    }
  return m_max_batch_particles;
}

std::size_t
AsyncMpiTransport::active_client_chunks () const noexcept
{
  return m_client_chunks.size ();
}

// ---------------------------------------------------------------------
//  find_parent -- linear lookup of a logical request record.
//
//  The expected number of live logical requests is small and bounded by
//  the caller's own outstanding work.  A linear scan avoids hash-table
//  machinery and keeps the request book-keeping deterministic.
// ---------------------------------------------------------------------
AsyncMpiTransport::ParentRequest *
AsyncMpiTransport::find_parent (std::size_t id) noexcept
{
  for (ParentRequest &request : m_parent_requests)
    {
      if (request.id == id) return &request;
    }
  return nullptr;
}

const AsyncMpiTransport::ParentRequest *
AsyncMpiTransport::find_parent (std::size_t id) const noexcept
{
  for (const ParentRequest &request : m_parent_requests)
    {
      if (request.id == id) return &request;
    }
  return nullptr;
}

// ---------------------------------------------------------------------
//  test_request -- poll one MPI_Request and normalize MPI_REQUEST_NULL
//  to "already complete".
// ---------------------------------------------------------------------
bool
AsyncMpiTransport::test_request (MPI_Request &request) noexcept
{
  if (request == MPI_REQUEST_NULL)
    {
      return true;
    }

  int  done = 0;
  MPI_Test (&request, &done, MPI_STATUS_IGNORE);
  return done != 0;
}

// ---------------------------------------------------------------------
//  wait_for_client_slot -- apply per-rank back-pressure.
//
//  This prevents a physics rank from posting an unbounded number of
//  large Isend/Irecv operations when the server side is slower than the
//  producer.  The loop calls progress(), so already completed chunks are
//  retired and their buffers are released before more work is posted.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::wait_for_client_slot ()
{
  while (m_max_inflight_requests > 0
         && active_client_chunks () >= m_max_inflight_requests)
    {
      progress ();
    }
}

// ---------------------------------------------------------------------
//  post_client_chunk -- pack and submit one transport chunk.
//
//  Inputs:  parent      -- logical request handle that owns the chunk.
//           batch       -- source SoA columns.
//           first/count -- contiguous particle range in batch/output.
//           out         -- caller-owned logical output buffer.
//           target_rank -- world rank of the ML server selected by the
//                          topology layer.
//
//  The payload is packed before MPI_Isend so the source ParticleBatch no
//  longer matters after this function returns.  All MPI buffers then live
//  in ClientChunk until progress() observes completion.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::post_client_chunk (CouplingRequest       parent,
                                      const ParticleBatch  &batch,
                                      std::size_t           first,
                                      std::size_t           count,
                                      Real *__restrict      out,
                                      int                   target_rank)
{
  wait_for_client_slot ();

  std::unique_ptr<ClientChunk>  chunk (new ClientChunk ());
  chunk->parent       = parent;
  chunk->hdr          = make_query_header_for_count (batch, count);
  chunk->target_rank  = target_rank;
  chunk->out          = (out == nullptr) ? nullptr : out + first;
  chunk->submit_start = MPI_Wtime ();

  const std::size_t  wire_n = wire_payload_size (chunk->hdr);
  chunk->wire.resize (wire_n);
  pack_payload_range (batch, first, chunk->hdr, chunk->wire);

  const int  hdr_count = checked_mpi_count (sizeof (chunk->hdr),
                                            "QueryHeader byte count");
  MPI_Isend (&chunk->hdr, hdr_count, MPI_BYTE, target_rank,
             TAG_QUERY, m_world, &chunk->header_request);
  chunk->header_done = false;

  if (wire_n > 0)
    {
      const int  data_count = checked_mpi_count (wire_n,
                                                 "query payload count");
      MPI_Isend (chunk->wire.data (), data_count, mpi_real_dtype (),
                 target_rank, TAG_DATA, m_world, &chunk->data_request);
      chunk->data_done = false;
    }
  else
    {
      chunk->data_done = true;
    }

  if (count > 0)
    {
      if (chunk->out == nullptr)
        {
          throw std::invalid_argument ("non-empty async request has null output");
        }
      const int  answer_count = checked_mpi_count (count,
                                                   "answer payload count");
      MPI_Irecv (chunk->out, answer_count, mpi_real_dtype (),
                 target_rank, TAG_ANSWER, m_world, &chunk->answer_request);
      chunk->answer_done = false;
    }
  else
    {
      chunk->answer_done = true;
    }

  chunk->submit_done = MPI_Wtime ();
  m_timings.send += (chunk->submit_done - chunk->submit_start);

  m_client_chunks.emplace_back (std::move (chunk));
}

// ---------------------------------------------------------------------
//  evaluate_async -- split, pack, and submit one logical request.
// ---------------------------------------------------------------------
CouplingRequest
AsyncMpiTransport::evaluate_async (const ParticleBatch       &batch,
                                   Real *__restrict           out,
                                   int                        target_rank)
{
  CouplingRequest  parent {};
  parent.id = ++m_next_request;

  ParentRequest  record {};
  record.id        = parent.id;
  record.remaining = 0;
  m_parent_requests.push_back (record);
  ParentRequest  &stored = m_parent_requests.back ();

  if (batch.n == 0)
    {
      stored.remaining += 1;
      post_client_chunk (parent, batch, 0, 0, out, target_rank);
      return parent;
    }

  std::size_t  first = 0;
  while (first < batch.n)
    {
      const std::size_t  count = chunk_size (batch.n - first);
      stored.remaining += 1;
      post_client_chunk (parent, batch, first, count, out, target_rank);
      first += count;
    }

  return parent;
}

// ---------------------------------------------------------------------
//  mark_client_chunk_complete -- retire one finished chunk and update
//  physics-side timing counters.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::mark_client_chunk_complete (ClientChunk &chunk,
                                               double       now)
{
  ParentRequest  *parent = find_parent (chunk.parent.id);
  if (parent != nullptr && parent->remaining > 0)
    {
      parent->remaining -= 1;
    }

  m_timings.wait    += (now - chunk.submit_done);
  m_timings.total   += (now - chunk.submit_start);
  m_timings.n_calls += 1;
}

// ---------------------------------------------------------------------
//  progress -- poll all client-side MPI requests.
//
//  Completion requires the header send, optional payload send, and
//  optional answer receive to have all completed.  Keeping the sends in
//  the completion condition is what makes it safe to release the header
//  and payload buffers when the ClientChunk is erased.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::progress ()
{
  for (std::size_t i = 0; i < m_client_chunks.size (); )
    {
      ClientChunk &chunk = *m_client_chunks [i];

      if (!chunk.header_done)
        {
          chunk.header_done = test_request (chunk.header_request);
        }
      if (!chunk.data_done)
        {
          chunk.data_done = test_request (chunk.data_request);
        }
      if (!chunk.answer_done)
        {
          chunk.answer_done = test_request (chunk.answer_request);
        }

      if (chunk.header_done && chunk.data_done && chunk.answer_done)
        {
          const double  now = MPI_Wtime ();
          mark_client_chunk_complete (chunk, now);
          m_client_chunks.erase (m_client_chunks.begin ()
                                 + static_cast<std::ptrdiff_t> (i));
        }
      else
        {
          ++i;
        }
    }
}

bool
AsyncMpiTransport::test (CouplingRequest request)
{
  progress ();

  const ParentRequest  *parent = find_parent (request.id);
  return parent == nullptr || parent->remaining == 0;
}

void
AsyncMpiTransport::wait (CouplingRequest request)
{
  while (!test (request))
    {
      progress ();
    }
}

void
AsyncMpiTransport::wait_all ()
{
  bool  done = false;
  while (!done)
    {
      progress ();
      done = true;
      for (const ParentRequest &request : m_parent_requests)
        {
          if (request.remaining != 0)
            {
              done = false;
              break;
            }
        }
    }
}

// ---------------------------------------------------------------------
//  drain_server_headers -- receive all currently available query headers.
//
//  Headers are small, so the server receives them immediately after an
//  MPI_Iprobe hit.  Payloads are posted as non-blocking receives and
//  completed later by progress_server_receives(), which lets one ML
//  server keep several physics requests in flight at the same time.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::drain_server_headers (int &shutdowns)
{
  while (true)
    {
      int         available = 0;
      MPI_Status  status {};
      MPI_Iprobe (MPI_ANY_SOURCE, TAG_QUERY, m_world,
                  &available, &status);
      if (!available) return;

      QueryHeader  hdr {};
      const int    hdr_count = checked_mpi_count (sizeof (hdr),
                                                  "QueryHeader byte count");
      MPI_Recv (&hdr, hdr_count, MPI_BYTE, status.MPI_SOURCE,
                TAG_QUERY, m_world, MPI_STATUS_IGNORE);

      if ((hdr.flags & HDR_SHUTDOWN) != 0)
        {
          shutdowns += 1;
          continue;
        }

      checked_nonnegative_particles (hdr);

      std::unique_ptr<ServerReceive>  recv (new ServerReceive ());
      recv->hdr         = hdr;
      recv->source_rank = status.MPI_SOURCE;
      recv->iter_start  = MPI_Wtime ();

      const std::size_t  wire_n = wire_payload_size (recv->hdr);
      recv->wire.resize (wire_n);
      if (wire_n > 0)
        {
          const int  data_count = checked_mpi_count (wire_n,
                                                     "query payload count");
          MPI_Irecv (recv->wire.data (), data_count, mpi_real_dtype (),
                     recv->source_rank, TAG_DATA, m_world,
                     &recv->data_request);
          recv->data_done = false;
        }
      else
        {
          recv->data_done = true;
        }

      m_server_receives.emplace_back (std::move (recv));
    }
}

// ---------------------------------------------------------------------
//  progress_server_receives -- complete ready payloads and run backend.
//
//  The function deliberately unpacks into a SoA ParticleBatch before the
//  backend call.  That preserves the backend contract, keeps feature
//  columns contiguous for vectorized CPU/GPU kernels, and avoids passing
//  wire-layout details outside the transport layer.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::progress_server_receives (const InferenceBackend &backend)
{
  for (std::size_t i = 0; i < m_server_receives.size (); )
    {
      ServerReceive &recv = *m_server_receives [i];
      if (!recv.data_done)
        {
          recv.data_done = test_request (recv.data_request);
        }

      if (!recv.data_done)
        {
          ++i;
          continue;
        }

      const double  recv_done = MPI_Wtime ();
      m_timings.wait += (recv_done - recv.iter_start);

      ParticleBatch  batch {};
      unpack_payload (recv.hdr, recv.wire, batch);

      std::unique_ptr<ServerSend>  send (new ServerSend ());
      send->target_rank = recv.source_rank;
      send->iter_start  = recv.iter_start;
      send->out.resize (batch.n);

      const double  kernel_start = MPI_Wtime ();
      backend.batch_inference (batch, send->out.data ());
      const double  kernel_done  = MPI_Wtime ();
      m_timings.kernel += (kernel_done - kernel_start);

      if (batch.n > 0)
        {
          const int  answer_count = checked_mpi_count (batch.n,
                                                       "answer payload count");
          send->send_start = MPI_Wtime ();
          MPI_Isend (send->out.data (), answer_count, mpi_real_dtype (),
                     send->target_rank, TAG_ANSWER, m_world,
                     &send->send_request);
          m_server_sends.emplace_back (std::move (send));
        }
      else
        {
          const double  now = MPI_Wtime ();
          m_timings.total   += (now - recv.iter_start);
          m_timings.n_calls += 1;
        }

      m_server_receives.erase (m_server_receives.begin ()
                               + static_cast<std::ptrdiff_t> (i));
    }
}

// ---------------------------------------------------------------------
//  progress_server_sends -- retire completed answer sends.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::progress_server_sends ()
{
  for (std::size_t i = 0; i < m_server_sends.size (); )
    {
      ServerSend &send = *m_server_sends [i];
      if (test_request (send.send_request))
        {
          const double  now = MPI_Wtime ();
          m_timings.send    += (now - send.send_start);
          m_timings.total   += (now - send.iter_start);
          m_timings.n_calls += 1;
          m_server_sends.erase (m_server_sends.begin ()
                                + static_cast<std::ptrdiff_t> (i));
        }
      else
        {
          ++i;
        }
    }
}

// ---------------------------------------------------------------------
//  serve -- shared ML-server event loop.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::serve (int                       expected_shutdowns,
                          const InferenceBackend   &backend)
{
  if (expected_shutdowns <= 0)
    {
      return;
    }

  int  shutdowns = 0;
  while (shutdowns < expected_shutdowns
         || !m_server_receives.empty ()
         || !m_server_sends.empty ())
    {
      progress_server_sends ();
      progress_server_receives (backend);
      drain_server_headers (shutdowns);

      if (shutdowns < expected_shutdowns
          && m_server_receives.empty ()
          && m_server_sends.empty ())
        {
          MPI_Status  status {};
          MPI_Probe (MPI_ANY_SOURCE, TAG_QUERY, m_world, &status);
        }
    }
}

// ---------------------------------------------------------------------
//  shutdown -- send a drain sentinel to one ML server.
//
//  The sentinel is tiny, but it still goes through MPI_Isend so the
//  control path uses the same non-blocking transport primitive as real
//  work.  We wait immediately because shutdown has no useful work to
//  overlap and the stack-allocated header must remain live until the
//  send completes.
// ---------------------------------------------------------------------
void
AsyncMpiTransport::shutdown (int target_rank)
{
  QueryHeader  hdr {};
  hdr.flags = HDR_SHUTDOWN;

  MPI_Request  request = MPI_REQUEST_NULL;
  const int    hdr_count = checked_mpi_count (sizeof (hdr),
                                              "QueryHeader byte count");
  MPI_Isend (&hdr, hdr_count, MPI_BYTE, target_rank, TAG_QUERY,
             m_world, &request);
  MPI_Wait (&request, MPI_STATUS_IGNORE);
}

Timings
AsyncMpiTransport::timings () const noexcept
{
  return m_timings;
}

}  // namespace mlhpc::detail
