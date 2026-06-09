// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/detail/coupling_async_mpi.hpp -- shared non-blocking MPI
//  transport for service-style coupling cases.
//
//  This helper owns the MPI query/answer mechanics that are independent
//  of rank layout.  Case B (ServiceCoupling) decides only which ranks
//  are physics ranks, which ranks are ML servers, and where each physics
//  rank sends its cooling requests.  Once that routing decision exists,
//  the transport below handles packing, chunk submission,
//  progress, waiting, server-side receive queues, answer sends, timings,
//  and shutdown sentinels.
// ---------------------------------------------------------------------

#ifndef MLHPC_DETAIL_COUPLING_ASYNC_MPI_HPP
#define MLHPC_DETAIL_COUPLING_ASYNC_MPI_HPP

#include "mlhpc/coupling.hpp"
#include "mlhpc/inference.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace mlhpc::detail
{

class AsyncMpiTransport final
{
public:

  // -------------------------------------------------------------------
  //  Constructor.
  //
  //  Inputs:  world -- borrowed communicator used for all service
  //                   traffic.  The owning Coupling object keeps this
  //                   communicator alive for the transport lifetime.
  //           cfg   -- batching and back-pressure knobs.  A zero
  //                   max_batch_particles means one MPI query per
  //                   logical evaluate_async() call.
  // -------------------------------------------------------------------
  AsyncMpiTransport (MPI_Comm                world,
                     const CouplingConfig   &cfg) noexcept;

  ~AsyncMpiTransport ();

  // -------------------------------------------------------------------
  //  evaluate_async -- submit one logical physics request.
  //
  //  The input ParticleBatch is copied into one or more contiguous wire
  //  buffers before return, so the caller may reuse or destroy it after
  //  this call.  The output pointer is not copied: it must remain valid
  //  until wait() / wait_all() reports completion.
  // -------------------------------------------------------------------
  CouplingRequest
  evaluate_async (const ParticleBatch       &batch,
                  Real *__restrict           out,
                  int                        target_rank);

  void
  progress ();

  bool
  test (CouplingRequest request);

  void
  wait (CouplingRequest request);

  void
  wait_all ();

  // -------------------------------------------------------------------
  //  serve -- ML-server event loop.
  //
  //  The loop receives query headers without blocking on a particular
  //  source, posts non-blocking payload receives, runs completed
  //  requests through the backend, and keeps answer buffers alive until
  //  their MPI_Isend calls complete.  It returns after the expected
  //  number of physics ranks have sent shutdown sentinels and all
  //  queued work has drained.
  // -------------------------------------------------------------------
  void
  serve (int                       expected_shutdowns,
         const InferenceBackend   &backend);

  void
  shutdown (int target_rank);

  Timings
  timings () const noexcept;

private:

  struct ClientChunk;
  struct ParentRequest;
  struct ServerReceive;
  struct ServerSend;

  std::size_t
  chunk_size (std::size_t remaining) const noexcept;

  std::size_t
  active_client_chunks () const noexcept;

  void
  wait_for_client_slot ();

  ParentRequest *
  find_parent (std::size_t id) noexcept;

  const ParentRequest *
  find_parent (std::size_t id) const noexcept;

  void
  mark_client_chunk_complete (ClientChunk &chunk,
                              double       now);

  bool
  test_request (MPI_Request &request) noexcept;

  void
  post_client_chunk (CouplingRequest       parent,
                     const ParticleBatch  &batch,
                     std::size_t           first,
                     std::size_t           count,
                     Real *__restrict      out,
                     int                   target_rank);

  void
  drain_server_headers (int &shutdowns);

  void
  progress_server_receives (const InferenceBackend &backend);

  void
  progress_server_sends ();

private:

                  MPI_Comm        m_world;
                  std::size_t     m_max_batch_particles;
                  std::size_t     m_max_inflight_requests;
                  std::size_t     m_next_request;
                  Timings         m_timings;

                  std::vector<std::unique_ptr<ClientChunk>>
                                  m_client_chunks;
                  std::vector<ParentRequest>
                                  m_parent_requests;
                  std::vector<std::unique_ptr<ServerReceive>>
                                  m_server_receives;
                  std::vector<std::unique_ptr<ServerSend>>
                                  m_server_sends;
};

}  // namespace mlhpc::detail

#endif  // MLHPC_DETAIL_COUPLING_ASYNC_MPI_HPP
