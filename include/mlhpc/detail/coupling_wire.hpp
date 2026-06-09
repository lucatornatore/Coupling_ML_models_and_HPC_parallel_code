// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/detail/coupling_wire.hpp -- private wire format shared by
//  the case-B and case-C Coupling implementations.
//
//  This is NOT part of the harness' public API.  The `detail`
//  sub-namespace marks it as implementation-only: only the
//  src/coupling_*.cpp translation units may include it.  Hoisting
//  the wire-format definitions here ensures that B and C speak
//  exactly the same protocol, with one source of truth.
//
//  Protocol summary (used by both B and C; case A has no wire
//  traffic for the cooling step):
//
//      TAG_QUERY   : QueryHeader, sent as MPI_BYTE.  16 bytes,
//                    trivially copyable.  Carries the particle
//                    count, the metal count, and a bitmask telling
//                    the receiver which optional columns follow on
//                    the data tag.  Setting bit 31 (HDR_SHUTDOWN)
//                    turns the header into a one-shot drain
//                    sentinel; in that case no TAG_DATA follows.
//
//      TAG_DATA    : flat Real-typed payload, length given by
//                    `wire_payload_size(hdr)`.  Layout:
//                        rho [n]  u [n]  [ne [n]]  [nh [n]]  [Zs [n*nmetal]]
//                    Skipped entirely when n == 0.
//
//      TAG_ANSWER  : the cooling-rate buffer, `n` Real elements.
//                    Skipped when n == 0.
//
//  Endianness: the harness targets homogeneous clusters (same
//  endianness on every node).  If a future deployment is
//  heterogeneous we'll switch QueryHeader to a typed MPI struct.
// ---------------------------------------------------------------------

#ifndef MLHPC_DETAIL_COUPLING_WIRE_HPP
#define MLHPC_DETAIL_COUPLING_WIRE_HPP

#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace mlhpc::detail
{

// ---------------------------------------------------------------------
//  Wire-format header.  Sent as MPI_BYTE so its bytes travel
//  verbatim across MPI; alignment is 8, total size 16.  Padding
//  is implicit between `nmetal` and `flags` (none -- they pack
//  tightly).  Trivially copyable so MPI_BYTE is safe.
// ---------------------------------------------------------------------
struct QueryHeader
{
                  std::int64_t           n;        // particles in this query
                  std::int32_t           nmetal;   // metals per particle (0 if Zs absent)
                  std::uint32_t          flags;    // bit-mask, see HDR_*
};

static_assert (std::is_trivially_copyable_v<QueryHeader>,
               "QueryHeader must be trivially copyable to be sent as MPI_BYTE.");
static_assert (sizeof (QueryHeader) == 16,
               "QueryHeader layout has shifted -- the wire format is "
               "now incompatible with peers built from the old header.");

// Flag bits inside QueryHeader::flags.  Low bits carry optional-column
// presence; the top bit carries the drain sentinel.
constexpr std::uint32_t  HDR_HAS_NE    = 1u << 0;
constexpr std::uint32_t  HDR_HAS_NH    = 1u << 1;
constexpr std::uint32_t  HDR_HAS_ZS    = 1u << 2;
constexpr std::uint32_t  HDR_SHUTDOWN  = 1u << 31;

// MPI tags.  Distinct values for header / payload / answer let the
// receiver post specific Recvs without re-probing.  Chosen well above
// zero so they don't collide with any caller tags the harness might
// add later.
constexpr int  TAG_QUERY   = 100;
constexpr int  TAG_DATA    = 101;
constexpr int  TAG_ANSWER  = 102;

// ---------------------------------------------------------------------
//  mpi_real_dtype -- pick MPI_FLOAT / MPI_DOUBLE based on the
//  compile-time `mlhpc::Real`.  Not constexpr because MPI_FLOAT /
//  MPI_DOUBLE are opaque runtime handles.
// ---------------------------------------------------------------------
inline MPI_Datatype
mpi_real_dtype () noexcept
{
  return sizeof (Real) == 8 ? MPI_DOUBLE : MPI_FLOAT;
}

// ---------------------------------------------------------------------
//  make_query_header_for_count -- build the wire header for a slice of
//  a ParticleBatch.
//
//  The optional-field bits are inherited from the full batch, while
//  hdr.n carries only the particle count that will be packed into this
//  particular message.  This keeps chunking cheap: the transport can
//  send contiguous ranges without constructing temporary ParticleBatch
//  objects for every chunk.
// ---------------------------------------------------------------------
inline QueryHeader
make_query_header_for_count (const ParticleBatch  &batch,
                             std::size_t          count) noexcept
{
  const std::size_t  nmetal = batch.nmetal;
  const bool         has_ne = (batch.ne.size () == batch.n) && (count > 0);
  const bool         has_nh = (batch.nh.size () == batch.n) && (count > 0);
  const bool         has_zs = (nmetal > 0) && (count > 0)
                              && (batch.Zs.size () == batch.n * nmetal);

  QueryHeader  hdr {};
  hdr.n      = static_cast<std::int64_t>  (count);
  hdr.nmetal = static_cast<std::int32_t>  (nmetal);
  hdr.flags  = 0;
  if (has_ne) hdr.flags |= HDR_HAS_NE;
  if (has_nh) hdr.flags |= HDR_HAS_NH;
  if (has_zs) hdr.flags |= HDR_HAS_ZS;
  return hdr;
}

// ---------------------------------------------------------------------
//  make_query_header -- full-batch convenience wrapper.
// ---------------------------------------------------------------------
inline QueryHeader
make_query_header (const ParticleBatch &batch) noexcept
{
  return make_query_header_for_count (batch, batch.n);
}

// ---------------------------------------------------------------------
//  wire_payload_size -- count of `Real` elements that follow `hdr`
//  on the TAG_DATA message.  Returns 0 if no payload should be sent.
// ---------------------------------------------------------------------
inline std::size_t
wire_payload_size (const QueryHeader &hdr) noexcept
{
  const std::size_t  n      = static_cast<std::size_t> (hdr.n);
  const std::size_t  nmetal = static_cast<std::size_t> (hdr.nmetal);
  const bool         has_ne = (hdr.flags & HDR_HAS_NE) != 0;
  const bool         has_nh = (hdr.flags & HDR_HAS_NH) != 0;
  const bool         has_zs = (hdr.flags & HDR_HAS_ZS) != 0;
  const std::size_t  cols     = 2 + (has_ne ? 1 : 0) + (has_nh ? 1 : 0);
  const std::size_t  zs_count = has_zs ? n * nmetal : 0;
  return cols * n + zs_count;
}

// ---------------------------------------------------------------------
//  pack_payload_range -- copy a contiguous particle range into a flat
//  MPI payload.
//
//  Inputs:  batch -- source SoA columns.  Required columns rho/u must
//                   have length batch.n; optional columns are present
//                   when their size matches batch.n.
//           first -- first particle index to pack.
//           hdr   -- header built for this range; hdr.n is the range
//                   length, not the full batch length.
//  Output:  wire  -- contiguous payload in the documented wire order.
//
//  The copies walk contiguous memory in each SoA column and write one
//  contiguous output buffer.  That is friendly to hardware prefetchers
//  and lets MPI send a single Real-typed payload per chunk.
// ---------------------------------------------------------------------
inline void
pack_payload_range (const    ParticleBatch       &batch,
                    const    std::size_t          first,
                    const    QueryHeader         &hdr,
                             std::vector<Real>   &wire)
{
  const std::size_t  n = static_cast<std::size_t> (hdr.n);
  if (n == 0) return;

  assert (first <= batch.n);
  assert (n <= batch.n - first);

  const std::size_t  nmetal = static_cast<std::size_t> (hdr.nmetal);
  const bool         has_ne = (hdr.flags & HDR_HAS_NE) != 0;
  const bool         has_nh = (hdr.flags & HDR_HAS_NH) != 0;
  const bool         has_zs = (hdr.flags & HDR_HAS_ZS) != 0;

  std::size_t  off = 0;
  std::copy_n (batch.rho.begin () + first, n, wire.begin () + off);
  off += n;
  std::copy_n (batch.u.begin   () + first, n, wire.begin () + off);
  off += n;
  if (has_ne)
    {
      std::copy_n (batch.ne.begin () + first, n, wire.begin () + off);
      off += n;
    }
  if (has_nh)
    {
      std::copy_n (batch.nh.begin () + first, n, wire.begin () + off);
      off += n;
    }
  if (has_zs)
    {
      const std::size_t  z_first = first * nmetal;
      const std::size_t  z_count = n * nmetal;
      std::copy_n (batch.Zs.begin () + z_first, z_count,
                   wire.begin () + off);
      off += z_count;
    }
}

// ---------------------------------------------------------------------
//  pack_payload -- full-batch convenience wrapper.
// ---------------------------------------------------------------------
inline void
pack_payload (const    ParticleBatch       &batch,    // in
              const    QueryHeader         &hdr,      // in
                       std::vector<Real>   &wire)     // out: pre-sized
{
  pack_payload_range (batch, 0, hdr, wire);
}

// ---------------------------------------------------------------------
//  unpack_payload -- inverse of pack_payload.  Resizes `batch` to
//  match `hdr` (clearing optional columns first so the size-0-means-
//  absent invariant holds on the resulting batch), then copies the
//  Real columns out of `wire`.
// ---------------------------------------------------------------------
inline void
unpack_payload (const    QueryHeader         &hdr,     // in
                const    std::vector<Real>   &wire,    // in
                         ParticleBatch       &batch)   // out
{
  const std::size_t  n      = static_cast<std::size_t> (hdr.n);
  const std::size_t  nmetal = static_cast<std::size_t> (hdr.nmetal);
  const bool         has_ne = (hdr.flags & HDR_HAS_NE) != 0;
  const bool         has_nh = (hdr.flags & HDR_HAS_NH) != 0;
  const bool         has_zs = (hdr.flags & HDR_HAS_ZS) != 0;

  batch.resize (n, nmetal);
  batch.ne.clear ();
  batch.nh.clear ();
  batch.Zs.clear ();

  if (n == 0) return;

  std::size_t  off = 0;
  std::copy (wire.begin () + off, wire.begin () + off + n,
             batch.rho.begin ());
  off += n;
  std::copy (wire.begin () + off, wire.begin () + off + n,
             batch.u.begin ());
  off += n;
  if (has_ne)
    {
      batch.ne.resize (n);
      std::copy (wire.begin () + off, wire.begin () + off + n,
                 batch.ne.begin ());
      off += n;
    }
  if (has_nh)
    {
      batch.nh.resize (n);
      std::copy (wire.begin () + off, wire.begin () + off + n,
                 batch.nh.begin ());
      off += n;
    }
  if (has_zs)
    {
      batch.Zs.resize (n * nmetal);
      std::copy (wire.begin () + off, wire.begin () + off + n * nmetal,
                 batch.Zs.begin ());
      off += n * nmetal;
    }
}

}  // namespace mlhpc::detail

#endif  // MLHPC_DETAIL_COUPLING_WIRE_HPP
