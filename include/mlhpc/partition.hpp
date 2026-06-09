// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/partition.hpp -- uniform-partition primitives.
//
//  The harness distributes the gas particles of a snapshot across the
//  MPI ranks with a simple uniform partition:
//
//      base   = total / nranks
//      rem    = total % nranks
//      rank r gets `base + (r < rem ? 1 : 0)` consecutive particles,
//      starting at `r * base + min(r, rem)`.
//
//  This keeps the per-rank counts within ±1 and the slices contiguous
//  in the global ordering -- which matches how Gadget files lay
//  particles out on disk (file f0 first, then f1, ...) and therefore
//  makes the file-mapping in `Format2Snapshot` trivial.
//
//  This is a pure-arithmetic header: it has no I/O and no MPI
//  dependency, so it stays cheap to test and reuse from other modules
//  (e.g. the future inference-pool partitioner in case B/C).
// ---------------------------------------------------------------------

#ifndef MLHPC_PARTITION_HPP
#define MLHPC_PARTITION_HPP

#include <cstddef>

namespace mlhpc
{

// ---------------------------------------------------------------------
//  GlobalSlice -- half-open interval [gstart, gend) into a global
//  particle index space.
// ---------------------------------------------------------------------
struct GlobalSlice
{
  std::size_t   gstart;
  std::size_t   gend;

  std::size_t count () const noexcept { return gend - gstart; }
  bool        empty () const noexcept { return gend == gstart; }
};

// ---------------------------------------------------------------------
//  partition_uniform
//
//  Compute one rank's slice of a uniform partition of `total`
//  contiguous items across `nranks` ranks.
//
//  Inputs:  total   -- size of the global sequence
//           rank    -- this rank's index in [0, nranks)
//           nranks  -- total number of ranks (must be > 0)
//  Returns: the half-open [gstart, gend) slice for this rank.
//
//  Preconditions (checked by assertions; the function is noexcept):
//      0 <= rank < nranks,  nranks > 0.
// ---------------------------------------------------------------------
GlobalSlice
partition_uniform (std::size_t   total,
                   int           rank,
                   int           nranks) noexcept;

}  // namespace mlhpc

#endif  // MLHPC_PARTITION_HPP
