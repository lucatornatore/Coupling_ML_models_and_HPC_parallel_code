// ---------------------------------------------------------------------
//  src/partition.cpp -- implementation of the uniform-partition
//  primitive declared in mlhpc/partition.hpp.
// ---------------------------------------------------------------------

#include "mlhpc/partition.hpp"

#include <cassert>
#include <cstddef>

namespace mlhpc
{

// ---------------------------------------------------------------------
//  partition_uniform
//
//  See header for the contract.  Implementation in one expression:
//      gstart = r * base + min(r, rem)
//      gend   = gstart + base + (r < rem ? 1 : 0)
//  where `base = total / nranks` and `rem = total % nranks`.
// ---------------------------------------------------------------------
GlobalSlice
partition_uniform (std::size_t   total,
                   int           rank,
                   int           nranks) noexcept
{
  assert (nranks > 0);
  assert (rank >= 0 && rank < nranks);

  const std::size_t   r     = static_cast<std::size_t> (rank);
  const std::size_t   n     = static_cast<std::size_t> (nranks);
  const std::size_t   base  = total / n;
  const std::size_t   rem   = total % n;
  const std::size_t   extra = (r < rem) ? 1u : 0u;

  GlobalSlice s {};
  s.gstart = r * base + (r < rem ? r : rem);
  s.gend   = s.gstart + base + extra;
  return s;
}

}  // namespace mlhpc
