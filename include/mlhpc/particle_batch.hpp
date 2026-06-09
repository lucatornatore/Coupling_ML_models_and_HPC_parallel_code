// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/particle_batch.hpp -- Struct-of-Arrays gas-particle batch.
//
//  Owns the per-particle cooling-feature buffers for one MPI rank's
//  slice of the snapshot.  The struct is intentionally simple --
//  vectors of `Real` -- so the inference backend can take read-only
//  pointers/spans into the columns it needs without any custom
//  iterator dance.
//
//  Layout:
//    * Scalar per-particle fields are individual `std::vector<Real>`
//      of length `n` (or empty if absent in the snapshot).
//    * The metal-abundance field is a flat `n * nmetal` row-major
//      vector: particle i's element j is at `Zs[i*nmetal + j]`.
//
//  Optional fields (`u`, `Zs`, `ne`, `nh`) may be left empty by the
//  driver when the corresponding block was missing from the
//  snapshot; consumers check `field.size() > 0` before reading them.
//
//  The struct doesn't enforce that the fields are in any particular
//  unit system: the driver decides whether to leave them in code
//  units or to convert to CGS via `mlhpc::units`.  Standard practice
//  in this harness: convert to physical/CGS once, immediately after
//  the read, so the inference layer always sees CGS inputs.
// ---------------------------------------------------------------------

#ifndef MLHPC_PARTICLE_BATCH_HPP
#define MLHPC_PARTICLE_BATCH_HPP

#include "mlhpc/types.hpp"

#include <cstddef>
#include <vector>

namespace mlhpc
{

struct ParticleBatch
{
  std::size_t           n;          // number of particles in this batch
  std::size_t           nmetal;     // metals per particle in Zs row (0 if absent)

  std::vector<Real>     rho;        // density            [g/cm^3] after units conv.
  std::vector<Real>     u;          // specific internal energy [erg/g]
  std::vector<Real>     ne;         // electron abundance n_e/n_H (optional)
  std::vector<Real>     nh;         // neutral H fraction (optional)
  std::vector<Real>     Zs;         // flat (n * nmetal) metal mass fractions (optional)

  // -------------------------------------------------------------------
  //  resize -- size every present field to (n, nmetal).  Fields that
  //  the caller will not populate can be left as size 0 by clearing
  //  them after the call.
  //
  //  Inputs:  n_particles -- target particle count
  //           n_metal     -- per-particle metal count (0 for "no Zs")
  // -------------------------------------------------------------------
  void
  resize (std::size_t   n_particles,
          std::size_t   n_metal) noexcept
  {
    n      = n_particles;
    nmetal = n_metal;
    rho.resize (n);
    u.resize   (n);
    ne.resize  (n);
    nh.resize  (n);
    Zs.resize  (n * n_metal);
  }
};

}  // namespace mlhpc

#endif  // MLHPC_PARTICLE_BATCH_HPP
