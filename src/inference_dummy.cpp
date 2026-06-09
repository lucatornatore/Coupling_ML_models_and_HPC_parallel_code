// ---------------------------------------------------------------------
//  src/inference_dummy.cpp -- the deterministic dummy backend.
//
//  See `mlhpc/inference.hpp` for the rationale.  The implementation
//  is one trivially-vectorisable loop.  Coefficients are explicit
//  `Real` literals to keep the kernel signed-precision-clean (no
//  silent double promotion when Real == float).
// ---------------------------------------------------------------------

#include "mlhpc/inference.hpp"

#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <cstddef>

namespace mlhpc
{

void
DummyBackend::batch_inference (const ParticleBatch    &batch,
                               Real *__restrict        output) const
{
  const Real  a_rho = static_cast<Real> (0.5);
  const Real  a_u   = static_cast<Real> (0.25);

  const Real *__restrict  rho = batch.rho.data ();
  const Real *__restrict  u   = batch.u.data ();

  for (std::size_t i = 0; i < batch.n; ++i)
    {
      output [i] = a_rho * rho [i] + a_u * u [i];
    }
}

}  // namespace mlhpc
