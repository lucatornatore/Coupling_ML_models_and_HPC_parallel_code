// ---------------------------------------------------------------------
//  src/units.cpp -- implementation of the unit-conversion module.
//
//  See `mlhpc/units.hpp` for the contract.  All inner loops are tight
//  enough that the compiler should auto-vectorise them under -O2 once
//  the `restrict` qualifiers on the array pointers communicate that
//  the buffers do not alias.
// ---------------------------------------------------------------------

#include "mlhpc/units.hpp"

#include "mlhpc/types.hpp"

#include <cmath>
#include <cstddef>

namespace mlhpc
{

// ---------------------------------------------------------------------
//  make_cosmology
//
//  Populate the seven fields of `Cosmology` from a snapshot header.
//  The pre-baked factors are computed in `double` and cast on use to
//  `Real` so the conversion is consistent regardless of whether the
//  build is single- or double-precision (FEATURE_DOUBLE).
// ---------------------------------------------------------------------
Cosmology
make_cosmology (const GadgetHeader2 &h) noexcept
{
  Cosmology c {};
  c.a            = h.time;
  c.z            = h.redshift;
  c.h            = h.HubbleParam;
  c.Omega0       = h.Omega0;
  c.OmegaLambda  = h.OmegaLambda;

  // h^2 / a^3 unwinds the (kpc/h)^-3 comoving volume into a physical
  // cm^-3.  Folding it into rho_to_cgs makes the hot path one mul.
  const double  a3  = c.a * c.a * c.a;
  c.rho_to_cgs       = UNIT_DENSITY_IN_CGS * (c.h * c.h) / a3;

  // The specific internal energy has units of velocity^2 and is
  // stored peculiar (no extra scale-factor factor) in Gadget.
  c.u_to_cgs         = UNIT_VELOCITY_IN_CM_PER_S
                       * UNIT_VELOCITY_IN_CM_PER_S;

  return c;
}

// ---------------------------------------------------------------------
//  rho_code_to_cgs
//
//  One multiply per element with the pre-baked factor.  Loop is
//  trivially vectorisable.
// ---------------------------------------------------------------------
void
rho_code_to_cgs (Real *__restrict   rho,
                 std::size_t        n,
                 const Cosmology   &cosmo) noexcept
{
  const Real  factor = static_cast<Real> (cosmo.rho_to_cgs);
  for (std::size_t i = 0; i < n; ++i)
    {
      rho [i] *= factor;
    }
}

// ---------------------------------------------------------------------
//  u_code_to_cgs
// ---------------------------------------------------------------------
void
u_code_to_cgs (Real *__restrict    u,
               std::size_t         n,
               const Cosmology    &cosmo) noexcept
{
  const Real  factor = static_cast<Real> (cosmo.u_to_cgs);
  for (std::size_t i = 0; i < n; ++i)
    {
      u [i] *= factor;
    }
}

// ---------------------------------------------------------------------
//  entropy_to_u_code
//
//  u = A * rho^(gamma-1) / (gamma-1).  gamma-1 = 2/3, so the exponent
//  is 2/3 and the divisor is 2/3.  Using std::pow keeps things
//  portable; if profiling later shows it's hot, replace with cbrt of
//  rho^2 (since rho^(2/3) = cbrt(rho^2)).
// ---------------------------------------------------------------------
void
entropy_to_u_code (Real       *__restrict  u_or_A,
                   const Real *__restrict  rho_code,
                   std::size_t             n) noexcept
{
  const Real  exponent = static_cast<Real> (GAMMA - 1.0);   // 2/3
  const Real  inv_gm1  = static_cast<Real> (1.0 / (GAMMA - 1.0));   // 3/2

  for (std::size_t i = 0; i < n; ++i)
    {
      const Real  rho_pow = std::pow (rho_code [i], exponent);
      u_or_A [i] = u_or_A [i] * rho_pow * inv_gm1;
    }
}

}  // namespace mlhpc
