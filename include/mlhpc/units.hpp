// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/units.hpp -- code-unit -> physical/CGS conversion.
//
//  Provides:
//    * Gadget code-unit constants (UnitLength_in_cm, UnitMass_in_g,
//      UnitVelocity_in_cm_per_s, derived UnitDensity_in_cgs and
//      UnitEnergy_in_cgs).
//    * `make_cosmology(header)` that builds a `Cosmology` struct from
//      a `GadgetHeader2`, pre-baking the (h, a)-dependent factors so
//      conversion at use-time is a single multiply per particle.
//    * In-place array conversions for the two cooling fields:
//        rho_code_to_cgs : comoving code density -> physical g/cm^3.
//        u_code_to_cgs   : code specific internal energy -> erg/g.
//    * `entropy_to_u_code` for snapshots where the `U ` block is
//      actually entropy A (header.flag_entropy_instead_u == 1):
//      u = A * rho^(gamma-1) / (gamma-1), gamma = 5/3.
//
//  Formulas (see ioformat.md sec. 3):
//      rho_phys_cgs = rho_code * UnitDensity_in_cgs * h^2 / a^3
//      u_phys_cgs   = u_code   * UnitVelocity_in_cm_per_s^2
//      u_code       = A_code   * rho_code^(gamma-1) / (gamma-1)
//
//  The cosmological factors (`h^2/a^3`) are folded into
//  `Cosmology::rho_to_cgs`, so the per-particle hot path is one
//  multiply.  The energy conversion is scale-factor-independent
//  in the Gadget convention.
// ---------------------------------------------------------------------

#ifndef MLHPC_UNITS_HPP
#define MLHPC_UNITS_HPP

#include "mlhpc/types.hpp"

#include <cstddef>

namespace mlhpc
{

// ---------------------------------------------------------------------
//  Gadget code-unit constants (the default values used by OpenGADGET3
//  for cosmological runs).  All are exact-as-given here so the unit
//  tests in `tests/test_units.cpp` can compare against hand-computed
//  values without precision-dependent slop.
// ---------------------------------------------------------------------
constexpr double  UNIT_LENGTH_IN_CM        = 3.085678e21;     // 1 kpc
constexpr double  UNIT_MASS_IN_G           = 1.989e43;        // 1e10 Msun
constexpr double  UNIT_VELOCITY_IN_CM_PER_S = 1.0e5;          // 1 km/s

constexpr double  UNIT_TIME_IN_S          =
    UNIT_LENGTH_IN_CM / UNIT_VELOCITY_IN_CM_PER_S;

constexpr double  UNIT_DENSITY_IN_CGS     =
    UNIT_MASS_IN_G
    / (UNIT_LENGTH_IN_CM * UNIT_LENGTH_IN_CM * UNIT_LENGTH_IN_CM);

constexpr double  UNIT_ENERGY_IN_CGS      =
    UNIT_MASS_IN_G * UNIT_VELOCITY_IN_CM_PER_S * UNIT_VELOCITY_IN_CM_PER_S;

constexpr double  GAMMA                   = 5.0 / 3.0;        // ideal gas

// ---------------------------------------------------------------------
//  make_cosmology
//
//  Build a `Cosmology` cache from a snapshot header.  The two
//  `*_to_cgs` factors are pre-multiplied so that a single
//  `value_code * factor` converts a code-unit field to CGS at
//  use-time.
//
//  Inputs:  h -- GadgetHeader2 (already in host byte order)
//  Returns: Cosmology with all fields populated.
//
//  Pre-conditions:  a > 0 and h > 0 (cosmological run).  For a
//  non-cosmological run the formulas degenerate (a=h=1) and the
//  returned conversion factors are still meaningful.
// ---------------------------------------------------------------------
Cosmology
make_cosmology (const GadgetHeader2 &h) noexcept;

// ---------------------------------------------------------------------
//  rho_code_to_cgs
//
//  Convert `n` comoving code-unit densities to physical CGS in place:
//      rho_phys_cgs[i] = rho_code[i] * cosmo.rho_to_cgs
//
//  cosmo.rho_to_cgs already folds in UnitDensity_in_cgs * h^2 / a^3.
// ---------------------------------------------------------------------
void
rho_code_to_cgs (Real *__restrict       rho,        // in/out: density values
                 std::size_t            n,          // count
                 const Cosmology       &cosmo) noexcept;

// ---------------------------------------------------------------------
//  u_code_to_cgs
//
//  Convert `n` specific internal energies from code units to physical
//  CGS (erg/g) in place:
//      u_phys_cgs[i] = u_code[i] * cosmo.u_to_cgs
// ---------------------------------------------------------------------
void
u_code_to_cgs (Real *__restrict        u,           // in/out
               std::size_t             n,
               const Cosmology        &cosmo) noexcept;

// ---------------------------------------------------------------------
//  entropy_to_u_code
//
//  For snapshots where header.flag_entropy_instead_u == 1, the `U `
//  block stores the entropy A.  Convert it to specific internal
//  energy in code units, in place:
//      u_code[i] = A_code[i] * rho_code[i]^(GAMMA-1) / (GAMMA-1)
//
//  Inputs:  u_or_A  -- in: entropy A in code units; out: u in code units
//           rho_code -- comoving density in code units
//           n       -- count
//
//  Both arrays must have at least `n` elements.  The two buffers must
//  not overlap (asserted via __restrict; UB if violated).
// ---------------------------------------------------------------------
void
entropy_to_u_code (Real       *__restrict  u_or_A,    // in: A, out: u (code units)
                   const Real *__restrict  rho_code,  // comoving density
                   std::size_t             n) noexcept;

}  // namespace mlhpc

#endif  // MLHPC_UNITS_HPP
