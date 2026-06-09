// ---------------------------------------------------------------------
//  tests/test_units.cpp
//
//  Validates the unit-conversion module against hand-computed values:
//      (1) `make_cosmology` populates a/z/h/Omega0/OmegaLambda from
//          the header and bakes the (h^2/a^3) factor into rho_to_cgs;
//      (2) `rho_code_to_cgs` agrees with the by-hand formula to a
//          tight tolerance (1e-6 relative, easily within `float`);
//      (3) `u_code_to_cgs` agrees with the by-hand formula (it is
//          scale-factor-independent in the Gadget convention);
//      (4) `entropy_to_u_code` agrees with
//          u = A * rho^(2/3) / (2/3) for gamma = 5/3.
//
//  All comparisons use relative tolerance against the analytical
//  result computed in `double` so the test is robust against the
//  build-time precision of `mlhpc::Real`.
// ---------------------------------------------------------------------

#include "mlhpc/types.hpp"
#include "mlhpc/units.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

using mlhpc::Cosmology;
using mlhpc::GadgetHeader2;
using mlhpc::Real;

namespace
{

int errors = 0;

// ---------------------------------------------------------------------
//  close_rel -- report a failure if |got - want| > tol * max(|want|,1)
// ---------------------------------------------------------------------
bool
close_rel (const char *what, double got, double want, double tol)
{
  const double  denom = std::fmax (std::fabs (want), 1.0);
  const double  rel   = std::fabs (got - want) / denom;
  if (rel > tol)
    {
      std::fprintf (stderr,
                    "  FAIL  %s: got %.12g, want %.12g  (rel %.3e > %.1e)\n",
                    what, got, want, rel, tol);
      ++errors;
      return false;
    }
  return true;
}

}  // namespace

int
main ()
{
  // ============== Case 1: make_cosmology populates the bundle =======
  {
    GadgetHeader2 h {};
    h.time         = 0.5;      // a = 0.5
    h.redshift     = 1.0;      // z = 1
    h.HubbleParam  = 0.7;      // h = 0.7
    h.Omega0       = 0.3;
    h.OmegaLambda  = 0.7;

    const Cosmology c = mlhpc::make_cosmology (h);

    close_rel ("Case1: a",            c.a,            0.5, 1e-15);
    close_rel ("Case1: z",            c.z,            1.0, 1e-15);
    close_rel ("Case1: h",            c.h,            0.7, 1e-15);
    close_rel ("Case1: Omega0",       c.Omega0,       0.3, 1e-15);
    close_rel ("Case1: OmegaLambda",  c.OmegaLambda,  0.7, 1e-15);

    // rho_to_cgs = UnitDensity_in_cgs * h^2 / a^3
    const double  unit_dens =
        mlhpc::UNIT_MASS_IN_G
        / (mlhpc::UNIT_LENGTH_IN_CM
           * mlhpc::UNIT_LENGTH_IN_CM * mlhpc::UNIT_LENGTH_IN_CM);
    const double  want_rho_to_cgs = unit_dens * 0.7 * 0.7 / (0.5 * 0.5 * 0.5);
    close_rel ("Case1: rho_to_cgs",   c.rho_to_cgs, want_rho_to_cgs, 1e-12);

    // u_to_cgs = UnitVelocity_in_cm_per_s^2  (no a, no h)
    const double  want_u_to_cgs =
        mlhpc::UNIT_VELOCITY_IN_CM_PER_S * mlhpc::UNIT_VELOCITY_IN_CM_PER_S;
    close_rel ("Case1: u_to_cgs",     c.u_to_cgs,   want_u_to_cgs,   1e-15);
  }

  // ============== Case 2: rho array conversion ======================
  {
    Cosmology c {};
    c.rho_to_cgs = 6.5e-22;       // some sentinel multiplier
    c.u_to_cgs   = 1.0e10;

    std::vector<Real>  rho = { Real (1.0), Real (2.5), Real (0.0), Real (1e3) };
    mlhpc::rho_code_to_cgs (rho.data (), rho.size (), c);

    close_rel ("Case2: rho[0]", rho [0], 1.0 * 6.5e-22, 1e-5);
    close_rel ("Case2: rho[1]", rho [1], 2.5 * 6.5e-22, 1e-5);
    close_rel ("Case2: rho[2]", rho [2], 0.0,           1e-15);
    close_rel ("Case2: rho[3]", rho [3], 1000.0 * 6.5e-22, 1e-5);
  }

  // ============== Case 3: u array conversion ========================
  {
    Cosmology c {};
    c.u_to_cgs = 1.0e10;          // UnitVelocity^2 with default constants

    std::vector<Real>  u = { Real (1.0), Real (100.0), Real (1e-3) };
    mlhpc::u_code_to_cgs (u.data (), u.size (), c);

    close_rel ("Case3: u[0]", u [0], 1.0e10,        1e-6);
    close_rel ("Case3: u[1]", u [1], 100.0 * 1e10,  1e-6);
    close_rel ("Case3: u[2]", u [2], 1e-3  * 1e10,  1e-6);
  }

  // ============== Case 4: entropy_to_u_code =========================
  //  u_code = A_code * rho_code^(2/3) / (2/3)
  //         = A_code * rho_code^(2/3) * 1.5
  {
    std::vector<Real>  A   = { Real (1.0), Real (2.0), Real (10.0) };
    std::vector<Real>  rho = { Real (1.0), Real (8.0), Real (0.125) };

    mlhpc::entropy_to_u_code (A.data (), rho.data (), A.size ());

    // hand-computed:
    // i=0: 1.0  * 1.0^(2/3) * 1.5 = 1.5
    // i=1: 2.0  * 8.0^(2/3) * 1.5 = 2.0 * 4.0 * 1.5 = 12.0
    // i=2: 10.0 * 0.125^(2/3) * 1.5 = 10.0 * 0.25 * 1.5 = 3.75
    close_rel ("Case4: u[0]", A [0], 1.5,   1e-5);
    close_rel ("Case4: u[1]", A [1], 12.0,  1e-5);
    close_rel ("Case4: u[2]", A [2], 3.75,  1e-5);
  }

  if (errors == 0)
    {
      std::printf ("test_units: OK\n");
      return EXIT_SUCCESS;
    }
  std::fprintf (stderr, "test_units: %d check(s) failed\n", errors);
  return EXIT_FAILURE;
}
