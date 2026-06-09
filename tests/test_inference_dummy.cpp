// ---------------------------------------------------------------------
//  tests/test_inference_dummy.cpp
//
//  Bit-exact verification of `DummyBackend::batch_inference`.  The
//  dummy is a fixed linear combination
//      output[i] = 0.5 * rho[i] + 0.25 * u[i]
//  so for chosen rho[i], u[i] the expected output is hand-computable
//  and (because both coefficients are exactly representable in IEEE
//  binary floating-point) the test compares with zero tolerance.
//
//  Also verifies that an empty batch is a no-op (no UB on size 0).
// ---------------------------------------------------------------------

#include "mlhpc/inference.hpp"
#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

using mlhpc::DummyBackend;
using mlhpc::ParticleBatch;
using mlhpc::Real;

namespace
{

int errors = 0;

void
expect_eq_real (const char *what, Real got, Real want)
{
  if (!(got == want))
    {
      std::fprintf (stderr,
                    "  FAIL  %s: got %.9g, want %.9g\n",
                    what,
                    static_cast<double> (got),
                    static_cast<double> (want));
      ++errors;
    }
}

}  // namespace

int
main ()
{
  // ============== Non-empty batch, exact arithmetic =================
  {
    ParticleBatch  batch {};
    batch.resize (4, 0);
    batch.rho = { Real (1.0), Real (2.0), Real (3.0), Real (4.0) };
    batch.u   = { Real (8.0), Real (16.0), Real (32.0), Real (64.0) };

    std::vector<Real>  out (batch.n);
    DummyBackend       backend;
    backend.batch_inference (batch, out.data ());

    // 0.5 * 1 + 0.25 * 8  =  0.5 + 2   = 2.5
    // 0.5 * 2 + 0.25 * 16 =  1   + 4   = 5.0
    // 0.5 * 3 + 0.25 * 32 =  1.5 + 8   = 9.5
    // 0.5 * 4 + 0.25 * 64 =  2   + 16  = 18.0
    expect_eq_real ("out[0]", out [0], Real (2.5));
    expect_eq_real ("out[1]", out [1], Real (5.0));
    expect_eq_real ("out[2]", out [2], Real (9.5));
    expect_eq_real ("out[3]", out [3], Real (18.0));
  }

  // ============== Empty batch -> no-op ==============================
  {
    ParticleBatch  batch {};
    batch.resize (0, 0);
    Real           sentinel = Real (42.0);  // must remain untouched
    DummyBackend   backend;
    backend.batch_inference (batch, &sentinel);
    expect_eq_real ("sentinel survives empty batch", sentinel, Real (42.0));
  }

  if (errors == 0)
    {
      std::printf ("test_inference_dummy: OK\n");
      return EXIT_SUCCESS;
    }
  std::fprintf (stderr,
                "test_inference_dummy: %d check(s) failed\n", errors);
  return EXIT_FAILURE;
}
