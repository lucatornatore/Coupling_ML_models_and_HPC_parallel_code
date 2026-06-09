// ---------------------------------------------------------------------
//  tests/test_header_layout.cpp
//
//  Verifies that the GadgetHeader2 in-memory layout matches the
//  on-disk Format-2 header exactly: total size 256 bytes and every
//  field at its expected byte offset.  A mismatch here would silently
//  corrupt every snapshot the harness reads, so the assertions are
//  hard failures.
// ---------------------------------------------------------------------

#include "mlhpc/types.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

using mlhpc::GadgetHeader2;

namespace
{

// Tiny assertion helper that reports the failing field name on stderr
// and increments an error count, instead of aborting immediately --
// this way one run lists every offset mismatch in one go.
template <typename T1, typename T2>
void
expect_eq (const char *what, T1 got, T2 want, int &errors)
{
  if (static_cast<std::size_t> (got) != static_cast<std::size_t> (want))
    {
      std::fprintf (stderr,
                    "  FAIL  %-32s : got %zu, expected %zu\n",
                    what,
                    static_cast<std::size_t> (got),
                    static_cast<std::size_t> (want));
      ++errors;
    }
}

}  // namespace

int
main ()
{
  int errors = 0;

  expect_eq ("sizeof(GadgetHeader2)",
             sizeof (GadgetHeader2), 256u, errors);

  expect_eq ("offsetof(npart)",
             offsetof (GadgetHeader2, npart),                   0u, errors);
  expect_eq ("offsetof(mass)",
             offsetof (GadgetHeader2, mass),                   24u, errors);
  expect_eq ("offsetof(time)",
             offsetof (GadgetHeader2, time),                   72u, errors);
  expect_eq ("offsetof(redshift)",
             offsetof (GadgetHeader2, redshift),               80u, errors);
  expect_eq ("offsetof(flag_sfr)",
             offsetof (GadgetHeader2, flag_sfr),               88u, errors);
  expect_eq ("offsetof(flag_feedback)",
             offsetof (GadgetHeader2, flag_feedback),          92u, errors);
  expect_eq ("offsetof(npartTotal)",
             offsetof (GadgetHeader2, npartTotal),             96u, errors);
  expect_eq ("offsetof(flag_cooling)",
             offsetof (GadgetHeader2, flag_cooling),          120u, errors);
  expect_eq ("offsetof(num_files)",
             offsetof (GadgetHeader2, num_files),             124u, errors);
  expect_eq ("offsetof(BoxSize)",
             offsetof (GadgetHeader2, BoxSize),               128u, errors);
  expect_eq ("offsetof(Omega0)",
             offsetof (GadgetHeader2, Omega0),                136u, errors);
  expect_eq ("offsetof(OmegaLambda)",
             offsetof (GadgetHeader2, OmegaLambda),           144u, errors);
  expect_eq ("offsetof(HubbleParam)",
             offsetof (GadgetHeader2, HubbleParam),           152u, errors);
  expect_eq ("offsetof(flag_stellarage)",
             offsetof (GadgetHeader2, flag_stellarage),       160u, errors);
  expect_eq ("offsetof(flag_metals)",
             offsetof (GadgetHeader2, flag_metals),           164u, errors);
  expect_eq ("offsetof(npartTotalHighWord)",
             offsetof (GadgetHeader2, npartTotalHighWord),    168u, errors);
  expect_eq ("offsetof(flag_entropy_instead_u)",
             offsetof (GadgetHeader2, flag_entropy_instead_u),192u, errors);
  expect_eq ("offsetof(flag_doubleprecision)",
             offsetof (GadgetHeader2, flag_doubleprecision),  196u, errors);
  expect_eq ("offsetof(flag_ic_info)",
             offsetof (GadgetHeader2, flag_ic_info),          200u, errors);
  expect_eq ("offsetof(lpt_scalingfactor)",
             offsetof (GadgetHeader2, lpt_scalingfactor),     204u, errors);
  expect_eq ("offsetof(fill)",
             offsetof (GadgetHeader2, fill),                  208u, errors);

  if (errors == 0)
    {
      std::printf ("test_header_layout: OK\n");
      return EXIT_SUCCESS;
    }

  std::fprintf (stderr,
                "test_header_layout: %d offset mismatch(es)\n", errors);
  return EXIT_FAILURE;
}
