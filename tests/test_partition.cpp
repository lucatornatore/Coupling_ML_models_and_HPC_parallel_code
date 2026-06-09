// ---------------------------------------------------------------------
//  tests/test_partition.cpp
//
//  Unit tests for `partition_uniform`.  The properties checked are:
//      (1) the slices are contiguous and cover [0, total) exactly,
//      (2) each rank's count differs by at most 1 from the others,
//      (3) the count formula matches `base + (rank < rem ? 1 : 0)`,
//      (4) edge cases: total = 0, total < nranks, nranks = 1,
//          and the rank counts sum to total in every case.
// ---------------------------------------------------------------------

#include "mlhpc/partition.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

using mlhpc::GlobalSlice;
using mlhpc::partition_uniform;

namespace
{

int errors = 0;

void
fail (const char *what)
{
  std::fprintf (stderr, "  FAIL  %s\n", what);
  ++errors;
}

// ---------------------------------------------------------------------
//  check_partition_covers_total
//
//  Sweep all ranks, verifying that the union of slices is exactly
//  [0, total), that adjacent slices abut without gap or overlap,
//  and that the rank counts differ by at most 1.
// ---------------------------------------------------------------------
void
check_partition_covers_total (std::size_t total,
                              int          nranks)
{
  std::vector<GlobalSlice>  slices;
  slices.reserve (static_cast<std::size_t> (nranks));

  for (int r = 0; r < nranks; ++r)
    {
      slices.push_back (partition_uniform (total, r, nranks));
    }

  if (slices.front ().gstart != 0)
    {
      fail ("first slice does not start at 0");
    }
  if (slices.back ().gend != total)
    {
      fail ("last slice does not end at total");
    }

  std::size_t sum = 0;
  for (int r = 0; r < nranks; ++r)
    {
      sum += slices [r].count ();
      if (r > 0 && slices [r].gstart != slices [r - 1].gend)
        {
          fail ("non-contiguous slices");
        }
    }
  if (sum != total)
    {
      fail ("slice counts do not sum to total");
    }

  // max count - min count <= 1
  std::size_t mn = slices.front ().count ();
  std::size_t mx = slices.front ().count ();
  for (const auto &s : slices)
    {
      if (s.count () < mn) mn = s.count ();
      if (s.count () > mx) mx = s.count ();
    }
  if (mx - mn > 1)
    {
      fail ("imbalance > 1 between heaviest and lightest rank");
    }
}

}  // namespace

int
main ()
{
  // --- spot-check the formula on a small case --------------------
  // total=10, nranks=3: base=3, rem=1. ranks: (4, 3, 3).
  {
    const auto s0 = partition_uniform (10, 0, 3);
    const auto s1 = partition_uniform (10, 1, 3);
    const auto s2 = partition_uniform (10, 2, 3);
    if (s0.gstart != 0 || s0.gend != 4) fail ("10/3 rank 0");
    if (s1.gstart != 4 || s1.gend != 7) fail ("10/3 rank 1");
    if (s2.gstart != 7 || s2.gend != 10) fail ("10/3 rank 2");
  }

  // --- edge cases ------------------------------------------------
  check_partition_covers_total ( 0, 1);
  check_partition_covers_total ( 0, 4);
  check_partition_covers_total ( 1, 1);
  check_partition_covers_total ( 1, 4);   // total < nranks
  check_partition_covers_total ( 7, 1);
  check_partition_covers_total ( 7, 7);
  check_partition_covers_total ( 7, 8);
  check_partition_covers_total (100, 4);
  check_partition_covers_total (1'000'000, 17);
  check_partition_covers_total (123'456'789, 31);

  if (errors == 0)
    {
      std::printf ("test_partition: OK\n");
      return EXIT_SUCCESS;
    }
  std::fprintf (stderr, "test_partition: %d check(s) failed\n", errors);
  return EXIT_FAILURE;
}
