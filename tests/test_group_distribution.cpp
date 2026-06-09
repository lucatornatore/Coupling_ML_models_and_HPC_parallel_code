// ---------------------------------------------------------------------
//  tests/test_group_distribution.cpp
//
//  Non-MPI unit test for `GroupDistribution`.  Validates:
//      * the canonical prefix-sum walk on a balanced example;
//      * the within-group partition reuses `partition_uniform`
//        correctly;
//      * the single-sub-file case (one group, all ranks in it);
//      * the empty-but-nonzero-sub-file fix-up (a rank gets stolen
//        from the largest group);
//      * `nranks < num_files` is rejected.
// ---------------------------------------------------------------------

#include "mlhpc/group_distribution.hpp"
#include "mlhpc/partition.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using mlhpc::GroupDistribution;
using mlhpc::GlobalSlice;
using mlhpc::partition_uniform;

namespace
{

int errors = 0;

template <typename T>
void
expect_eq (const char *what, T got, T want)
{
  if (!(got == want))
    {
      std::fprintf (stderr, "  FAIL  %s (mismatch)\n", what);
      ++errors;
    }
}

void
expect_true (const char *what, bool cond)
{
  if (!cond)
    {
      std::fprintf (stderr, "  FAIL  %s\n", what);
      ++errors;
    }
}

}  // namespace

int
main ()
{
  // ============== Case 1: canonical example ==========================
  //   3 sub-files [7, 8, 5], 4 ranks.  Ngas = 20.  target = 5.
  //   ideal_starts = [0, 5, 10, 15].  prefix = [0, 7, 15, 20].
  //   Walk: r0->0, r1->0 (5<7), r2->1 (7<=10<15), r3->2 (15<=15<20).
  {
    GroupDistribution gd (4, { 7u, 8u, 5u });

    expect_eq ("Case1: num_groups",  gd.num_groups (), 3);
    expect_eq ("Case1: num_ranks",   gd.num_ranks (),  4);
    expect_eq <std::size_t> ("Case1: ngas_total",  gd.ngas_total (), 20u);

    expect_eq ("Case1: group_of(0)", gd.group_of (0), 0);
    expect_eq ("Case1: group_of(1)", gd.group_of (1), 0);
    expect_eq ("Case1: group_of(2)", gd.group_of (2), 1);
    expect_eq ("Case1: group_of(3)", gd.group_of (3), 2);

    expect_eq ("Case1: leader_of(0)", gd.leader_of (0), 0);
    expect_eq ("Case1: leader_of(1)", gd.leader_of (1), 0);
    expect_eq ("Case1: leader_of(2)", gd.leader_of (2), 2);
    expect_eq ("Case1: leader_of(3)", gd.leader_of (3), 3);

    expect_eq ("Case1: position 0",  gd.position_in_group (0), 0);
    expect_eq ("Case1: position 1",  gd.position_in_group (1), 1);
    expect_eq ("Case1: position 2",  gd.position_in_group (2), 0);
    expect_eq ("Case1: position 3",  gd.position_in_group (3), 0);

    expect_eq ("Case1: group_size 0", gd.group_size (0), 2);
    expect_eq ("Case1: group_size 2", gd.group_size (2), 1);

    // Slices: group 0 has 7 particles split [0..4)/[4..7) for pos 0/1;
    //         group 1 has 8 particles, single rank: [0..8);
    //         group 2 has 5 particles, single rank: [0..5).
    const GlobalSlice s0 = gd.slice_in_subfile (0);
    const GlobalSlice s1 = gd.slice_in_subfile (1);
    const GlobalSlice s2 = gd.slice_in_subfile (2);
    const GlobalSlice s3 = gd.slice_in_subfile (3);
    expect_eq <std::size_t> ("Case1: s0.gstart", s0.gstart, 0u);
    expect_eq <std::size_t> ("Case1: s0.count",  s0.count (), 4u);
    expect_eq <std::size_t> ("Case1: s1.gstart", s1.gstart, 4u);
    expect_eq <std::size_t> ("Case1: s1.count",  s1.count (), 3u);
    expect_eq <std::size_t> ("Case1: s2.gstart", s2.gstart, 0u);
    expect_eq <std::size_t> ("Case1: s2.count",  s2.count (), 8u);
    expect_eq <std::size_t> ("Case1: s3.count",  s3.count (), 5u);
  }

  // ============== Case 2: single sub-file, 4 ranks ===================
  //   All 4 ranks in group 0.  Within-group: partition_uniform(100,p,4).
  {
    GroupDistribution gd (4, { 100u });
    expect_eq ("Case2: num_groups", gd.num_groups (), 1);
    for (int r = 0; r < 4; ++r)
      {
        expect_eq ("Case2: all in group 0", gd.group_of (r), 0);
        expect_eq ("Case2: all share leader 0", gd.leader_of (r), 0);
      }
    expect_eq ("Case2: group_size", gd.group_size (0), 4);
    expect_eq ("Case2: position 3", gd.position_in_group (3), 3);

    // Slice counts must cover the whole sub-file.
    std::size_t sum = 0;
    for (int r = 0; r < 4; ++r) sum += gd.slice_in_subfile (r).count ();
    expect_eq <std::size_t> ("Case2: slices cover all", sum, 100u);
  }

  // ============== Case 3: empty-group fix-up =========================
  //   3 sub-files [30, 1, 1], 3 ranks.  Ngas = 32.
  //   target = 10 (integer), ideal_starts = [0, 10, 21].
  //   prefix = [0, 30, 31, 32].
  //   Walk: r0->0, r1->0, r2->0.
  //   Groups 1 and 2 both have npart>0 but no ranks.
  //   After fix-up: stealing assigns one rank each to groups 1 and 2,
  //   leaving group 0 with one rank (the originals' lowest survivor).
  {
    GroupDistribution gd (3, { 30u, 1u, 1u });
    // Every group must have >= 1 member because all sub-files are
    // non-empty.
    expect_true ("Case3: group 0 non-empty", !gd.group_members (0).empty ());
    expect_true ("Case3: group 1 non-empty", !gd.group_members (1).empty ());
    expect_true ("Case3: group 2 non-empty", !gd.group_members (2).empty ());

    // The sum of all group sizes must equal nranks.
    int sum = 0;
    for (int g = 0; g < gd.num_groups (); ++g)
      {
        sum += static_cast<int> (gd.group_members (g).size ());
      }
    expect_eq ("Case3: total ranks across groups", sum, 3);
  }

  // ============== Case 4: nranks < num_files must throw ==============
  {
    bool threw = false;
    try
      {
        GroupDistribution gd (2, { 1u, 1u, 1u });
      }
    catch (const std::invalid_argument &)
      {
        threw = true;
      }
    expect_true ("Case4: nranks < num_files rejected", threw);
  }

  if (errors == 0)
    {
      std::printf ("test_group_distribution: OK\n");
      return EXIT_SUCCESS;
    }
  std::fprintf (stderr,
                "test_group_distribution: %d check(s) failed\n", errors);
  return EXIT_FAILURE;
}
