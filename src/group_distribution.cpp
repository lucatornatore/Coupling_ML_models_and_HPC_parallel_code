// ---------------------------------------------------------------------
//  src/group_distribution.cpp -- implementation of GroupDistribution.
//
//  Algorithm:
//    1. Compute prefix sums of npart_per_subfile.
//    2. For each rank r, its "ideal start" in the global gas-particle
//       linear ordering is g_r = floor(r * Ngas / Ntasks).  Find the
//       sub-file index f such that prefix[f] <= g_r < prefix[f+1] via
//       binary search.  That's r's group.
//    3. Build the per-group member list as the ascending-rank ranks
//       assigned to each group.  The leader is members[0].
//    4. Fix-up: any non-empty sub-file with zero ranks gets a rank
//       stolen from the largest group with size > 1.
//    5. Inside each group the per-rank slice is `partition_uniform`
//       applied to the sub-file's particle count, indexed by
//       position-in-group.
//
//  Cost: O(Ntasks * log(num_files) + num_files).  At realistic scales
//  (Ntasks ~ 10⁴, num_files ~ 10²) this is microseconds; we don't
//  bother caching anything beyond what's needed for the accessors.
// ---------------------------------------------------------------------

#include "mlhpc/group_distribution.hpp"

#include "mlhpc/partition.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlhpc
{

namespace
{

// ---------------------------------------------------------------------
//  find_subfile_for_ideal_start
//
//  Largest f such that prefix[f] <= ideal_start.  Equivalent to
//  std::upper_bound(prefix, ideal_start) - prefix - 1, but written
//  explicitly to keep the intent obvious.
// ---------------------------------------------------------------------
int
find_subfile_for_ideal_start (const std::vector<std::size_t> &prefix,
                              std::size_t                      ideal_start) noexcept
{
  // upper_bound returns the first iterator pointing strictly past
  // ideal_start; the previous index is what we want.
  const auto  it  = std::upper_bound (prefix.begin (), prefix.end (), ideal_start);
  const int   idx = static_cast<int> (it - prefix.begin ()) - 1;
  // For r in [0, nranks) we have ideal_start in [0, Ngas), so idx is
  // always in [0, num_files); a defensive clamp keeps us safe under
  // pathological inputs.
  if (idx < 0)
    {
      return 0;
    }
  if (idx >= static_cast<int> (prefix.size ()) - 1)
    {
      return static_cast<int> (prefix.size ()) - 2;
    }
  return idx;
}

}  // anonymous namespace

// ---------------------------------------------------------------------
//  GroupDistribution (constructor)
// ---------------------------------------------------------------------
GroupDistribution::GroupDistribution (int                              nranks,
                                      const std::vector<std::size_t> &npart_per_subfile)
  : nranks_           (nranks)
  , npart_per_subfile_(npart_per_subfile)
  , ngas_total_       (0)
{
  const int num_files = static_cast<int> (npart_per_subfile.size ());

  if (num_files == 0)
    {
      throw std::invalid_argument (
          "GroupDistribution: npart_per_subfile is empty");
    }
  if (nranks <= 0)
    {
      throw std::invalid_argument (
          "GroupDistribution: nranks must be positive (got "
          + std::to_string (nranks) + ")");
    }
  if (nranks < num_files)
    {
      throw std::invalid_argument (
          "GroupDistribution: nranks (" + std::to_string (nranks)
          + ") < num_files (" + std::to_string (num_files)
          + "); cannot place one group per sub-file");
    }

  for (std::size_t n : npart_per_subfile)
    {
      ngas_total_ += n;
    }

  // ---- prefix sums -------------------------------------------------
  std::vector<std::size_t> prefix (num_files + 1, 0);
  for (int f = 0; f < num_files; ++f)
    {
      prefix [f + 1] = prefix [f] + npart_per_subfile [f];
    }

  // ---- step 2: per-rank group assignment via prefix-sum walk ------
  group_of_rank_.assign (nranks_, 0);
  for (int r = 0; r < nranks_; ++r)
    {
      // ideal_start = r * Ngas / Ntasks (integer math; safe because
      // r < nranks so r * Ngas fits as long as Ngas < 2^63 / nranks,
      // which is comfortably true for any realistic snapshot).
      const std::size_t ideal_start =
          (static_cast<std::size_t> (r) * ngas_total_)
          / static_cast<std::size_t> (nranks_);
      group_of_rank_ [r] = find_subfile_for_ideal_start (prefix, ideal_start);
    }

  // ---- step 3: build per-group member lists -----------------------
  members_per_group_.assign (static_cast<std::size_t> (num_files), {});
  for (int r = 0; r < nranks_; ++r)
    {
      members_per_group_ [static_cast<std::size_t> (group_of_rank_ [r])]
          .push_back (r);
    }

  // ---- step 4: fix-up empty-but-nonzero groups --------------------
  // A sub-file with npart > 0 must have at least one rank, otherwise
  // its particles would never be read.  We steal a rank from the
  // largest donor group (must have size > 1 so the donor itself
  // doesn't become empty).
  for (int f = 0; f < num_files; ++f)
    {
      if (npart_per_subfile_ [f] > 0
          && members_per_group_ [static_cast<std::size_t> (f)].empty ())
        {
          int donor      = -1;
          std::size_t mx = 1;
          for (int g = 0; g < num_files; ++g)
            {
              const std::size_t sz =
                  members_per_group_ [static_cast<std::size_t> (g)].size ();
              if (sz > mx)
                {
                  mx    = sz;
                  donor = g;
                }
            }
          if (donor < 0)
            {
              throw std::logic_error (
                  "GroupDistribution: cannot rescue empty non-zero "
                  "sub-file " + std::to_string (f)
                  + " (no donor group has size > 1)");
            }

          // Donate the *highest-numbered* rank: it is the one whose
          // assignment was most "marginal" (its ideal_start sits at
          // the far end of the donor's range) and moving it preserves
          // the ascending-rank ordering of both groups.
          const int stolen =
              members_per_group_ [static_cast<std::size_t> (donor)].back ();
          members_per_group_ [static_cast<std::size_t> (donor)].pop_back ();
          members_per_group_ [static_cast<std::size_t> (f)].push_back (stolen);
          group_of_rank_ [stolen] = f;
        }
    }
}

// =====================================================================
//  Accessors
// =====================================================================

int
GroupDistribution::group_of (int rank) const noexcept
{
  assert (rank >= 0 && rank < nranks_);
  return group_of_rank_ [static_cast<std::size_t> (rank)];
}

int
GroupDistribution::leader_of (int rank) const noexcept
{
  return members_per_group_ [static_cast<std::size_t> (group_of (rank))]
      .front ();
}

int
GroupDistribution::position_in_group (int rank) const noexcept
{
  const auto &mem = members_per_group_
      [static_cast<std::size_t> (group_of (rank))];
  const auto  it  = std::find (mem.begin (), mem.end (), rank);
  assert (it != mem.end ());
  return static_cast<int> (it - mem.begin ());
}

int
GroupDistribution::group_size (int rank) const noexcept
{
  return static_cast<int> (
      members_per_group_ [static_cast<std::size_t> (group_of (rank))].size ());
}

GlobalSlice
GroupDistribution::slice_in_subfile (int rank) const noexcept
{
  const int          f   = group_of (rank);
  const int          pos = position_in_group (rank);
  const int          gs  = group_size (rank);
  return partition_uniform (npart_per_subfile_ [static_cast<std::size_t> (f)],
                            pos, gs);
}

const std::vector<int> &
GroupDistribution::group_members (int group) const noexcept
{
  return members_per_group_ [static_cast<std::size_t> (group)];
}

std::size_t
GroupDistribution::npart_in_group (int group) const noexcept
{
  return npart_per_subfile_ [static_cast<std::size_t> (group)];
}

}  // namespace mlhpc
