// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/group_distribution.hpp -- MPI-rank-to-sub-file grouping.
//
//  Given the gas-particle counts per sub-file and the number of MPI
//  ranks, compute, for every rank, which sub-file's "group" it belongs
//  to, who the group's leader is, the rank's position inside the
//  group, and the contiguous slice of the sub-file's gas particles
//  that the rank is responsible for.
//
//  The class is pure arithmetic: no MPI calls, no file I/O.  Every
//  rank constructs an identical `GroupDistribution` from the
//  broadcast-shared `npart_per_subfile` array and looks up its own
//  info.  This is the strategy specified in `tasks.md` (session 5):
//
//    * One group per sub-file (so num_groups == num_files).
//    * Group sizes vary so that particles-per-rank are balanced as
//      well as the "one sub-file per group" constraint allows.
//    * The walk: each rank's "ideal start" g_r = r * Ngas / Ntasks
//      lands in some sub-file's prefix-sum interval; that determines
//      its group.
//    * Inside each group the sub-file's particles are split among
//      members with the usual uniform-partition (±1).
//
//  Edge-case handling (documented choices):
//    * If a non-empty sub-file ends up with no ranks (this can happen
//      when a sub-file is much smaller than `Ngas/Ntasks` and falls
//      between two consecutive `ideal_start`s) the constructor steals
//      one rank from the largest group with size > 1.
//    * If `Ntasks < num_files` the constructor throws: we cannot
//      satisfy the "one group per sub-file" requirement with fewer
//      ranks than sub-files.
// ---------------------------------------------------------------------

#ifndef MLHPC_GROUP_DISTRIBUTION_HPP
#define MLHPC_GROUP_DISTRIBUTION_HPP

#include "mlhpc/partition.hpp"

#include <cstddef>
#include <vector>

namespace mlhpc
{

class GroupDistribution
{
public:

  // -------------------------------------------------------------------
  //  Compute the full assignment.
  //
  //  Inputs:  nranks             -- MPI world size  (must be >= num_files)
  //           npart_per_subfile  -- gas particles per sub-file
  //  Throws:  std::invalid_argument on `nranks < num_files` or empty
  //           input; std::logic_error if no donor is available when a
  //           non-empty sub-file would otherwise get no rank.
  // -------------------------------------------------------------------
  GroupDistribution (int                              nranks,
                     const std::vector<std::size_t> &npart_per_subfile);

  // ----- snapshot-wide accessors ------------------------------------
  int           num_groups  () const noexcept { return static_cast<int> (members_per_group_.size ()); }
  int           num_ranks   () const noexcept { return nranks_; }
  std::size_t   ngas_total  () const noexcept { return ngas_total_; }

  // ----- per-rank accessors -----------------------------------------
  // For a given world rank r, return:
  //   group_of(r)            -- the sub-file index r is assigned to
  //   leader_of(r)           -- the world rank of r's group leader
  //                             (i.e. the lowest-numbered rank in the
  //                             same group)
  //   position_in_group(r)   -- r's index within its group's
  //                             (ascending-rank) member list
  //   group_size(r)          -- size of r's group
  //   slice_in_subfile(r)    -- r's contiguous slice of *its sub-file's*
  //                             gas particles (uniform partition of
  //                             npart_per_subfile[group_of(r)] across
  //                             the group_size members)
  int           group_of           (int rank) const noexcept;
  int           leader_of          (int rank) const noexcept;
  int           position_in_group  (int rank) const noexcept;
  int           group_size         (int rank) const noexcept;
  GlobalSlice   slice_in_subfile   (int rank) const noexcept;

  // ----- per-group accessors ----------------------------------------
  // Members are stored in ascending world-rank order.  members[0] is
  // therefore the leader.  npart_in_group(g) returns the sub-file's
  // own particle count (== npart_per_subfile[g]).
  const std::vector<int> &group_members  (int group) const noexcept;
  std::size_t             npart_in_group (int group) const noexcept;

private:

  int                                nranks_;
  std::vector<std::size_t>           npart_per_subfile_;
  std::size_t                        ngas_total_;
  std::vector<int>                   group_of_rank_;
  std::vector<std::vector<int>>      members_per_group_;
};

}  // namespace mlhpc

#endif  // MLHPC_GROUP_DISTRIBUTION_HPP
