// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/format2_distributor.hpp -- MPI-aware Format-2 reader.
//
//  Implements the strategy specified in `tasks.md` (session 5):
//
//    Step 1: Rank 0 reads every sub-file's header and gets Ngas_gas
//            and per-sub-file gas counts; broadcasts that summary
//            (plus the reference header) to all ranks via MPI_Bcast.
//
//    Step 2: All ranks compute -- locally, deterministically -- the
//            same `GroupDistribution`: one group per sub-file, group
//            sizes proportional to per-sub-file particle counts via
//            the prefix-sum walk over `r * Ngas / Ntasks`.  Each rank
//            knows its group, its position in the group, its group's
//            leader (the lowest-numbered rank in the group), and its
//            slice within the sub-file.
//
//    Step 3: On read_gas_block(): every group leader opens its sub-
//            file (lazily, on first read), reads the gas-particle
//            portion of the requested block, and scatters slices to
//            its group members via MPI_Isend.  Non-leaders MPI_Recv
//            their slice from their leader.  Block-presence is
//            verified with a single MPI_Allreduce up front to avoid
//            non-leaders deadlocking when a leader's sub-file is
//            missing the block.
//
//  This replaces the session-3 `Format2Snapshot` strategy (every rank
//  opens every needed sub-file via POSIX I/O).  The new strategy
//  trades aggregate I/O bandwidth for fewer open files and less
//  duplicate disk reads, and exercises the MPI infrastructure that
//  cases B and C will lean on more heavily.
//
//  The class is non-copyable; it owns MPI state (a leader's
//  GadgetReader2) and a logical group-distribution view.
// ---------------------------------------------------------------------

#ifndef MLHPC_FORMAT2_DISTRIBUTOR_HPP
#define MLHPC_FORMAT2_DISTRIBUTOR_HPP

#include "mlhpc/group_distribution.hpp"
#include "mlhpc/io_gadget2.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlhpc
{

class Format2Distributor
{
public:

  // -------------------------------------------------------------------
  //  Construct: discover the sub-file set on rank 0, read every
  //  header on rank 0, broadcast the summary to all ranks, and
  //  compute the per-rank group assignment.
  //
  //  Inputs:  comm          -- MPI communicator (typically MPI_COMM_WORLD)
  //           base_or_path  -- single-file path or multi-file basename;
  //                            same semantics as the session-3
  //                            Format2Snapshot accepted.
  //
  //  Throws:  std::runtime_error or std::invalid_argument on rank 0
  //           if the snapshot is malformed or inconsistent.  Failure
  //           paths call MPI_Abort to avoid stranding peers in the
  //           Bcast.
  // -------------------------------------------------------------------
  Format2Distributor (MPI_Comm           comm,
                      const std::string &base_or_path);

  ~Format2Distributor ();

  Format2Distributor (const Format2Distributor &)            = delete;
  Format2Distributor &operator= (const Format2Distributor &) = delete;

  // ----- snapshot-wide accessors ------------------------------------
  const GadgetHeader2 &reference_header  () const noexcept { return ref_header_; }
  std::size_t          npart_total_gas   () const noexcept { return group_dist_->ngas_total (); }
  int                  num_files         () const noexcept { return num_files_; }

  // ----- this rank's MPI state --------------------------------------
  int                  my_rank           () const noexcept { return rank_; }
  int                  comm_size         () const noexcept { return size_; }

  // ----- this rank's group assignment -------------------------------
  int                  my_group          () const noexcept { return group_dist_->group_of (rank_); }
  int                  my_leader         () const noexcept { return group_dist_->leader_of (rank_); }
  int                  my_group_size     () const noexcept { return group_dist_->group_size (rank_); }
  int                  my_position       () const noexcept { return group_dist_->position_in_group (rank_); }
  bool                 is_group_leader   () const noexcept { return rank_ == my_leader (); }
  std::size_t          local_count       () const noexcept { return group_dist_->slice_in_subfile (rank_).count (); }

  // For tests, diagnostics, main-driver reporting.
  const GroupDistribution &distribution  () const noexcept { return *group_dist_; }

  // -------------------------------------------------------------------
  //  read_gas_block
  //
  //  Collective.  Every rank must call it with the same tag.  On
  //  return:
  //    * each rank's `dst` holds exactly `local_count() * element_size`
  //      bytes corresponding to its slice of its sub-file's gas
  //      portion of the named block;
  //    * the function returns true iff every leader found the tag in
  //      its sub-file (verified via MPI_Allreduce).
  //
  //  `dst` must have room for `local_count() * element_size` bytes,
  //  including when local_count() == 0 (in which case nothing is
  //  written but the collective protocol still runs).
  //
  //  Throws std::runtime_error on a local I/O failure during the
  //  leader read; calls MPI_Abort to avoid leaving peers stuck in
  //  MPI_Recv.
  // -------------------------------------------------------------------
  bool read_gas_block (const char    tag [4],
                       std::size_t   element_size,
                       void         *dst);

private:

  // Path of sub-file `f` reconstructed from `base_` and `is_multifile_`.
  std::string  subfile_path (int f) const;

  // Lazily open the leader's reader on first use.
  GadgetReader2 &leader_reader ();

  MPI_Comm                                comm_;
  int                                     rank_;
  int                                     size_;
  std::string                             base_;
  bool                                    is_multifile_;
  int                                     num_files_;
  GadgetHeader2                           ref_header_;
  std::vector<std::size_t>                npart_per_subfile_;
  std::unique_ptr<GroupDistribution>      group_dist_;
  std::unique_ptr<GadgetReader2>          leader_reader_;
};

}  // namespace mlhpc

#endif  // MLHPC_FORMAT2_DISTRIBUTOR_HPP
