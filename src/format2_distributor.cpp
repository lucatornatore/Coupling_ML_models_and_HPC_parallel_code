// ---------------------------------------------------------------------
//  src/format2_distributor.cpp -- implementation of Format2Distributor.
//
//  See the header for the contract and the three-step strategy.
//
//  MPI tag scheme: read_gas_block packs the 4-byte block tag into the
//  bottom 31 bits of an int (top bit cleared so the result fits in
//  the MPI tag space).  Different blocks therefore use different MPI
//  tags, which lets concurrent transfers of distinct blocks coexist
//  if we ever pipeline them.
// ---------------------------------------------------------------------

#include "mlhpc/format2_distributor.hpp"

#include "mlhpc/group_distribution.hpp"
#include "mlhpc/io_gadget2.hpp"
#include "mlhpc/partition.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace mlhpc
{

namespace
{

// ---------------------------------------------------------------------
//  file_exists -- POSIX stat() existence check.
// ---------------------------------------------------------------------
bool
file_exists (const std::string &path)
{
  struct stat st = {};
  return ::stat (path.c_str (), &st) == 0;
}

// ---------------------------------------------------------------------
//  tag_to_mpi_tag -- pack a 4-char block tag into a non-negative int
//  suitable for MPI_TAG.
// ---------------------------------------------------------------------
int
tag_to_mpi_tag (const char tag [4]) noexcept
{
  std::uint32_t v = 0;
  std::memcpy (&v, tag, 4);
  return static_cast<int> (v & 0x7FFFFFFFu);
}

// ---------------------------------------------------------------------
//  abort_with -- write an error message and tear down the whole MPI
//  job.  Used on rank-0 failure paths during construction, where the
//  other ranks are blocked in MPI_Bcast and would otherwise hang
//  forever.
// ---------------------------------------------------------------------
[[noreturn]] void
abort_with (MPI_Comm comm,
            const std::string &message,
            int code = 1)
{
  std::fprintf (stderr, "mlhpc::Format2Distributor: %s\n",
                message.c_str ());
  std::fflush (stderr);
  MPI_Abort (comm, code);
  std::abort ();  // unreachable, but pleases [[noreturn]]
}

// ---------------------------------------------------------------------
//  DistSummary -- the rank-0-computed summary that gets broadcast.
//  Trivially copyable so we can MPI_Bcast it as raw bytes.
// ---------------------------------------------------------------------
struct DistSummary
{
  std::int32_t    num_files;
  std::int32_t    is_multifile;     // 1 -> sub-files are <base>.0...
  GadgetHeader2   ref_header;
};

static_assert (std::is_trivially_copyable_v<DistSummary>);

}  // anonymous namespace

// =====================================================================
//  Constructor: header read + Bcast + group computation
// =====================================================================

Format2Distributor::Format2Distributor (MPI_Comm           comm,
                                        const std::string &base_or_path)
  : comm_          (comm)
  , rank_          (0)
  , size_          (1)
  , base_          (base_or_path)
  , is_multifile_  (false)
  , num_files_     (0)
{
  MPI_Comm_rank (comm_, &rank_);
  MPI_Comm_size (comm_, &size_);

  // -------------- step 1: rank 0 reads all headers ------------------
  DistSummary                       summary {};
  std::vector<std::uint64_t>        npart_gas_u64;

  if (rank_ == 0)
    {
      // Determine sub-file layout: try <base>.0 first (multi-file),
      // fall back to <base> (single-file).
      std::vector<std::string> paths;
      const std::string        multi0 = base_or_path + ".0";

      if (file_exists (multi0))
        {
          summary.is_multifile = 1;
          // Read file 0's header for num_files.
          try
            {
              GadgetReader2  probe (multi0);
              const auto    &h = probe.header ();
              if (h.num_files < 1)
                {
                  abort_with (comm_,
                              "'" + multi0 + "' header.num_files = "
                              + std::to_string (h.num_files)
                              + " (must be >= 1)");
                }
              summary.num_files = h.num_files;
              summary.ref_header = h;
            }
          catch (const std::exception &e)
            {
              abort_with (comm_, e.what ());
            }
          for (int i = 0; i < summary.num_files; ++i)
            {
              paths.push_back (base_or_path + "." + std::to_string (i));
            }
        }
      else if (file_exists (base_or_path))
        {
          summary.is_multifile = 0;
          try
            {
              GadgetReader2  probe (base_or_path);
              const auto    &h = probe.header ();
              if (h.num_files != 1)
                {
                  abort_with (comm_,
                              "'" + base_or_path
                              + "' looks single-file but header says "
                                "num_files = "
                              + std::to_string (h.num_files)
                              + "; pass the basename instead.");
                }
              summary.num_files = 1;
              summary.ref_header = h;
            }
          catch (const std::exception &e)
            {
              abort_with (comm_, e.what ());
            }
          paths.push_back (base_or_path);
        }
      else
        {
          abort_with (comm_,
                      "neither '" + base_or_path + "' nor '"
                      + multi0 + "' exists");
        }

      // Read every sub-file header, cross-check cosmology / flags
      // consistency, collect per-file gas counts.
      npart_gas_u64.resize (static_cast<std::size_t> (summary.num_files));
      for (int i = 0; i < summary.num_files; ++i)
        {
          try
            {
              GadgetReader2  probe (paths [static_cast<std::size_t> (i)]);
              const auto    &h = probe.header ();
              if (i > 0)
                {
                  const auto &r = summary.ref_header;
                  const auto same = [&] (bool b, const char *what)
                    {
                      if (!b)
                        abort_with (comm_,
                                    "'" + paths [static_cast<std::size_t> (i)]
                                    + "': header field '"
                                    + what + "' disagrees with file 0");
                    };
                  same (h.time            == r.time,            "time");
                  same (h.redshift        == r.redshift,        "redshift");
                  same (h.BoxSize         == r.BoxSize,         "BoxSize");
                  same (h.Omega0          == r.Omega0,          "Omega0");
                  same (h.OmegaLambda     == r.OmegaLambda,     "OmegaLambda");
                  same (h.HubbleParam     == r.HubbleParam,     "HubbleParam");
                  same (h.num_files       == r.num_files,       "num_files");
                  same (h.flag_doubleprecision == r.flag_doubleprecision,
                        "flag_doubleprecision");
                  same (h.flag_entropy_instead_u == r.flag_entropy_instead_u,
                        "flag_entropy_instead_u");
                }
              npart_gas_u64 [static_cast<std::size_t> (i)] = h.npart [0];
            }
          catch (const std::exception &e)
            {
              abort_with (comm_, e.what ());
            }
        }
    }

  // -------------- broadcast: summary + per-sub-file counts ----------
  MPI_Bcast (&summary, sizeof (summary), MPI_BYTE, 0, comm_);

  if (rank_ != 0)
    {
      npart_gas_u64.resize (static_cast<std::size_t> (summary.num_files));
    }
  MPI_Bcast (npart_gas_u64.data (),
             summary.num_files,
             MPI_UINT64_T,
             0, comm_);

  // -------------- absorb broadcast state ----------------------------
  num_files_    = summary.num_files;
  is_multifile_ = (summary.is_multifile != 0);
  ref_header_   = summary.ref_header;

  npart_per_subfile_.resize (static_cast<std::size_t> (num_files_));
  for (int i = 0; i < num_files_; ++i)
    {
      npart_per_subfile_ [static_cast<std::size_t> (i)] =
          static_cast<std::size_t>
              (npart_gas_u64 [static_cast<std::size_t> (i)]);
    }

  // -------------- step 2: every rank builds the same group dist -----
  try
    {
      group_dist_ = std::make_unique<GroupDistribution>
          (size_, npart_per_subfile_);
    }
  catch (const std::exception &e)
    {
      // GroupDistribution throws on invalid input (e.g. nranks <
      // num_files); abort so peers don't deadlock later.
      abort_with (comm_, std::string ("group distribution: ")
                          + e.what ());
    }
}

Format2Distributor::~Format2Distributor () = default;

// =====================================================================
//  Helpers
// =====================================================================

std::string
Format2Distributor::subfile_path (int f) const
{
  if (is_multifile_)
    {
      return base_ + "." + std::to_string (f);
    }
  return base_;
}

GadgetReader2 &
Format2Distributor::leader_reader ()
{
  if (!leader_reader_)
    {
      leader_reader_ =
          std::make_unique<GadgetReader2> (subfile_path (my_group ()));
    }
  return *leader_reader_;
}

// =====================================================================
//  Step 3: collective read + scatter
// =====================================================================

// ---------------------------------------------------------------------
//  read_gas_block
//
//  Per-rank flow:
//    1. Leaders attempt seek_block(tag); MPI_Allreduce the result so
//       all ranks agree on whether the block is universally present.
//       If not, return false on every rank without sending anything.
//    2. Leaders read their sub-file's gas portion of the block into a
//       buffer (size = npart_per_subfile_[my_group] * element_size).
//    3. Leaders MPI_Isend each member's slice (their own slice is
//       memcpy'd in-place); members MPI_Recv from their leader.
//    4. Leaders MPI_Waitall on their outstanding sends.
//
//  Buffer lifetime: the leader's read buffer is a function-local
//  vector that outlives all MPI_Isends because we MPI_Waitall before
//  returning.
// ---------------------------------------------------------------------
bool
Format2Distributor::read_gas_block (const char    tag [4],
                                    std::size_t   element_size,
                                    void         *dst)
{
  // --- presence check (collective) ---------------------------------
  int        local_ok = 1;
  BlockInfo  blk {};
  if (is_group_leader () && npart_per_subfile_ [static_cast<std::size_t> (my_group ())] > 0)
    {
      try
        {
          if (!leader_reader ().seek_block (tag, blk))
            {
              local_ok = 0;
            }
        }
      catch (const std::exception &e)
        {
          abort_with (comm_, std::string ("seek_block(") + tag + "): "
                              + e.what ());
        }
    }
  int global_ok = 1;
  MPI_Allreduce (&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm_);
  if (global_ok == 0)
    {
      return false;
    }

  const int mpi_tag = tag_to_mpi_tag (tag);

  // --- leader path: read whole gas portion, scatter ----------------
  if (is_group_leader ())
    {
      const int    g          = my_group ();
      const auto  &members    = group_dist_->group_members (g);
      const auto   sub_count  = npart_per_subfile_ [static_cast<std::size_t> (g)];
      const auto   gs         = static_cast<int> (members.size ());

      std::vector<unsigned char>  buf (sub_count * element_size);
      if (sub_count > 0)
        {
          try
            {
              leader_reader ().read_data (blk, 0, sub_count,
                                             element_size, buf.data ());
            }
          catch (const std::exception &e)
            {
              abort_with (comm_, std::string ("read_data(") + tag
                                  + "): " + e.what ());
            }
        }

      std::vector<MPI_Request>  reqs;
      reqs.reserve (static_cast<std::size_t> (gs));

      for (int p = 0; p < gs; ++p)
        {
          const GlobalSlice  s     = partition_uniform (sub_count, p, gs);
          const std::size_t  off   = s.gstart * element_size;
          const std::size_t  bytes = s.count () * element_size;
          const int          dest  = members [static_cast<std::size_t> (p)];

          if (dest == rank_)
            {
              if (bytes > 0)
                {
                  std::memcpy (dst, buf.data () + off, bytes);
                }
            }
          else
            {
              MPI_Request  req {};
              MPI_Isend (buf.data () + off, static_cast<int> (bytes),
                         MPI_BYTE, dest, mpi_tag, comm_, &req);
              reqs.push_back (req);
            }
        }

      if (!reqs.empty ())
        {
          MPI_Waitall (static_cast<int> (reqs.size ()),
                       reqs.data (), MPI_STATUSES_IGNORE);
        }
    }
  else
    {
      // --- non-leader: receive from my leader ----------------------
      const std::size_t  bytes = local_count () * element_size;
      MPI_Recv (dst, static_cast<int> (bytes), MPI_BYTE,
                my_leader (), mpi_tag, comm_, MPI_STATUS_IGNORE);
    }

  return true;
}

}  // namespace mlhpc
