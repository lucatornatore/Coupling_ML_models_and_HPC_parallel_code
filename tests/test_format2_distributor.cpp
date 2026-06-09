// ---------------------------------------------------------------------
//  tests/test_format2_distributor.cpp
//
//  End-to-end MPI test of Format2Distributor.  Must be launched with
//  `mpirun -np N`; the harness Makefile runs it with N=4.
//
//  Test scenario:
//    Rank 0 writes a 3-sub-file synthetic Format-2 snapshot to /tmp
//    with per-sub-file gas counts [7, 8, 5] (chosen to match the
//    canonical example in test_group_distribution).  Sub-file f's
//    RHO block is filled with `1000 * f + i` for i in [0, npart[f]),
//    so every value uniquely identifies (file, index) and the test
//    can verify exactly what each rank received.
//
//    Every rank then:
//      * opens the snapshot via Format2Distributor (which does
//        step 1's MPI_Bcast),
//      * computes the same GroupDistribution (step 2),
//      * collectively calls read_gas_block (step 3, with MPI_Isend
//        from each leader),
//      * verifies its received RHO values match what its
//        (group, position-in-group)-derived slice should hold.
//
//    Finally rank 0 deletes the temp files.
//
//  Exits 0 on success, non-zero on any rank's failure (reduced via
//  MPI_Reduce of an error count on rank 0).
// ---------------------------------------------------------------------

#include "mlhpc/format2_distributor.hpp"
#include "mlhpc/group_distribution.hpp"
#include "mlhpc/partition.hpp"
#include "mlhpc/types.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>          // mkstemp, unlink, close
#include <vector>

using mlhpc::Format2Distributor;
using mlhpc::GadgetHeader2;
using mlhpc::GlobalSlice;
using mlhpc::partition_uniform;

namespace
{

// ---------------------------------------------------------------------
//  Format-2 record-writer helpers, duplicated to keep the test
//  standalone (no library dependency on private writers).
// ---------------------------------------------------------------------
void
wlabel (std::FILE     *fp,
        const char     tag [4],
        std::uint32_t  nb)
{
  const std::uint32_t  m = 8;
  std::fwrite (&m, 4, 1, fp);
  std::fwrite (tag, 4, 1, fp);
  std::fwrite (&nb, 4, 1, fp);
  std::fwrite (&m, 4, 1, fp);
}

void
wdata (std::FILE      *fp,
       const void     *p,
       std::uint32_t   sz)
{
  std::fwrite (&sz, 4, 1, fp);
  std::fwrite (p,   sz, 1, fp);
  std::fwrite (&sz, 4, 1, fp);
}

// ---------------------------------------------------------------------
//  write_subfile -- one Format-2 sub-file with HEAD + RHO blocks.
//  RHO is filled with `1000 * file_index + i`.
// ---------------------------------------------------------------------
void
write_subfile (const std::string &path,
               std::uint32_t      nloc,        // gas particles in this file
               std::uint32_t      ntot,        // total across all files
               std::int32_t       nfiles,      // total number of files
               int                file_index)  // 0-based index of this file
{
  GadgetHeader2 h {};
  h.npart [0]            = nloc;
  h.npartTotal [0]       = ntot;
  h.num_files            = nfiles;
  h.time                 = 0.5;
  h.redshift             = 1.0;
  h.BoxSize              = 1000.0;
  h.Omega0               = 0.3;
  h.OmegaLambda          = 0.7;
  h.HubbleParam          = 0.7;
  h.flag_doubleprecision = 0;

  std::FILE *fp = std::fopen (path.c_str (), "wb");
  if (fp == nullptr)
    {
      throw std::runtime_error ("cannot open " + path);
    }

  wlabel (fp, "HEAD", sizeof (GadgetHeader2) + 8);
  wdata  (fp, &h, sizeof (h));

  std::vector<float>  rho (nloc);
  for (std::uint32_t i = 0; i < nloc; ++i)
    {
      rho [i] = static_cast<float> (1000.0 * file_index + i);
    }
  wlabel (fp, "RHO ", static_cast<std::uint32_t> (nloc * sizeof (float)) + 8);
  wdata  (fp, rho.data (),
          static_cast<std::uint32_t> (nloc * sizeof (float)));

  std::fclose (fp);
}

}  // namespace

int
main (int argc, char **argv)
{
  int provided = 0;
  MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  // ----- set up the synthetic snapshot on rank 0 ------------------
  // Sub-file counts [7, 8, 5] match Case 1 of test_group_distribution
  // so the partitioning is well understood.
  constexpr std::uint32_t n0 = 7;
  constexpr std::uint32_t n1 = 8;
  constexpr std::uint32_t n2 = 5;
  constexpr std::uint32_t N  = n0 + n1 + n2;

  char base_buf [256] = {};
  if (rank == 0)
    {
      char tmpl [] = "/tmp/mlhpc_dist_XXXXXX";
      const int fd = mkstemp (tmpl);
      if (fd < 0)
        {
          std::fprintf (stderr, "mkstemp failed\n");
          MPI_Abort (MPI_COMM_WORLD, 2);
        }
      close  (fd);
      unlink (tmpl);
      std::strncpy (base_buf, tmpl, sizeof (base_buf) - 1);
      try
        {
          const std::string base = tmpl;
          write_subfile (base + ".0", n0, N, 3, 0);
          write_subfile (base + ".1", n1, N, 3, 1);
          write_subfile (base + ".2", n2, N, 3, 2);
        }
      catch (const std::exception &e)
        {
          std::fprintf (stderr, "setup failed: %s\n", e.what ());
          MPI_Abort (MPI_COMM_WORLD, 3);
        }
    }
  MPI_Bcast (base_buf, sizeof (base_buf), MPI_CHAR, 0, MPI_COMM_WORLD);
  const std::string base = base_buf;

  // ----- exercise the distributor ----------------------------------
  int local_errors = 0;
  try
    {
      Format2Distributor dist (MPI_COMM_WORLD, base);

      if (dist.num_files () != 3)               ++local_errors;
      if (dist.npart_total_gas () != N)         ++local_errors;
      if (dist.reference_header ().num_files != 3) ++local_errors;

      // Read RHO into the local buffer.
      std::vector<float>  rho (dist.local_count ());
      const bool          ok = dist.read_gas_block ("RHO ",
                                                    sizeof (float),
                                                    rho.data ());
      if (!ok)
        {
          ++local_errors;
        }
      else
        {
          // Verify each received value matches the expected
          // (1000 * file_index + index_within_file) sentinel.
          const int    g     = dist.my_group ();
          const int    pos   = dist.my_position ();
          const int    gs    = dist.my_group_size ();
          const auto  &subnp = (g == 0 ? n0 : g == 1 ? n1 : n2);
          const GlobalSlice slice = partition_uniform (subnp, pos, gs);

          if (slice.count () != rho.size ())
            {
              ++local_errors;
            }
          for (std::size_t i = 0; i < rho.size (); ++i)
            {
              const float expected =
                  static_cast<float> (1000.0 * g + (slice.gstart + i));
              if (rho [i] != expected)
                {
                  std::fprintf (stderr,
                                "  [rank %d] FAIL  RHO[%zu]: "
                                "got %g, expected %g\n",
                                rank, i,
                                static_cast<double> (rho [i]),
                                static_cast<double> (expected));
                  ++local_errors;
                }
            }
        }
    }
  catch (const std::exception &e)
    {
      std::fprintf (stderr, "[rank %d] threw: %s\n", rank, e.what ());
      local_errors = 1;
    }

  // ----- aggregate result, cleanup ---------------------------------
  int total_errors = 0;
  MPI_Reduce (&local_errors, &total_errors, 1, MPI_INT, MPI_SUM, 0,
              MPI_COMM_WORLD);

  MPI_Barrier (MPI_COMM_WORLD);
  if (rank == 0)
    {
      const std::string base_str = base;
      unlink ((base_str + ".0").c_str ());
      unlink ((base_str + ".1").c_str ());
      unlink ((base_str + ".2").c_str ());
    }

  MPI_Finalize ();

  // Only rank 0's exit status is observed by the parent process when
  // running under mpirun (mpirun -np N exits non-zero if any rank
  // exits non-zero).  To make the test definitive across all ranks
  // we already MPI_Reduce'd; non-rank-0 ranks just exit 0 on success.
  if (rank == 0)
    {
      if (total_errors == 0)
        {
          std::printf ("test_format2_distributor: OK\n");
          return EXIT_SUCCESS;
        }
      std::fprintf (stderr,
                    "test_format2_distributor: %d check(s) failed\n",
                    total_errors);
      return EXIT_FAILURE;
    }
  return (local_errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
