// ---------------------------------------------------------------------
//  tests/test_io_gadget2.cpp
//
//  End-to-end test for mlhpc::GadgetReader2 against a synthetic
//  Gadget Format-2 file that we build in this very test:
//
//      [HEAD record]
//      [BAR  record  (8-float payload, sentinel block to exercise the
//                     "skip a non-matching block" code path)]
//      [POS  record  (12 floats: 4 particles * 3 coords)]
//      [ID   record  (4 uint32 ids)]
//
//  The test exercises:
//      * endianness detection on a native-order file,
//      * header parsing (every field round-trips),
//      * seek_block() across skip-then-find,
//      * read_data_floats with a partial slice (offset + count),
//      * read_data_uint32,
//      * out-of-bounds payload-read error path.
//
//  No external test framework: any std::runtime_error or wrong value
//  flips `errors` and the test exits non-zero, which CTest reads as
//  a failure.
// ---------------------------------------------------------------------

#include "mlhpc/io_gadget2.hpp"
#include "mlhpc/types.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>          // mkstemp

using mlhpc::BlockInfo;
using mlhpc::GadgetHeader2;
using mlhpc::GadgetReader2;

namespace
{

int errors = 0;

template <typename T>
void
expect_eq (const char *what, T got, T want)
{
  if (!(got == want))
    {
      std::fprintf (stderr, "  FAIL  %s: got vs expected differ\n", what);
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

// ---------------------------------------------------------------------
//  write_label_record
//
//  Emit one Format-2 *label* record: leading marker 8, 4-char tag,
//  uint32 nextblock, trailing marker 8.  `nextblock` is the value
//  that would let a scanner skip the data record without parsing it;
//  we set it to (data_size + 8) to match OpenGADGET3's convention.
// ---------------------------------------------------------------------
void
write_label_record (std::FILE    *fp,
                    const char    tag [4],
                    std::uint32_t nextblock)
{
  const std::uint32_t marker = 8;
  std::fwrite (&marker,   sizeof (marker), 1, fp);
  std::fwrite (tag,       4,               1, fp);
  std::fwrite (&nextblock, sizeof (nextblock), 1, fp);
  std::fwrite (&marker,   sizeof (marker), 1, fp);
}

// ---------------------------------------------------------------------
//  write_data_record
//
//  Emit one Format-2 *data* record: leading marker N, N bytes of
//  payload, trailing marker N.
// ---------------------------------------------------------------------
void
write_data_record (std::FILE      *fp,
                   const void     *payload,
                   std::uint32_t   data_size)
{
  std::fwrite (&data_size, sizeof (data_size), 1, fp);
  std::fwrite (payload,       data_size,          1, fp);
  std::fwrite (&data_size, sizeof (data_size), 1, fp);
}

// ---------------------------------------------------------------------
//  make_temp_path
//
//  Create a unique temporary file path under /tmp.  We use mkstemp
//  so we get an actual unique name; we immediately close the fd
//  because the rest of the test uses stdio fopen on the path.
// ---------------------------------------------------------------------
std::string
make_temp_path ()
{
  char tmpl [] = "/tmp/mlhpc_fmt2_XXXXXX";
  const int fd = mkstemp (tmpl);
  if (fd < 0)
    {
      throw std::runtime_error ("mkstemp failed");
    }
  close (fd);
  return std::string (tmpl);
}

// ---------------------------------------------------------------------
//  build_synthetic_snapshot
//
//  Write a tiny Format-2 snapshot at `path`.  The header carries
//  recognisable sentinel values so we can verify round-trip parsing.
//  Returns the truth-table data we wrote, so the test can compare
//  what the reader returns.
// ---------------------------------------------------------------------
struct Truth
{
  GadgetHeader2  header;
  float          pos [12];   // 4 particles * 3 coords
  std::uint32_t  ids [4];
};

Truth
build_synthetic_snapshot (const std::string &path)
{
  Truth t {};

  // --- header (sentinel values) -------------------------------------
  t.header.npart [0]       = 4;       // gas
  t.header.npart [1]       = 0;
  t.header.mass  [0]       = 0.0;     // -> MASS block would follow (we skip it)
  t.header.time            = 0.5;     // a = 0.5  -> z = 1
  t.header.redshift        = 1.0;
  t.header.npartTotal [0]  = 4;
  t.header.num_files       = 1;
  t.header.BoxSize         = 1000.0;
  t.header.Omega0          = 0.3;
  t.header.OmegaLambda     = 0.7;
  t.header.HubbleParam     = 0.7;
  t.header.flag_doubleprecision = 0;

  // --- POS payload --------------------------------------------------
  for (int i = 0; i < 12; ++i)
    {
      t.pos [i] = static_cast<float> (0.5f * (i + 1));   // 0.5, 1.0, 1.5, ...
    }

  // --- ID payload ---------------------------------------------------
  t.ids [0] = 100;
  t.ids [1] = 101;
  t.ids [2] = 102;
  t.ids [3] = 103;

  // --- write it out -------------------------------------------------
  std::FILE *fp = std::fopen (path.c_str (), "wb");
  if (fp == nullptr)
    {
      throw std::runtime_error ("cannot open temp file for writing");
    }

  // HEAD label + 256-byte header data record
  write_label_record (fp, "HEAD", sizeof (GadgetHeader2) + 8);
  write_data_record  (fp, &t.header, sizeof (GadgetHeader2));

  // A red-herring "BAR " block of 8 floats, to exercise the skip path
  float bar [8] = { 9, 9, 9, 9, 9, 9, 9, 9 };
  write_label_record (fp, "BAR ", sizeof (bar) + 8);
  write_data_record  (fp, bar, sizeof (bar));

  // POS  block
  write_label_record (fp, "POS ", sizeof (t.pos) + 8);
  write_data_record  (fp, t.pos, sizeof (t.pos));

  // ID   block
  write_label_record (fp, "ID  ", sizeof (t.ids) + 8);
  write_data_record  (fp, t.ids, sizeof (t.ids));

  std::fclose (fp);
  return t;
}

}  // namespace

int
main ()
{
  std::string path;
  Truth       truth;

  try
    {
      path  = make_temp_path ();
      truth = build_synthetic_snapshot (path);
    }
  catch (const std::exception &e)
    {
      std::fprintf (stderr, "setup failed: %s\n", e.what ());
      return EXIT_FAILURE;
    }

  try
    {
      GadgetReader2 reader (path);

      // --- header round-trip --------------------------------------
      const GadgetHeader2 &h = reader.header ();
      expect_eq      ("byte_swap is false on a native-order file",
                     reader.byte_swap (), false);
      expect_eq <std::uint32_t> ("header.npart[0]",
                                 h.npart [0], 4);
      expect_eq <double>        ("header.time",
                                 h.time, 0.5);
      expect_eq <double>        ("header.redshift",
                                 h.redshift, 1.0);
      expect_eq <double>        ("header.BoxSize",
                                 h.BoxSize, 1000.0);
      expect_eq <double>        ("header.Omega0",
                                 h.Omega0, 0.3);
      expect_eq <double>        ("header.OmegaLambda",
                                 h.OmegaLambda, 0.7);
      expect_eq <double>        ("header.HubbleParam",
                                 h.HubbleParam, 0.7);
      expect_eq <std::int32_t>  ("header.num_files",
                                 h.num_files, 1);

      // --- locate POS, read all 12 floats -------------------------
      BlockInfo pos_info {};
      const bool got_pos = reader.seek_block ("POS ", pos_info);
      expect_true ("seek_block(POS ) returns true", got_pos);
      expect_eq <std::size_t> ("POS payload size",
                              pos_info.data_size,
                              sizeof (truth.pos));

      float pos_read [12] = {};
      reader.read_data_floats (pos_info, 0, 12, pos_read);
      for (int i = 0; i < 12; ++i)
        {
          if (pos_read [i] != truth.pos [i])
            {
              std::fprintf (stderr,
                            "  FAIL  POS[%d]: got %f, expected %f\n",
                            i,
                            static_cast<double> (pos_read [i]),
                            static_cast<double> (truth.pos [i]));
              ++errors;
            }
        }

      // --- partial slice: 2 particles starting at particle 1 -------
      // payload offset = 1 particle * 3 floats * 4 bytes = 12
      float pos_slice [6] = {};
      reader.read_data_floats (pos_info, 12, 6, pos_slice);
      for (int i = 0; i < 6; ++i)
        {
          if (pos_slice [i] != truth.pos [3 + i])
            {
              std::fprintf (stderr,
                            "  FAIL  POS slice[%d]: got %f, expected %f\n",
                            i,
                            static_cast<double> (pos_slice [i]),
                            static_cast<double> (truth.pos [3 + i]));
              ++errors;
            }
        }

      // --- locate ID, read all 4 uints ----------------------------
      BlockInfo id_info {};
      const bool got_id = reader.seek_block ("ID  ", id_info);
      expect_true ("seek_block(ID  ) returns true", got_id);

      std::uint32_t ids_read [4] = {};
      reader.read_data_uint32 (id_info, 0, 4, ids_read);
      for (int i = 0; i < 4; ++i)
        {
          expect_eq <std::uint32_t>
              ("ID[i]", ids_read [i], truth.ids [i]);
        }

      // --- missing block reports false, no throw -------------------
      BlockInfo missing {};
      const bool got_missing = reader.seek_block ("XXXX", missing);
      expect_true ("seek_block on absent tag returns false",
                   !got_missing);

      // --- out-of-bounds payload read must throw -------------------
      bool threw = false;
      try
        {
          float overflow_buf [16] = {};
          reader.read_data_floats (pos_info, 0, 16, overflow_buf);
        }
      catch (const std::runtime_error &)
        {
          threw = true;
        }
      expect_true ("out-of-bounds read throws runtime_error", threw);
    }
  catch (const std::exception &e)
    {
      std::fprintf (stderr, "reader threw unexpectedly: %s\n", e.what ());
      ++errors;
    }

  std::remove (path.c_str ());

  if (errors == 0)
    {
      std::printf ("test_io_gadget2: OK\n");
      return EXIT_SUCCESS;
    }

  std::fprintf (stderr,
                "test_io_gadget2: %d check(s) failed\n", errors);
  return EXIT_FAILURE;
}
