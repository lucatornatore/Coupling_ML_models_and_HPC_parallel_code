// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/io_gadget2.hpp -- single-file Gadget Format-2 reader.
//
//  Scope of this header: this is the *low-level* reader, working on
//  one on-disk Format-2 snapshot file.  It knows about:
//    - Fortran record markers and endianness detection,
//    - the labelled (HEAD/<TAG>/<TAG>/...) block layout,
//    - finding a block by tag,
//    - reading a contiguous slice of a block's payload as float,
//      double, uint32 or uint64, with on-the-fly byte-swap when the
//      file's endianness differs from the host's.
//
//  What this header deliberately does NOT do (kept for higher layers):
//    - multi-file snapshots (sequence base.0 ... base.N-1):
//      a higher-level `Format2Snapshot` will own a collection of
//      `GadgetReader2` instances and stitch them together.
//    - MPI distribution of particles across ranks: that lives in
//      the (forthcoming) `mpi_partition` module, which decides
//      *which* particles each rank wants and asks this reader for
//      that local slice.
//
//  See `ioformat.md` (sections 1 and 2) for the file-format
//  specification and the I/O strategy that this code implements.
// ---------------------------------------------------------------------

#ifndef MLHPC_IO_GADGET2_HPP
#define MLHPC_IO_GADGET2_HPP

#include "mlhpc/types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace mlhpc
{

// ---------------------------------------------------------------------
//  BlockInfo -- where a Format-2 block sits in the file.
//
//  Populated by GadgetReader2::seek_block().  Keeping the file offset
//  in this struct means a caller can read multiple slices of the same
//  block without re-walking the file every time.
// ---------------------------------------------------------------------
struct BlockInfo
{
  char            tag [4];          // 4-char tag, space-padded, e.g. "POS "
  std::size_t     data_offset;      // byte offset of the *first payload byte*
                                    // in the file (i.e. *after* the leading
                                    // Fortran record marker of the data record)
  std::size_t     data_size;        // size in bytes of the payload (excludes
                                    // both Fortran record markers)
};

// ---------------------------------------------------------------------
//  GadgetReader2 -- RAII handle on a single Format-2 file.
//
//  Construction opens the file, detects endianness from the leading
//  Fortran record marker, and reads the 256-byte header.  After that
//  the public API offers block lookup and typed slice reads.  All
//  failures throw std::runtime_error with a descriptive message:
//  this is a research-grade harness and silent corruption is far
//  worse than a hard stop.
//
//  The class is non-copyable and non-movable: it owns a FILE* and a
//  position; sharing it across threads is not supported, but multiple
//  instances pointing to the same file are fine (each gets its own
//  file descriptor, which is what we want for rank-parallel reads).
// ---------------------------------------------------------------------
class GadgetReader2
{
public:

  explicit GadgetReader2 (const std::string &path);
  ~GadgetReader2 ();

  GadgetReader2 (const GadgetReader2 &)            = delete;
  GadgetReader2 &operator= (const GadgetReader2 &) = delete;

  // Path the reader was opened on (handy for error messages).
  const std::string &path () const noexcept { return path_; }

  // Snapshot header in host byte order.
  const GadgetHeader2 &header () const noexcept { return header_; }

  // True if the file's byte order differs from the host's, so the
  // reader is swapping bytes on every read.
  bool byte_swap () const noexcept { return swap_; }

  // ---------------------------------------------------------------
  //  seek_block
  //
  //  Locate the block with the given 4-char tag.  On success, fills
  //  `out` with the block's file offset and payload size and returns
  //  true.  On EOF without finding the tag, returns false (the
  //  caller decides whether absence is fatal).
  //
  //  Tag matching is byte-exact, including trailing spaces -- callers
  //  should pass e.g. "POS " (with the trailing space) for the
  //  position block.
  // ---------------------------------------------------------------
  bool seek_block (const char tag [4],
                   BlockInfo &out);

  // ---------------------------------------------------------------
  //  read_data_floats / _doubles / _uint32 / _uint64
  //
  //  Read `count` elements from `blk`'s data, starting at byte
  //  offset `local_offset_bytes` within the payload, into `dst`.
  //  Byte-swap is applied per element when `byte_swap()` is true.
  //
  //  These are thin wrappers around the common bytes+swap routine
  //  below; they exist mainly to make the call sites self-documenting
  //  about the element type.
  // ---------------------------------------------------------------
  void read_data_floats  (const BlockInfo  &blk,                   // block to read from
			  std::size_t       local_offset_bytes,    // offset within the data
			  std::size_t       count,                 // # of elements
			  float            *dst);                  // out: element_size = 4
  void read_data_doubles (const BlockInfo  &blk,
			  std::size_t       local_offset_bytes,
			  std::size_t       count,
			  double           *dst);
  void read_data_uint32  (const BlockInfo  &blk,
			  std::size_t       local_offset_bytes,
			  std::size_t       count,
			  std::uint32_t    *dst);
  void read_data_uint64  (const BlockInfo  &blk,
			  std::size_t       local_offset_bytes,
			  std::size_t       count,
			  std::uint64_t    *dst);
  
  // ---------------------------------------------------------------
  //  read_data
  //
  //  Raw-bytes variant.  Reads `count` * `element_size` bytes from
  //  the data at byte offset `local_offset_bytes`, and byte-swaps
  //  each `element_size`-sized chunk in place if needed.
  //
  //  Use this when you need to read a block of an exotic width
  //  (e.g. the Zs metal block with per-element width that depends
  //  on the snapshot's `flag_doubleprecision`).
  // ---------------------------------------------------------------
  void read_data (const BlockInfo  &blk,
		  std::size_t       local_offset_bytes,
		  std::size_t       count,
		  std::size_t       element_size,
		  void             *dst);
  
private:
  
  std::string     path_;       // file path (for diagnostics)
  std::FILE      *fp_;         // owning FILE*; closed in dtor
  GadgetHeader2   header_;     // header in host byte order
  bool            swap_;       // true if file endianness != host's
};

}  // namespace mlhpc

#endif  // MLHPC_IO_GADGET2_HPP
