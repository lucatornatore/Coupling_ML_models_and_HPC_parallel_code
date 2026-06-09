// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/types.hpp -- POD types shared between I/O and the rest of the
//  harness.
//
//  * GadgetHeader2  : the on-disk 256-byte Gadget snapshot header,
//                     laid out *exactly* as produced by OpenGADGET3
//                     (CodeBase/allvars.hpp:3022-3060).  We pin the
//                     size with a static_assert so a layout drift
//                     fails the build.
//
//  * Cosmology      : the small derived bundle used by the inference
//                     layer to convert code units to physical / CGS
//                     once per particle (see ioformat.md, sec. 3).
//
//  * IDType         : default particle-id width.  Auto-detection at
//                     read time may override this on a per-block
//                     basis; this typedef only fixes the in-memory
//                     default.
//
//  All structs here are PODs (std::is_trivially_copyable) so they can
//  be moved as raw bytes by MPI / read straight from disk.
// ---------------------------------------------------------------------

#ifndef MLHPC_TYPES_HPP
#define MLHPC_TYPES_HPP

#include <cstdint>
#include <type_traits>

namespace mlhpc
{

// ---------------------------------------------------------------------
//  Particle-id width.  Default is 64-bit (production cosmological runs
//  routinely exceed 2^32 particles).  Switch back to 32-bit at compile
//  time with -DMLHPC_ID_32BIT=1 if the snapshots in use are known to
//  carry 32-bit ids and the memory saving matters.
//
//  Note: this only fixes the *in-memory* default; the reader still
//  autodetects the on-disk width from the ID block's payload size,
//  so reading a 32-bit-id snapshot into a uint64_t buffer is fine.
// ---------------------------------------------------------------------
#if defined(MLHPC_ID_32BIT) && MLHPC_ID_32BIT
using IDType = std::uint32_t;
#else
using IDType = std::uint64_t;
#endif

// ---------------------------------------------------------------------
//  Real -- in-memory floating-point dtype for the cooling-relevant
//  particle data (rho, u, Zs[], NE, NH, ...).  Default is `float`
//  because the OpenGADGET3 snapshots we target (see
//  ./tests/snapshot) carry these fields as single precision on
//  disk, so `float` is lossless.  Switch to `double` at compile time
//  with -DMLHPC_FEATURE_DOUBLE=1 -- typically only useful if the
//  snapshot itself is double precision, or if the conversion to
//  physical / CGS units would lose accuracy in the later pipeline.
//
//  The reader itself does NOT convert between disk and in-memory
//  precision: a snapshot whose `flag_doubleprecision` disagrees
//  with `sizeof(Real)` must be rejected by the driver, otherwise
//  the bytes would be misinterpreted silently.
// ---------------------------------------------------------------------
#if defined(MLHPC_FEATURE_DOUBLE) && MLHPC_FEATURE_DOUBLE
using Real = double;
#else
using Real = float;
#endif

static_assert (std::is_floating_point_v<Real>,
               "mlhpc::Real must be a floating-point type.");

#if defined(MLHPC_FEATURE_DOUBLE) && MLHPC_FEATURE_DOUBLE
static_assert (sizeof (Real) == 8,
               "MLHPC_FEATURE_DOUBLE was requested but Real is not 8 bytes.");
#else
static_assert (sizeof (Real) == 4,
               "Default Real must be 4 bytes (single precision).");
#endif

// ---------------------------------------------------------------------
//  GadgetHeader2 -- canonical 256-byte header of a Format-2 Gadget
//  snapshot file.  See ioformat.md sec. 1.2 for the field list and
//  byte offsets.
//
//  Endianness handling: a Format-2 file is little-endian on x86 and
//  big-endian on legacy ppc; the reader detects mismatch via the
//  leading Fortran record marker and byte-swaps every field on the
//  fly.  This struct is therefore *always* in host byte order by the
//  time it lands here.
// ---------------------------------------------------------------------
struct GadgetHeader2
{
  std::uint32_t   npart [6];                  //   0  particles in *this* file per type
  double          mass  [6];                  //  24  mass per type; 0 -> MASS block present
  double          time;                       //  72  scale factor a (cosmological)
  double          redshift;                   //  80  z = 1/a - 1
  std::int32_t    flag_sfr;                   //  88
  std::int32_t    flag_feedback;              //  92
  std::uint32_t   npartTotal [6];             //  96  totals across all files, low 32 bits
  std::int32_t    flag_cooling;               // 120
  std::int32_t    num_files;                  // 124  files this snapshot is split across
  double          BoxSize;                    // 128  comoving, code units
  double          Omega0;                     // 136
  double          OmegaLambda;                // 144
  double          HubbleParam;                // 152  h = H_0 / (100 km/s/Mpc)
  std::int32_t    flag_stellarage;            // 160
  std::int32_t    flag_metals;                // 164
  std::uint32_t   npartTotalHighWord [6];     // 168  totals high 32 bits (>2^32 support)
  std::int32_t    flag_entropy_instead_u;     // 192  0 -> U block is u; 1 -> entropy
  std::int32_t    flag_doubleprecision;       // 196  0 -> float fields; 1 -> double fields
  std::int32_t    flag_ic_info;               // 200
  float           lpt_scalingfactor;          // 204
  char            fill [48];                  // 208  padding to 256 bytes
};

static_assert (sizeof (GadgetHeader2) == 256,
               "GadgetHeader2 must be exactly 256 bytes -- the on-disk "
               "Gadget Format-2 header is fixed by the file format and "
               "any drift would silently corrupt every snapshot read.");

static_assert (std::is_trivially_copyable_v<GadgetHeader2>,
               "GadgetHeader2 must be trivially copyable so it can be "
               "read straight from disk and broadcast via MPI.");

// ---------------------------------------------------------------------
//  Cosmology -- cached per-snapshot quantities, ready for the
//  inference layer.  The two `*_to_cgs` factors fold in the comoving
//  -> physical and the unit-system constants so a single multiply
//  converts a code-unit field to CGS at use-time (ioformat.md sec. 3).
// ---------------------------------------------------------------------
struct Cosmology
{
  double   a;             // scale factor              (header.time)
  double   z;             // redshift                  (header.redshift)
  double   h;             // dimensionless Hubble      (header.HubbleParam)
  double   Omega0;
  double   OmegaLambda;
  double   rho_to_cgs;    // multiply rho_code -> g/cm^3 (physical)
  double   u_to_cgs;      // multiply u_code   -> erg/g  (physical)
};

static_assert (std::is_trivially_copyable_v<Cosmology>);

}  // namespace mlhpc

#endif  // MLHPC_TYPES_HPP
