# Gadget Format-2 I/O, Parallel Strategy, and Code-Unit Conversion

*Note: written by Claude as documentation FOR BSC ( now CINECA) colleagues when asked to produce the data imported*

--------



This document records
(a) the layout of the Gadget *Format-2* snapshot file as used by OpenGADGET3
(b) the parallel-I/O strategy planned for this project
(c) how code units are translated to physical / CGS units in a cosmological context.

References point to the OpenGADGET3 tree at
`/scratch/WORK/GADGET/STABLE/OpenGadget3/` (notably `IO/read_ic.cpp`,
`IO/io.cpp`, `CodeBase/allvars.hpp`,
`CoolingSfr/Sfr_LT/{lt.hpp,lt_wal_cooling.cpp}`).

---

## 1. Gadget Format-2 file layout

Format-2 is the *labelled* variant of the original Gadget binary
format. Each data block is preceded by a small label block that
identifies it by a 4-character tag. The whole structure is wrapped in
Fortran-style record markers (a 4-byte integer giving the size of the
record both **before and after** the record itself).

### 1.1 Per-block on-disk layout

For every block (label + data):

```
   ┌──────────────────────────────────────────┐
   │  4 bytes : Fortran record marker  = 8    │   ← size of the label record
   │  4 bytes : ASCII tag, e.g. "POS "        │   ← block identifier, space-padded
   │  4 bytes : uint32  nextblock             │   ← bytes from here to start of next label
   │  4 bytes : Fortran record marker  = 8    │
   ├──────────────────────────────────────────┤
   │  4 bytes : Fortran record marker  = N    │   ← N = size of the data payload in bytes
   │  N bytes : the actual data               │
   │  4 bytes : Fortran record marker  = N    │
   └──────────────────────────────────────────┘
```

The very first record of the file is always the **HEAD** label, then
the 256-byte header itself. After that, data blocks follow in an
implementation-defined order; readers must therefore *seek by tag*,
not by position.

Notes:
- `nextblock` in the label is `size_of_data_payload + 2*4` (the two
  Fortran markers), so it can be used to jump straight to the next
  label without parsing the data.
- The Fortran record markers are reflective; comparing the leading
  one against the value `8` (the expected size of the label record) is
  what OpenGADGET3 uses to **detect endianness**
  (`read_ic.cpp:1703-1722`): if it is not 8, the file is byte-swapped
  and `swap_file` is set so every subsequent read is swapped on the
  fly.

### 1.2 The 256-byte header

Canonical layout (matches `header` struct, `allvars.hpp:3022-3060`):

| offset | bytes | C type        | field                              | meaning                                                   |
|-------:|------:|---------------|------------------------------------|-----------------------------------------------------------|
|      0 |    24 | uint32 [6]    | `npart[6]`                         | particles per type **in this file**                       |
|     24 |    48 | double[6]     | `mass[6]`                          | mass per type; 0 if a `MASS` block is present             |
|     72 |     8 | double        | `time`                             | scale factor `a` (cosmological) or time (non-cosmological)|
|     80 |     8 | double        | `redshift`                         | redshift `z = 1/a − 1`                                    |
|     88 |     4 | int32         | `flag_sfr`                         |                                                           |
|     92 |     4 | int32         | `flag_feedback`                    |                                                           |
|     96 |    24 | uint32 [6]    | `npartTotal[6]`                    | total particles per type across all files (low 32 bits)   |
|    120 |     4 | int32         | `flag_cooling`                     |                                                           |
|    124 |     4 | int32         | `num_files`                        | number of files this snapshot is split across            |
|    128 |     8 | double        | `BoxSize`                          | comoving box size, code units                             |
|    136 |     8 | double        | `Omega0`                           | matter density today                                      |
|    144 |     8 | double        | `OmegaLambda`                      | dark-energy density today                                 |
|    152 |     8 | double        | `HubbleParam`                      | h ≡ H₀ / (100 km s⁻¹ Mpc⁻¹)                              |
|    160 |     4 | int32         | `flag_stellarage`                  |                                                           |
|    164 |     4 | int32         | `flag_metals`                      |                                                           |
|    168 |    24 | uint32 [6]    | `npartTotalHighWord[6]`            | high 32 bits, to support > 2³² particles                  |
|    192 |     4 | int32         | `flag_entropy_instead_u`           | 0 → `U` block holds u; 1 → it holds entropy A             |
|    196 |     4 | int32         | `flag_doubleprecision`             | 0 → float fields; 1 → double fields                       |
|    200 |     4 | int32         | `flag_ic_info`                     | IC generator flag                                         |
|    204 |     4 | float         | `lpt_scalingfactor`                | 2LPT scaling                                              |
|    208 |    48 | char[]        | padding / unused                   | brings struct to 256 bytes                                |

Total: **256 bytes**, wrapped in the usual two 4-byte Fortran markers.

### 1.3 Block tags relevant for cooling

All tags are 4 characters, space-padded. We only need the gas-particle
blocks of type 0:

| tag    | name in code        | dtype (gas)          | meaning                                                          |
|:-------|:--------------------|:---------------------|:-----------------------------------------------------------------|
| `HEAD` | header              | 256 B struct         | header (always first)                                            |
| `POS ` | `IO_POS`            | 3 × float/double     | comoving positions [length code units]                           |
| `VEL ` | `IO_VEL`            | 3 × float/double     | peculiar velocity × √a (Gadget convention) [velocity code units] |
| `ID  ` | `IO_ID`             | uint32 or uint64     | particle identifier                                              |
| `MASS` | `IO_MASS`           | float/double         | mass [mass code units]; absent if `header.mass[type] != 0`       |
| `U   ` | `IO_U`              | float/double         | internal energy per unit mass *or* entropy (see header flag)     |
| `RHO ` | `IO_RHO`            | float/double         | comoving density [code mass / code length³]                      |
| `NE  ` | `IO_NE`             | float/double         | electron abundance `n_e / n_H`                                   |
| `NH  ` | `IO_NH`             | float/double         | neutral hydrogen fraction `n_H0 / n_H`                           |
| `HSML` | `IO_HSML`           | float/double         | SPH smoothing length                                             |
| `Z   ` | `IO_Z`              | float/double         | total metal mass fraction (legacy, single number)                |
| `Zs  ` | `IO_Zs`             | `LT_NMetP` × f/d     | individual element mass fractions (per particle)                 |

Floats vs. doubles is controlled by `flag_doubleprecision`. The width
of `ID` blocks must be inferred from `npart * sizeof(IDtype) ==
record_size` since both 32- and 64-bit IDs are in use.

The minimal set we need for the ML cooling inference is:
**RHO, U (+ entropy flag), Zs (or Z), NE, NH**, plus the header
(cosmological parameters, scale factor, units).

### 1.4 Multi-file snapshots

When `header.num_files > 1`, the snapshot lives in files named
`<base>.0`, `<base>.1`, ..., `<base>.N-1`. Each file carries its own
header with its own `npart[]` (per-file count) and the same
`npartTotal[]` / `npartTotalHighWord[]`. The total number of gas
particles is

    Ngas_total = npartTotal[0] + (npartTotalHighWord[0] << 32) .

In OpenGADGET3 the I/O is **not** truly parallel: a small group of
"reading tasks" reads the files sequentially and scatters chunks via
`MPI_Send`/`MPI_Recv` to the consuming tasks (`read_ic.cpp:2213-2370`).
We will not replicate this design — see § 2.

---

## 2. Parallel-I/O strategy for this project

Goal: every MPI task ends up owning a slice of the gas particles,
with the slices balanced as well as the multi-file structure of the
snapshot allows. We do not need to preserve any spatial
decomposition at read time — this is an *ML inference harness*, not
the hydro solver.

The strategy was redefined in session 5 (see `tasks.md` and the
session-5 entry in `log.md`); this section documents the live
design. The previous "every rank opens its own POSIX files"
approach is recorded in §2.4 below for context.

### 2.1 The plan — Step 1: headers + broadcast

**Rank 0 alone** opens every sub-file (`<base>.0`, `<base>.1`, ...)
and reads its 256-byte header. It cross-checks that the cosmology
fields and per-snapshot flags match across sub-files (a mismatch
means the user mixed sub-files from different runs and we fail
loudly). Rank 0 then `MPI_Bcast`s:

- a small fixed-size summary (`num_files`, an `is_multifile` flag,
  the full reference `GadgetHeader2` — broadcast as raw bytes since
  it's trivially copyable);
- the per-sub-file gas-particle counts (`uint64_t * num_files`,
  one `MPI_Bcast` of the array).

After this every rank knows the same snapshot summary at the cost of
a handful of header reads and two `Bcast`s — KB-scale, microseconds.

### 2.2 The plan — Step 2: group computation

Every rank runs the same deterministic local arithmetic
(`GroupDistribution`):

1. Compute prefix sums of the per-sub-file gas counts.
2. For each rank `r` in `[0, Ntasks)`, its *ideal start* in the
   global gas ordering is `g_r = r * Ngas_total / Ntasks` (integer).
3. Rank `r` is assigned to the sub-file `f` whose prefix-sum
   interval contains `g_r` (binary search). All ranks assigned to
   the same sub-file form one **group**, in ascending-rank order.
   The lowest-numbered rank in a group is its **leader**.
4. Fix-up: any non-empty sub-file that ends up with zero ranks (can
   happen when a sub-file is smaller than `Ngas/Ntasks` and falls
   between two consecutive `g_r`s) steals one rank from the largest
   group with more than one member. This guarantees every non-empty
   sub-file has a leader.
5. Within each group the sub-file's gas particles are split via
   `partition_uniform` — the standard ±1 uniform partition — across
   the group members.

The constraint `nranks >= num_files` is required (we cannot place
one group per sub-file with fewer ranks than sub-files); the
constructor throws otherwise.

### 2.3 The plan — Step 3: read and MPI-Isend scatter

For each block we want to read (`RHO`, `U`, `NE`, `NH`, `Zs`, ...):

1. **Block-presence check** (collective): each group leader does a
   `seek_block(tag)` on its sub-file; the result is `MPI_Allreduce`d
   with `MPI_MIN` so all ranks agree on whether the block is present
   in every sub-file. If not, `read_gas_block` returns `false`
   uniformly without any further MPI traffic.
2. **Leader read**: each leader reads the entire gas portion of the
   block (`npart_per_subfile[f] * element_size` bytes) into a local
   buffer via the standard `GadgetReader2` read path.
3. **Scatter**: the leader walks its group members and, for each
   one, computes the member's slice within the sub-file
   (`partition_uniform(npart[f], pos, group_size)`). The leader
   `MPI_Isend`s each slice to its destination rank, then waits on
   all sends with `MPI_Waitall`. The leader's own slice is
   `memcpy`'d directly into its `dst` buffer.
4. **Receive**: non-leader ranks `MPI_Recv` their slice from the
   group leader.

The MPI tag used for `Isend`/`Recv` is the 4-byte block tag packed
into the bottom 31 bits of an `int`, so different blocks use
different MPI tags and could be pipelined in the future.

### 2.4 What we *do not* do (yet)

- No spatial domain decomposition. The inference experiment does
  not need ghost cells; particles are just batches of features.
  When (case B / C) we route particles to inference pools we will
  decide ordering at that point, independently of the I/O.
- No collective MPI-IO (`MPI_File_read_at_all`). Reserved for a
  future iteration if leader-read throughput becomes a bottleneck.
- No streaming on the leader. The leader allocates the full
  sub-file's gas portion of a block before scattering; for the
  cooling case (≲200 MB per block per sub-file at the scales we
  care about) this is fine, but very large sub-files would want
  chunked streaming.
- No write path. We only read OpenGADGET3 snapshots in this
  project.

### 2.5 Historical note — the session-3 strategy (superseded)

Until session 5 the harness used a fully decentralised approach:
every MPI rank independently opened every sub-file it needed and
read its own contiguous slice via `fopen`+`fseek`+`fread`. That gave
genuinely parallel reads from the filesystem and was simple to
reason about, but it (a) duplicated the header-read work across
ranks, (b) introduced per-rank open/close churn, and (c) didn't
exercise the MPI infrastructure that cases B and C will rely on
more heavily. The new strategy trades aggregate disk parallelism
for fewer open files and an MPI-driven distribution path.

---

## 3. Code units → physical → CGS

Gadget's internal units are defined by three independent base
units, chosen at configure time:

```
    UnitLength_in_cm       = 3.085678e21       // 1 kpc / h           (default)
    UnitMass_in_g          = 1.989e43          // 1e10 Msun / h       (default)
    UnitVelocity_in_cm_per_s = 1.0e5           // 1 km/s              (default)
```

From these, every other unit is derived (`allvars.hpp:1267-1269`,
internal energies and densities computed at runtime):

```
    UnitTime_in_s       = UnitLength_in_cm / UnitVelocity_in_cm_per_s
    UnitDensity_in_cgs  = UnitMass_in_g    / UnitLength_in_cm^3
    UnitEnergy_in_cgs   = UnitMass_in_g    * UnitVelocity_in_cm_per_s^2
    UnitPressure_in_cgs = UnitMass_in_g    / UnitLength_in_cm / UnitTime_in_s^2
```

Cosmological snapshots use **comoving** lengths and **comoving**
densities, with the Hubble parameter `h` baked into the units (the
`/h` in the defaults above). Let

```
    a = header.time                      // scale factor
    z = header.redshift = 1/a - 1
    h = header.HubbleParam
```

### 3.1 Density (the one we actually need)

Snapshot stores `ρ_code` in *comoving* units. To get the
**physical, CGS** mass density used by the cooling tables:

```
    ρ_phys_cgs = ρ_code * UnitDensity_in_cgs * h^2 / a^3
```

The `h^2 / a^3` factor unwinds the `(kpc/h)^-3` comoving volume
into a physical `cm^-3`. Equivalently, the hydrogen number density is

```
    n_H = X_H * ρ_phys_cgs / m_p
```

with `X_H ≈ 0.76` (or computed from the `Zs` array if it carries
hydrogen explicitly).

### 3.2 Internal energy / entropy / temperature

The `U` block contains either specific internal energy `u` (when
`flag_entropy_instead_u == 0`) or entropy
`A = P / ρ^γ = (γ-1) u / ρ^{γ-1}` (when the flag is 1). In the
entropy case, recover `u` first:

```
    u_code = A_code * ρ_code^{γ-1} / (γ - 1) ,    γ = 5/3
```

The conversion `u_code → u_phys` is **independent of the scale
factor** in Gadget's convention because velocities (and therefore
`u`, which has units of velocity²) are stored as peculiar
quantities in the comoving frame, with no extra `a` factor for
gas internal energy:

```
    u_phys_cgs = u_code * UnitVelocity_in_cm_per_s^2     // erg / g
```

Temperature from `u`:

```
    T  = (γ - 1) * μ * m_p * u_phys / k_B
```

The mean molecular weight `μ` depends on the ionization state.
Given the snapshot's electron abundance `n_e/n_H ≡ NE` and the
hydrogen mass fraction `X_H`,

```
    1/μ  =  X_H * (1 + NE)  +  Y_He / 4 * (1 + Y_He_ionized_state)
```

For our ML pipeline the exact `μ` is provided by the cooling
table itself, so we expose `u_phys`, `ρ_phys`, `NE`, `NH` and the
abundance vector to the inference; we do not need to compute `T`
on the C++ side.

### 3.3 Cosmology factors we cache once per snapshot

At I/O time we store in a `Cosmology` struct, ready for the
inference layer:

```cpp
struct Cosmology
{
  double a;             // header.time
  double z;             // header.redshift
  double h;             // header.HubbleParam
  double Omega0;        // header.Omega0
  double OmegaLambda;   // header.OmegaLambda
  double rho_to_cgs;    // = UnitDensity_in_cgs * h*h / (a*a*a)
  double u_to_cgs;      // = UnitVelocity_in_cm_per_s^2
};
```

The conversion `rho_to_cgs` already folds in the comoving → physical
and the `/h` of the unit system, so the inference code does a single
multiply per particle.

### 3.4 Note on `Zs`

`Zs` carries individual element **mass fractions** (mass of element
*i* divided by total particle mass). The hydrogen entry may be
implicit (computed as `1 − Σ other`) or explicit, depending on the
Sfr_LT configuration of the run that produced the snapshot
(`lt.hpp:128-136`). The reader exposes the raw `Zs` vector and the
name list; mapping to the ML model's expected feature order is the
inference layer's job.

---

## 4. Open questions to confirm with the OpenGADGET3 run we will use

1. Will the snapshots we test have `flag_entropy_instead_u = 0` or
   `1`? The reader supports both, but it changes the formula.
2. What is the actual `LT_NMetP` of the production runs (i.e. how
   many element mass fractions are stored)? This determines the
   width of the per-particle `Zs` array.
3. Are `NE` and `NH` written in the snapshots we will use, or do
   we need to recompute them from `u` + abundances via the WAL
   cooling tables? (`lt_wal_cooling.cpp:483-1000` does it
   internally.)
4. ID width: 32 or 64 bit? The reader will autodetect from the
   record size, but worth confirming.

These will be resolved as soon as we point the harness at a real
snapshot.
