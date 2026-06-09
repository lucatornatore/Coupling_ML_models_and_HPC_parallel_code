// -*- C++ -*-
// ---------------------------------------------------------------------
//  mlhpc/inference.hpp -- abstract ML-inference plug-in interface.
//
//  The harness treats the actual ML cooling surrogate as a plug-in:
//  any class that derives from `InferenceBackend` and implements
//  `batch_inference` can be wired in.  This way we can develop and
//  benchmark the MPI / GPU coupling against a deterministic *dummy*
//  backend now and switch to a LibTorch / ONNX Runtime / TensorRT
//  backend later without touching the orchestration.
//
//  The interface is intentionally narrow:
//      batch_inference (const ParticleBatch &input,
//                       Real * __restrict     output) const = 0;
//  - read-only batch of N particles (SoA),
//  - caller-owned output buffer of length N (cooling rate per
//    particle, in whatever units the backend agrees with the
//    surrounding pipeline).
//
//  See `inference_dummy.{hpp,cpp}` for the deterministic stub that
//  cases A/B/C currently use.
// ---------------------------------------------------------------------

#ifndef MLHPC_INFERENCE_HPP
#define MLHPC_INFERENCE_HPP

#include "mlhpc/particle_batch.hpp"
#include "mlhpc/types.hpp"

namespace mlhpc
{

class InferenceBackend
{
public:

  virtual ~InferenceBackend () = default;

  // -------------------------------------------------------------------
  //  batch_inference
  //
  //  Inputs:  batch  -- SoA particle batch (length batch.n)
  //  Output:  output -- caller-owned buffer with room for batch.n
  //                     elements of `Real`.  Backend writes one
  //                     cooling-rate value per particle.
  //
  //  Pure function from input to output; backends must not retain
  //  references to either buffer after the call returns.
  // -------------------------------------------------------------------
  virtual void
  batch_inference (const ParticleBatch    &batch,
                   Real *__restrict        output) const = 0;
};

// ---------------------------------------------------------------------
//  DummyBackend
//
//  Deterministic stand-in for the real cooling surrogate.  Computes
//      output[i] = 0.5 * rho[i] + 0.25 * u[i]
//  -- a fixed linear combination of the two mandatory features.  No
//  attempt at physical realism; the purpose is to exercise the end-
//  to-end pipeline (units conversion -> batch construction -> kernel
//  call -> reduction / output) on a function whose result can be
//  hand-verified bit-for-bit in the unit tests.
//
//  The dummy ignores `ne`, `nh`, `Zs`.  When we plug in the real
//  backend (or a more elaborate stub), those columns will be
//  consumed too -- without changing the orchestration code that
//  builds the batch.
// ---------------------------------------------------------------------
class DummyBackend final : public InferenceBackend
{
public:

  void
  batch_inference (const ParticleBatch    &batch,
                   Real *__restrict        output) const override;
};

}  // namespace mlhpc

#endif  // MLHPC_INFERENCE_HPP
