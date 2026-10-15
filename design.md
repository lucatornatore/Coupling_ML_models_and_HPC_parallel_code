# Design note — ML/HPC coupling harness

The rationale of this framework is to study the overhead of coupling HPC codes with the inference of ML engines that runs surrogate models to get the solution to some physical modules.
Hence, given a parallel cosmological simulation where every MPI rank owns a chunk of particles, the question is: where should the ML inference for those particles actually run, and how should the data move?
I see two extremes and a continuum in between, and we want to measure all of them in a controlled setting before committing to a design in the production code.

The harness is minimal on the ML side — a dummy backend that computes `0.5 * rho + 0.25 * u` stands in for the real model. 
The real backend is a plugin prepared by BSC - now the task has been taken over by CINECA.


## The two cases

**Case A** : the embarrassingly parallel baseline.  Every rank evaluates inference locally, there is no cooling-related MPI traffic at all, and the cost is whatever the backend costs on that rank. 

**Case B** is the service topology.  A subset of ranks are designated ML servers; the remaining physics ranks pack their particle batches into MPI messages, ship them to a server, and wait for the cooling rates back.
In case the serving rank is on the same node, we could explore the usage of shared-memory windows for storing batched requests and answers.

The interesting knob here is how the servers are placed relative to the physics ranks, which is controlled at runtime by `--ml-server-nodes=N`:

- With `N = 0` (the default) we put one server rank per shared-memory 
  node, specifically the last rank in each node's `MPI_COMM_TYPE_SHARED`
  group.  Each physics rank talks only to the server on its own node, so
  the communication is intra-node.  This is the locality-friendly extreme.

- With `N > 0` we reserve `N` entire nodes as ML servers.  The node
  index space is cut into `N` contiguous groups, the last node of each
  group becomes the server for that group, and physics ranks are assigned
  to server ranks round-robin.  With `N = 1` we get a single global
  server node, which is the other extreme — all inference is centralised.

  Note: If `N` is larger than the number of nodes used, the code silently
  falls back to `N = 0`.

This way we can preliminary explore the whole spectrum from fully distributed to fully centralised serving, which is exactly what we need. Relying on the obtained data, we will move forward refining the framework on the basis measured performance.


## How the code is structured

The architecture has three layers that talk to each other top-down:

```
driver (main.cpp)
  └─ Coupling  [public abstract interface]
       ├─ CouplingA          (Case A: local, no MPI)
       └─ ServiceCoupling    (Case B: topology decisions)
            └─ AsyncMpiTransport  (all MPI message exchange)
                  └─ coupling_wire  (packing/unpacking helpers)
```

The driver only sees the `Coupling` interface and knows nothing about MPI tags, sub-communicators, or wire layouts.  `ServiceCoupling` knows which ranks are servers and how to route requests, but delegates every byte of MPI work to `AsyncMpiTransport`.  
The transport layer, in turn, knows how to pack and exchange payloads but has no idea why a rank is a server or what physics produced the batch. 
The backend is independent of all of this.

This separation matters because it is a way to keep the three dimensions of the experiment — topology, transport, and model — cleanly disentangled.  
The rationale is to keep the option of seamlessly change the server placement policy without touching the transport code; as well, to eventually swap in a GPU-backed or persistent-MPI transport without touching the topology logic, etc.


## The Coupling interface

The public interface in `coupling.hpp` has an asynchronous path and a blocking wrapper around it:

```cpp
CouplingRequest  evaluate_async (const ParticleBatch &, Real *__restrict,
                                 const InferenceBackend &) const;
void             progress () const;
bool             test     (CouplingRequest) const;
void             wait     (CouplingRequest) const;
void             wait_all () const;
void             evaluate (const ParticleBatch &, Real *__restrict,
                           const InferenceBackend &) const;   // submit + wait

void             serve    (const InferenceBackend &) const;  // server ranks
void             shutdown () const;                          // collective
```


 `evaluate_async()` copies or packs the input batch internally before returning, so the caller is free to reuse the batch immediately — this may be important when iterating over time steps; the precise implementation may change when integrated in the real code, but the logic would be already in place.
The output buffer, on the other hand, must stay alive until the matching `wait()` returns.  Progress is driven entirely by the calling thread, so`MPI_THREAD_FUNNELED` is sufficient until a progress thread is not needed (why should it, I sai).

For Case A the async path is trivial — the request is already complete when `evaluate_async()` returns.  This keeps the driver code identical across cases and leaves a natural hook for a future local GPU queue.


## The transport

`AsyncMpiTransport` is a bit more complex.  On the client side, a logical cooling request is cut into chunks of at most `max_batch_particles` particles; each chunk becomes one `QueryHeader` plus one payload sent with `MPI_Isend`, with an `MPI_Irecv` posted immediately for the answer. 
At most `max_inflight_requests` chunks are in flight at once — this is the back-pressure mechanism that prevents a physics rank with a very large batch from flooding the server.

On the server side the loop is straightforward: probe for an incoming header, receive the payload, call `backend.batch_inference()`, send the answer back, count shutdown sentinels.  When all expected sentinels have arrived the loop exits.

## Wire format

The query consists of a small header followed by a contiguous payload:

```cpp
struct QueryHeader {
  std::int64_t   n;        // particles in this chunk
  std::int32_t   nmetal;   // metal species (0 if absent)
  std::uint32_t  flags;    // HAS_NE | HAS_NH | HAS_ZS
};
```

Payload layout: `rho[n] | u[n] | ne[n]? | nh[n]? | Zs[n*nmetal]?`, with optional fields gated by the flags.  The answer is a plain `Real[n]`.
The header is sent as raw bytes — this is fine on a homogeneous cluster, which is our target platform.  `Real` is `float` by default and `double` with `-DMLHPC_FEATURE_DOUBLE=1`; the MPI datatype is picked at runtime.


## Data flow 

```
MPI_Init_thread(FUNNELED)
 -> parse CLI, build Coupling
 -> [physics ranks]
      Format2Distributor reads RHO, U from snapshot on physics_comm
      convert to CGS
      coupling.evaluate_async(batch, out, backend)
      ... overlap window for other physics work goes here ...
      coupling.wait(request)
      coupling.shutdown()
 -> [Case B server ranks]
      coupling.serve(backend)   // blocks until all clients shut down
```

With `--nsteps N` the evaluate loop repeats `N` times, selecting a different pseudo-random active-particle subset each step and scattering the returned cooling rates back into the full output array.  This is meant to mimic what happens in the real code, where only a fraction of particles need cooling at each time step.


## A few additional notes

The particle batch is stored in Struct-of-Arrays layout (`rho[]`, `u[]`,`ne[]`, …) because that is what gives stride-1 access per feature in the packing loops and in the backend.
This is not how it is in the code, we’ll see later how to proceed. **Actually, I admit, this may be the weakest point of this experiment, it could be the case to shift to AoS immediately.**

The transport copies a contiguous slice of each column directly into the MPI payload buffer, avoiding any temporary sub-batch allocation per chunk.

Output arrays in inference APIs are declared `Real *__restrict` to let the compiler assume no aliasing with the input arrays.  This occasionally helps auto-vectorisation, if God's will is in agreement.

The coarse virtual dispatch at the `Coupling` / `InferenceBackend` boundary is one call per cooling evaluation, not one call per particle. Everything inside the transport and backend operates on plain contiguous arrays with no virtual overhead in the inner loops.

Finally, the transport checks that each MPI message fits within `INT_MAX` elements before posting.  At the moment an oversized request aborts loudly rather than being split further; that is a limitation and can be addressed when we start running with very large per-rank particle counts. 
Of course we’ll need a batching that makes sense, this is only a placeholder.
