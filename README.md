# memhook - redirect CUDA allocations to a remote memory server

Run models larger than VRAM by hijacking `cudaMalloc` with `LD_PRELOAD` and
serving the big buffers from a separate "memory server" process, reached over
TCP (localhost now, remotable later).

## The failure this addresses

```
./build/bin/llama-cli -m models/Qwen2.5-32B-Instruct-Q8_0.gguf -ngl 999 ...
-> ggml_backend_cuda_buffer_type_alloc_buffer  (ggml/src/ggml-cuda/ggml-cuda.cu:883)
   -> ggml_cuda_device_malloc                  (ggml-cuda.cu:140)
      -> cudaMalloc                            (ggml-cuda.cu:165)  == OUT OF MEMORY
```

The model needs one 32 GiB weight buffer; the L4 GPU has only 23 GiB. `cudaMalloc`
returns `cudaErrorMemoryAllocation`, then `alloc_tensor_range` / `llama_model_load`
abort. `cudaMalloc` is resolved through `libcudart.so.12` (dynamic), so it can be
interposed with `LD_PRELOAD`.

## Design

- `memserver` (TCP) is the memory manager: it grants and accounts allocations
  against a budget (`MEMSERVER_MAX_BYTES`). Control travels over the network.
- `libmemhook.so` interposes `cudaMalloc`/`cudaFree`. A request `>=` the threshold
  is granted by the server; below it, it passes through to the real `cudaMalloc`
  (stays in VRAM). Two data planes:

  - `MEMHOOK_MODE=paged` (default): allocate the buffer as CUDA managed memory
    (`cudaMallocManaged`). The GPU driver demand-pages it - only the layers in
    use stay resident in VRAM, the rest live in host RAM and page in/out. Weights
    are marked `cudaMemAdviseSetReadMostly` so evicted pages are dropped, not
    written back. Fast, and needs no memlock. The server here only manages the
    oversubscription budget; the bytes are CUDA-managed host RAM.

  - `MEMHOOK_MODE=zerocopy`: the server creates a POSIX shm object; the client
    `mmap`s it and `cudaHostRegister`s it so the GPU reads it in place over PCIe.
    The bytes physically live in the server pool, but every access is latency
    bound, so it is very slow. Needs a high `ulimit -l`.

## Measured (L4 23 GiB, Qwen2.5-32B Q8_0, weights 32 GiB oversubscribed)

    zerocopy            : no token produced in 8 min, GPU 0%
    paged               : prompt 8.4 t/s, gen 0.2 t/s, GPU 100%
    paged + ReadMostly  : prompt 14.1 t/s, gen 0.4 t/s, GPU 100%

Generation stays slow because 32 GiB heavily oversubscribes 23 GiB VRAM: every
token streams the overflow over PCIe. A quant that fits VRAM, or a bigger GPU,
is the only way past that floor.

## Build

    make            # builds memserver and libmemhook.so

## Run

    ./run.sh                       # starts memserver + runs llama-cli via the hook

or manually:

    ./memserver 9797 &
    ulimit -l unlimited            # pinning host memory for the GPU needs this
    LD_PRELOAD=$PWD/libmemhook.so \
    MEMHOOK_MIN_BYTES=1073741824 MEMHOOK_VERBOSE=1 \
      ./build/bin/llama-cli -m models/Qwen2.5-32B-Instruct-Q8_0.gguf -ngl 999 -p "..." -n 16

Env knobs: `MEMHOOK_HOST` (127.0.0.1), `MEMHOOK_PORT` (9797),
`MEMHOOK_MIN_BYTES` (1 GiB), `MEMHOOK_MODE` (paged | zerocopy),
`MEMHOOK_READMOSTLY` (1), `MEMHOOK_VERBOSE` (0). Server: `MEMSERVER_MAX_BYTES`
caps the pool.

## Verify the mechanism

    nvcc -O2 -cudart shared -o test_hook test_hook.cu   # MUST be -cudart shared
    LD_PRELOAD=$PWD/libmemhook.so ./test_hook           # kernel writes/reads remote mem -> PASS

## Caveats

- Keep the threshold high so only the giant weight buffer is redirected and the
  compute/KV buffers stay in fast VRAM.
- zerocopy mode's data plane is `/dev/shm`: same-host only. For a truly remote host
  swap it for RDMA or a paging scheme; the TCP control protocol already generalizes.
  paged mode uses CUDA-managed host RAM, so the server there is a budget manager,
  not the physical byte store.
- Only binaries that link cudart dynamically can be hooked. llama.cpp does; your own
  nvcc programs need `-cudart shared`.
- `cudaHostRegister` page-locks memory, so raise `ulimit -l`.
