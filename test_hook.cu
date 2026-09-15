// Quick end-to-end check: memory returned by the hooked cudaMalloc
// (served by memserver over the wire) must be real, GPU-usable memory.
// We write a pattern with a kernel, copy it back, and verify.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill(int *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = i * 3 + 7;
}

int main() {
    const size_t N = (size_t)400 * 1024 * 1024;   // 1.6 GiB -> above redirect threshold
    int *d = nullptr;
    if (cudaMalloc(&d, N * sizeof(int)) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }

    fill<<<(N + 255) / 256, 256>>>(d, (int)N);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("kernel failed\n"); return 1; }

    int host[5];
    size_t idx[5] = {0, 1, N/2, N-2, N-1};
    int ok = 1;
    for (int k = 0; k < 5; k++) {
        cudaMemcpy(&host[k], d + idx[k], sizeof(int), cudaMemcpyDeviceToHost);
        int want = (int)idx[k] * 3 + 7;
        printf("  [%zu] got %d want %d %s\n", idx[k], host[k], want, host[k]==want?"OK":"MISMATCH");
        if (host[k] != want) ok = 0;
    }
    cudaFree(d);
    printf(ok ? "RESULT: PASS (remote-served GPU memory works)\n" : "RESULT: FAIL\n");
    return ok ? 0 : 1;
}
