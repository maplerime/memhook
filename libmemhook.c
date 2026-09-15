// libmemhook - LD_PRELOAD interposer for cudaMalloc / cudaFree.
//
// The failing call in llama.cpp is:
//   ggml_backend_cuda_buffer_type_alloc_buffer -> ggml_cuda_device_malloc
//     -> cudaMalloc   (out of memory: 32 GiB weights vs 23 GiB VRAM)
//
// We interpose cudaMalloc. Large requests are granted by memserver over TCP
// (the server is the memory manager: it tracks/limits the oversubscription
// budget). Two data planes:
//
//   MEMHOOK_MODE=paged     (default) - allocate the buffer as CUDA managed
//       memory. The GPU driver demand-pages it: only the layers in use stay
//       resident in VRAM, the rest live in host RAM and are paged in/out.
//       Fast (streams at PCIe bandwidth), and needs no memlock.
//
//   MEMHOOK_MODE=zerocopy  - map a shared-memory object from the server and
//       cudaHostRegister it. The GPU reads it in place over PCIe. The bytes
//       physically live in the server pool, but every access is latency-bound
//       so it is slow. Needs a high 'ulimit -l'.
//
// Small requests fall through to the real cudaMalloc and stay in VRAM.
//
// Controls (env):
//   MEMHOOK_HOST       server host      (default 127.0.0.1)
//   MEMHOOK_PORT       server port      (default 9797)
//   MEMHOOK_MIN_BYTES  redirect threshold in bytes (default 1073741824 = 1 GiB)
//   MEMHOOK_MODE       paged | zerocopy (default paged)
//   MEMHOOK_VERBOSE    1 to log each hooked call
//
// build: gcc -O2 -Wall -fPIC -shared -o libmemhook.so libmemhook.c -ldl -lrt

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cuda_runtime_api.h>
#include "proto.h"

static cudaError_t (*real_cudaMalloc)(void **, size_t);
static cudaError_t (*real_cudaMallocManaged)(void **, size_t, unsigned int);
static cudaError_t (*real_cudaMemAdvise)(const void *, size_t, int, int);
static cudaError_t (*real_cudaGetDevice)(int *);
static cudaError_t (*real_cudaFree)(void *);
static cudaError_t (*real_cudaHostRegister)(void *, size_t, unsigned int);
static cudaError_t (*real_cudaHostUnregister)(void *);
static cudaError_t (*real_cudaHostGetDevicePointer)(void **, void *, unsigned int);

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int    g_sock = -1;
static size_t g_min  = 1ULL << 30;
static char   g_host[64] = "127.0.0.1";
static int    g_port = MEMHOOK_PORT;
static int    g_verbose = 0;
static int    g_paged = 0;   // default: zerocopy (set MEMHOOK_MODE=paged for demand paging)

// one record per redirected allocation, keyed by the pointer we returned
struct rec {
    void   *devptr;
    void   *host;    // zerocopy: mmap base to unmap; paged: NULL
    size_t  size;
    uint64_t id;
    int      paged;
    struct rec *next;
};
static struct rec *g_recs = NULL;

#define LOGE(...) do { fprintf(stderr, "memhook: " __VA_ARGS__); } while (0)
#define LOGV(...) do { if (g_verbose) fprintf(stderr, "memhook: " __VA_ARGS__); } while (0)

static void init_once(void) {
    real_cudaMalloc               = dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaMallocManaged        = dlsym(RTLD_NEXT, "cudaMallocManaged");
    real_cudaMemAdvise            = dlsym(RTLD_NEXT, "cudaMemAdvise");
    real_cudaGetDevice            = dlsym(RTLD_NEXT, "cudaGetDevice");
    real_cudaFree                 = dlsym(RTLD_NEXT, "cudaFree");
    real_cudaHostRegister         = dlsym(RTLD_NEXT, "cudaHostRegister");
    real_cudaHostUnregister       = dlsym(RTLD_NEXT, "cudaHostUnregister");
    real_cudaHostGetDevicePointer = dlsym(RTLD_NEXT, "cudaHostGetDevicePointer");

    const char *h = getenv("MEMHOOK_HOST");
    if (h) snprintf(g_host, sizeof(g_host), "%s", h);
    const char *p = getenv("MEMHOOK_PORT");
    if (p) g_port = atoi(p);
    const char *m = getenv("MEMHOOK_MIN_BYTES");
    if (m) g_min = strtoull(m, NULL, 10);
    const char *v = getenv("MEMHOOK_VERBOSE");
    if (v) g_verbose = atoi(v);
    const char *mode = getenv("MEMHOOK_MODE");
    if (mode && strcmp(mode, "paged") == 0) g_paged = 1;

    LOGE("active: server=%s:%d  mode=%s  redirect >= %.2f MiB\n",
         g_host, g_port, g_paged ? "paged" : "zerocopy", g_min / 1048576.0);
}

static int full_read(int fd, void *buf, size_t n) {
    char *p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r == 0) return -1;
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        p += r; n -= (size_t) r;
    }
    return 0;
}

static int full_write(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n) {
        ssize_t r = write(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        p += r; n -= (size_t) r;
    }
    return 0;
}

// caller holds g_lock
static int ensure_conn(void) {
    if (g_sock >= 0) return 0;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port   = htons((uint16_t) g_port);
    if (inet_pton(AF_INET, g_host, &a.sin_addr) != 1) { close(fd); return -1; }
    if (connect(fd, (struct sockaddr *) &a, sizeof(a)) != 0) {
        LOGE("cannot connect to %s:%d: %s\n", g_host, g_port, strerror(errno));
        close(fd);
        return -1;
    }
    g_sock = fd;
    return 0;
}

// ask the server to grant size bytes; name is filled only in zerocopy mode
static int server_alloc(size_t size, uint32_t flags, char *name, size_t namecap,
                        uint64_t *out_id, size_t *out_size) {
    if (ensure_conn() != 0) return -1;

    struct msg_req req;
    memset(&req, 0, sizeof(req));
    req.magic = MEMHOOK_MAGIC;
    req.op    = OP_ALLOC;
    req.size  = size;
    req.flags = flags;
    if (full_write(g_sock, &req, sizeof(req)) != 0) goto broken;

    struct msg_resp resp;
    if (full_read(g_sock, &resp, sizeof(resp)) != 0) goto broken;
    if (resp.magic != MEMHOOK_MAGIC || resp.status != ST_OK) return -1;
    if (resp.name_len >= namecap) return -1;
    if (resp.name_len && full_read(g_sock, name, resp.name_len) != 0) goto broken;
    name[resp.name_len] = '\0';

    *out_id   = resp.id;
    *out_size = resp.size;
    return 0;

broken:
    close(g_sock);
    g_sock = -1;
    return -1;
}

static void server_free(uint64_t id) {
    if (g_sock < 0) return;
    struct msg_req req;
    memset(&req, 0, sizeof(req));
    req.magic = MEMHOOK_MAGIC;
    req.op    = OP_FREE;
    req.id    = id;
    if (full_write(g_sock, &req, sizeof(req)) != 0) { close(g_sock); g_sock = -1; return; }
    struct msg_resp resp;
    if (full_read(g_sock, &resp, sizeof(resp)) != 0) { close(g_sock); g_sock = -1; }
}

// managed memory: server just grants, GPU driver demand-pages the bytes
static cudaError_t alloc_paged(void **ptr, size_t size) {
    if (!real_cudaMallocManaged) return real_cudaMalloc(ptr, size);

    pthread_mutex_lock(&g_lock);
    char name[64];
    uint64_t id = 0; size_t granted = 0;
    if (server_alloc(size, FLAG_PAGED, name, sizeof(name), &id, &granted) != 0) {
        pthread_mutex_unlock(&g_lock);
        LOGE("server_alloc(%zu) failed, falling back to real cudaMalloc\n", size);
        return real_cudaMalloc(ptr, size);
    }

    void *dev = NULL;
    cudaError_t err = real_cudaMallocManaged(&dev, size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        LOGE("cudaMallocManaged(%zu) failed: %d\n", size, err);
        server_free(id);
        pthread_mutex_unlock(&g_lock);
        return err;
    }

    // weights are read-only during inference: mark read-mostly so evicted
    // pages are dropped, not written back to host (halves paging traffic)
    const char *rm = getenv("MEMHOOK_READMOSTLY");
    if ((!rm || atoi(rm)) && real_cudaMemAdvise) {
        int dev_id = 0;
        if (real_cudaGetDevice) real_cudaGetDevice(&dev_id);
        (void) real_cudaMemAdvise(dev, size, cudaMemAdviseSetReadMostly, dev_id);
    }

    struct rec *r = calloc(1, sizeof(*r));
    r->devptr = dev; r->host = NULL; r->size = size; r->id = id; r->paged = 1;
    r->next = g_recs; g_recs = r;
    pthread_mutex_unlock(&g_lock);

    *ptr = dev;
    LOGV("cudaMalloc(%.2f MiB) -> PAGED id=%llu ptr=%p\n",
         size / 1048576.0, (unsigned long long) id, dev);
    return cudaSuccess;
}

// zerocopy: map the server's shm object and let the GPU read it in place
static cudaError_t alloc_zerocopy(void **ptr, size_t size) {
    if (!real_cudaHostRegister || !real_cudaHostGetDevicePointer) {
        return real_cudaMalloc(ptr, size);
    }

    pthread_mutex_lock(&g_lock);
    char name[64];
    uint64_t id = 0; size_t granted = 0;
    if (server_alloc(size, 0, name, sizeof(name), &id, &granted) != 0 || name[0] == '\0') {
        pthread_mutex_unlock(&g_lock);
        LOGE("server_alloc(%zu) failed, falling back to real cudaMalloc\n", size);
        return real_cudaMalloc(ptr, size);
    }

    int fd = shm_open(name, O_RDWR, 0600);
    if (fd < 0) {
        LOGE("shm_open(%s) failed: %s\n", name, strerror(errno));
        server_free(id); pthread_mutex_unlock(&g_lock);
        return cudaErrorMemoryAllocation;
    }
    void *host = mmap(NULL, granted, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (host == MAP_FAILED) {
        LOGE("mmap(%zu) failed: %s\n", granted, strerror(errno));
        server_free(id); pthread_mutex_unlock(&g_lock);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t err = real_cudaHostRegister(host, granted, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        LOGE("cudaHostRegister(%zu) failed: %d (raise 'ulimit -l')\n", granted, err);
        munmap(host, granted); server_free(id); pthread_mutex_unlock(&g_lock);
        return err;
    }
    void *dev = NULL;
    err = real_cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess) {
        LOGE("cudaHostGetDevicePointer failed: %d\n", err);
        real_cudaHostUnregister(host); munmap(host, granted);
        server_free(id); pthread_mutex_unlock(&g_lock);
        return err;
    }

    struct rec *r = calloc(1, sizeof(*r));
    r->devptr = dev; r->host = host; r->size = granted; r->id = id; r->paged = 0;
    r->next = g_recs; g_recs = r;
    pthread_mutex_unlock(&g_lock);

    *ptr = dev;
    LOGV("cudaMalloc(%.2f MiB) -> ZEROCOPY id=%llu dev=%p\n",
         size / 1048576.0, (unsigned long long) id, dev);
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **ptr, size_t size) {
    pthread_once(&g_once, init_once);
    if (size < g_min) return real_cudaMalloc(ptr, size);
    return g_paged ? alloc_paged(ptr, size) : alloc_zerocopy(ptr, size);
}

cudaError_t cudaFree(void *ptr) {
    pthread_once(&g_once, init_once);

    pthread_mutex_lock(&g_lock);
    struct rec **pp = &g_recs, *r = NULL;
    while (*pp) {
        if ((*pp)->devptr == ptr) { r = *pp; *pp = r->next; break; }
        pp = &(*pp)->next;
    }
    if (!r) {
        pthread_mutex_unlock(&g_lock);
        return real_cudaFree(ptr);
    }
    if (r->paged) {
        real_cudaFree(r->devptr);
    } else {
        real_cudaHostUnregister(r->host);
        munmap(r->host, r->size);
    }
    server_free(r->id);
    uint64_t id = r->id;
    pthread_mutex_unlock(&g_lock);

    LOGV("cudaFree(ptr=%p) -> id=%llu released\n", ptr, (unsigned long long) id);
    free(r);
    return cudaSuccess;
}
