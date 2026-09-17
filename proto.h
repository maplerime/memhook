// Wire protocol between libmemhook (LD_PRELOAD hook) and memserver.
// Control plane is TCP (localhost now, remote later). Data plane is a
// POSIX shared-memory object whose name the server hands back.
#ifndef MEMHOOK_PROTO_H
#define MEMHOOK_PROTO_H

#include <stdint.h>

#define MEMHOOK_PORT  9797
#define MEMHOOK_MAGIC 0x4d454d48u  // "MEMH"

enum {
    OP_ALLOC = 1,
    OP_FREE  = 2,
    OP_WRITE = 3,   // stream mode: push bytes into the server-side master buffer
    OP_READ  = 4,   // stream mode: pull bytes back from the server-side master buffer
};

enum {
    ST_OK       = 0,
    ST_NOMEM    = 1,
    ST_BADREQ   = 2,
    ST_NOTFOUND = 3,
};

enum {
    // paged mode: client backs the bytes with CUDA managed memory and the
    // GPU driver demand-pages them. The server only grants/accounts the
    // budget and returns no shm object (name_len == 0).
    FLAG_PAGED  = 1u,
    // stream mode: server keeps a master byte buffer that the client fills
    // with OP_WRITE over the socket. No shm, no address mapping.
    FLAG_STREAM = 2u,
};

// Fixed request header. x86-64 is little-endian; client and server share it.
struct msg_req {
    uint32_t magic;
    uint32_t op;
    uint64_t size;   // OP_ALLOC: bytes wanted; OP_WRITE: payload length
    uint64_t id;     // OP_FREE / OP_WRITE: allocation id
    uint64_t offset; // OP_WRITE: byte offset into the allocation
    uint32_t flags;  // OP_ALLOC: FLAG_PAGED / FLAG_STREAM
    uint32_t _pad;
};

// Fixed response header. For OP_ALLOC it is followed by name_len raw
// bytes holding the shm object name (no trailing NUL).
struct msg_resp {
    uint32_t magic;
    uint32_t status;
    uint64_t id;        // allocation id (used later for OP_FREE)
    uint64_t size;      // granted size, rounded up to a page
    uint32_t name_len;
    uint32_t _pad;
};

#endif // MEMHOOK_PROTO_H
