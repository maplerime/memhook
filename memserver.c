// memserver - a tiny remote memory pool manager.
//
// It listens on TCP and serves ALLOC/FREE requests from libmemhook.
// For each ALLOC it creates a POSIX shared-memory object (backed by
// /dev/shm) and returns the name. The client maps that name to get the
// actual bytes. So the server owns and tracks the pool; the client just
// borrows it. Control travels over the network, data over shared memory.
//
// build: gcc -O2 -Wall -o memserver memserver.c -lrt -lpthread

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <signal.h>

#include "proto.h"

static uint64_t g_next_id  = 1;
static uint64_t g_total    = 0;   // bytes currently handed out
static uint64_t g_max      = 0;   // 0 = unlimited
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

struct alloc {
    uint64_t id;
    char     name[64];   // zerocopy: shm object name
    int      fd;         // zerocopy: shm fd, else -1
    char    *data;       // stream: master byte buffer, else NULL
    size_t   size;
    struct alloc *next;
};

static size_t page_round(size_t s) {
    long pg = sysconf(_SC_PAGESIZE);
    size_t p = (size_t) pg;
    return (s + p - 1) & ~(p - 1);
}

static void ts(char *buf, size_t n) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    struct tm tm;
    localtime_r(&t.tv_sec, &tm);
    snprintf(buf, n, "%02d:%02d:%02d.%03ld",
             tm.tm_hour, tm.tm_min, tm.tm_sec, t.tv_nsec / 1000000);
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

// serve one client connection until it disconnects
static void *serve(void *arg) {
    int cfd = (int)(intptr_t) arg;
    struct alloc *head = NULL;
    char tb[32];
    uint64_t written = 0, last_logged = 0;   // stream bytes received on this connection

    for (;;) {
        struct msg_req req;
        if (full_read(cfd, &req, sizeof(req)) != 0) break;
        if (req.magic != MEMHOOK_MAGIC) break;

        struct msg_resp resp;
        memset(&resp, 0, sizeof(resp));
        resp.magic = MEMHOOK_MAGIC;

        if (req.op == OP_ALLOC) {
            size_t size = page_round(req.size);

            pthread_mutex_lock(&g_lock);
            int over = (g_max && g_total + size > g_max);
            uint64_t id = over ? 0 : g_next_id++;
            if (!over) g_total += size;
            pthread_mutex_unlock(&g_lock);

            if (over) {
                resp.status = ST_NOMEM;
                full_write(cfd, &resp, sizeof(resp));
                ts(tb, sizeof(tb));
                fprintf(stderr, "%s  ALLOC %10zu B  REJECTED (pool cap reached)\n", tb, size);
                continue;
            }

            int paged  = (req.flags & FLAG_PAGED)  != 0;
            int stream = (req.flags & FLAG_STREAM) != 0;
            char name[64] = "";
            int  fd = -1;
            char *data = NULL;
            const char *tag = "";

            if (stream) {
                data = calloc(1, size);   // master copy (zeroed so weight padding reads back as 0)
                if (!data) {
                    pthread_mutex_lock(&g_lock); g_total -= size; pthread_mutex_unlock(&g_lock);
                    resp.status = ST_NOMEM;
                    full_write(cfd, &resp, sizeof(resp));
                    ts(tb, sizeof(tb));
                    fprintf(stderr, "%s  ALLOC %10zu B  FAILED: malloc\n", tb, size);
                    continue;
                }
                tag = "[stream]";
            } else if (paged) {
                tag = "[paged/managed]";
            } else {
                snprintf(name, sizeof(name), "/memhook.%d.%llu",
                         (int) getpid(), (unsigned long long) id);
                fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
                if (fd < 0 || ftruncate(fd, (off_t) size) != 0) {
                    if (fd >= 0) { close(fd); shm_unlink(name); }
                    pthread_mutex_lock(&g_lock);
                    g_total -= size;
                    pthread_mutex_unlock(&g_lock);
                    resp.status = ST_NOMEM;
                    full_write(cfd, &resp, sizeof(resp));
                    ts(tb, sizeof(tb));
                    fprintf(stderr, "%s  ALLOC %10zu B  FAILED: %s\n", tb, size, strerror(errno));
                    continue;
                }
                tag = name;
            }

            struct alloc *a = calloc(1, sizeof(*a));
            a->id = id; a->fd = fd; a->data = data; a->size = size;
            snprintf(a->name, sizeof(a->name), "%s", name);
            a->next = head; head = a;

            resp.status   = ST_OK;
            resp.id       = id;
            resp.size     = size;
            resp.name_len = (uint32_t) strlen(name);
            if (full_write(cfd, &resp, sizeof(resp)) != 0) break;
            if (resp.name_len && full_write(cfd, name, resp.name_len) != 0) break;

            ts(tb, sizeof(tb));
            fprintf(stderr, "%s  ALLOC %10zu B  id=%llu  %-16s (pool now %.2f GiB)\n",
                    tb, size, (unsigned long long) id, tag, g_total / 1073741824.0);

        } else if (req.op == OP_FREE) {
            struct alloc **pp = &head, *a = NULL;
            while (*pp) {
                if ((*pp)->id == req.id) { a = *pp; *pp = a->next; break; }
                pp = &(*pp)->next;
            }
            if (a) {
                if (a->fd >= 0) { shm_unlink(a->name); close(a->fd); }
                free(a->data);
                pthread_mutex_lock(&g_lock);
                g_total -= a->size;
                pthread_mutex_unlock(&g_lock);
                resp.status = ST_OK;
                resp.id     = req.id;
                ts(tb, sizeof(tb));
                fprintf(stderr, "%s  FREE  %10zu B  id=%llu  (pool now %.2f GiB)\n",
                        tb, a->size, (unsigned long long) req.id, g_total / 1073741824.0);
                free(a);
            } else {
                resp.status = ST_NOTFOUND;
            }
            full_write(cfd, &resp, sizeof(resp));

        } else if (req.op == OP_WRITE) {
            struct alloc *a = NULL;
            for (struct alloc *p = head; p; p = p->next)
                if (p->id == req.id) { a = p; break; }
            size_t   len = req.size;
            uint64_t off = req.offset;
            int ok = (a && a->data && off + len <= a->size);
            if (ok) {
                if (full_read(cfd, a->data + off, len) != 0) break;   // bytes land in master buffer
                written += len;
                if (written >> 30 != last_logged >> 30) {             // log each new GiB
                    last_logged = written;
                    ts(tb, sizeof(tb));
                    fprintf(stderr, "%s  WRITE received %.2f GiB total (over socket)\n",
                            tb, written / 1073741824.0);
                }
                resp.status = ST_OK;
            } else {
                char sink[65536];                                     // drain to keep stream aligned
                size_t rem = len, bad = 0;
                while (rem) {
                    size_t c = rem < sizeof(sink) ? rem : sizeof(sink);
                    if (full_read(cfd, sink, c) != 0) { bad = 1; break; }
                    rem -= c;
                }
                if (bad) break;
                resp.status = ST_NOTFOUND;
            }
            resp.id = req.id;
            full_write(cfd, &resp, sizeof(resp));

        } else if (req.op == OP_READ) {
            struct alloc *a = NULL;
            for (struct alloc *p = head; p; p = p->next)
                if (p->id == req.id) { a = p; break; }
            size_t   len = req.size;
            uint64_t off = req.offset;
            int ok = (a && a->data && off + len <= a->size);
            resp.status = ok ? ST_OK : ST_NOTFOUND;
            resp.id     = req.id;
            if (full_write(cfd, &resp, sizeof(resp)) != 0) break;
            if (ok && full_write(cfd, a->data + off, len) != 0) break;  // send bytes back

        } else {
            resp.status = ST_BADREQ;
            full_write(cfd, &resp, sizeof(resp));
        }
    }

    // client gone: reclaim whatever it did not free
    while (head) {
        struct alloc *a = head; head = a->next;
        if (a->fd >= 0) { shm_unlink(a->name); close(a->fd); }
        free(a->data);
        pthread_mutex_lock(&g_lock);
        g_total -= a->size;
        pthread_mutex_unlock(&g_lock);
        ts(tb, sizeof(tb));
        fprintf(stderr, "%s  RECLAIM id=%llu %s (client left)\n",
                tb, (unsigned long long) a->id, a->name);
        free(a);
    }
    close(cfd);
    return NULL;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);

    int port = MEMHOOK_PORT;
    if (argc > 1) port = atoi(argv[1]);
    const char *cap = getenv("MEMSERVER_MAX_BYTES");
    if (cap) g_max = strtoull(cap, NULL, 10);

    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons((uint16_t) port);
    if (bind(sfd, (struct sockaddr *) &addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    if (listen(sfd, 16) != 0) { perror("listen"); return 1; }

    fprintf(stderr, "memserver: listening on 0.0.0.0:%d  (pool cap: %s)\n",
            port, g_max ? "set" : "unlimited");

    for (;;) {
        struct sockaddr_in ca;
        socklen_t cl = sizeof(ca);
        int cfd = accept(sfd, (struct sockaddr *) &ca, &cl);
        if (cfd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        int nd = 1, buf = 16 << 20;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
        setsockopt(cfd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
        setsockopt(cfd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
        fprintf(stderr, "memserver: client connected from %s:%d\n",
                inet_ntoa(ca.sin_addr), ntohs(ca.sin_port));
        pthread_t th;
        if (pthread_create(&th, NULL, serve, (void *)(intptr_t) cfd) == 0) {
            pthread_detach(th);
        } else {
            close(cfd);
        }
    }
    close(sfd);
    return 0;
}
