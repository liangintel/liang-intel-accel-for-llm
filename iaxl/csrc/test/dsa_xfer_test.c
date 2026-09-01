// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Raw DSA <-> GPU transfer ceiling across N GPUs and M work queues, no compression.
//
// Two limits of the production path are deliberately removed here:
//   * torch_ext funnels every transfer through one d2h_queue / h2d_queue worker thread;
//     this probe owns one pthread per work queue instead.
//   * dsa_memcpy_batch submits a single batch descriptor and then blocks on it, so a
//     256-entry work queue never holds more than one batch; --inflight keeps several
//     outstanding. --inflight 1 reproduces the production behaviour for comparison.
//
// Chunks are interleaved across GPUs so that a single batch spans every device.

#include <cuda.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/idxd.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#include "dsa_gd.h"
#include "env.h"

#define MAX_WQ 16
#define MAX_GPU 16
#define MAX_NODE 8
#define MAX_INFLIGHT 16
#define PORTAL_SIZE 0x1000
#define DSA_ALIGN 8u
#define WAIT_TIMEOUT_NS (10LL * 1000 * 1000 * 1000)

#define DIE(...)                                                                                   \
    do {                                                                                           \
        fprintf(stderr, "[dsa_xfer] " __VA_ARGS__);                                                \
        exit(1);                                                                                   \
    } while (0)

// ---------------------------------------------------------------------- NUMA
// Done with raw syscalls so the probe does not have to link libnuma.

#define MPOL_BIND 2
#define MPOL_MF_MOVE (1 << 1)

static int sysfs_int(const char *path) {
    char buf[32];
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    ssize_t r = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (r <= 0)
        return -1;
    buf[r] = '\0';
    return atoi(buf);
}

static void bind_pages(void *addr, size_t len, int node) {
    unsigned long mask = 1UL << node;
    if (node < 0)
        return;
    if (syscall(SYS_mbind, addr, len, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_MOVE) != 0)
        fprintf(stderr, "[dsa_xfer] warning: mbind to node %d failed: %s\n", node,
                strerror(errno));
}

// Restrict the calling thread to every CPU of "node" (both hyperthread siblings).
static void pin_to_node(int node) {
    char path[64], list[1024];
    cpu_set_t set;
    int fd;
    ssize_t r;

    if (node < 0)
        return;
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    fd = open(path, O_RDONLY);
    if (fd < 0)
        return;
    r = read(fd, list, sizeof(list) - 1);
    close(fd);
    if (r <= 0)
        return;
    list[r] = '\0';

    CPU_ZERO(&set);
    for (char *save = NULL, *tok = strtok_r(list, ",\n", &save); tok;
         tok = strtok_r(NULL, ",\n", &save)) {
        int lo = atoi(tok), hi = lo;
        char *dash = strchr(tok, '-');
        if (dash)
            hi = atoi(dash + 1);
        for (int c = lo; c <= hi && c < CPU_SETSIZE; c++)
            CPU_SET(c, &set);
    }
    sched_setaffinity(0, sizeof(set), &set);
}

#define CU_CHECK(call)                                                                             \
    do {                                                                                           \
        CUresult _r = (call);                                                                      \
        if (_r != CUDA_SUCCESS) {                                                                  \
            const char *_s = NULL;                                                                 \
            cuGetErrorString(_r, &_s);                                                             \
            DIE("%s failed: %s\n", #call, _s ? _s : "?");                                          \
        }                                                                                          \
    } while (0)

// ---------------------------------------------------------------- work queues

static char g_wq_name[MAX_WQ][32];
static void *g_portal[MAX_WQ];
static int g_wq_node[MAX_WQ];
static size_t g_nwq;
static size_t g_max_batch = 128;
static size_t g_max_xfer = 2147483648u;

static inline void movdir64b(const void *desc, volatile void *portal) {
    asm volatile(".byte 0x66, 0x0f, 0x38, 0xf8, 0x02\t\n" : : "a"(portal), "d"(desc));
}

static size_t read_wq_attr(const char *wq, const char *attr, size_t fallback) {
    char path[128], buf[32];
    int fd;
    ssize_t r;
    unsigned long long v;

    snprintf(path, sizeof(path), "/sys/bus/dsa/devices/%s/%s", wq, attr);
    fd = open(path, O_RDONLY);
    if (fd < 0)
        return fallback;
    r = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (r <= 0)
        return fallback;
    buf[r] = '\0';
    v = strtoull(buf, NULL, 0);
    return v ? (size_t)v : fallback;
}

static void wq_open_all(const char *list) {
    char buf[MAX_WQ * 32];
    char *tok, *save;

    snprintf(buf, sizeof(buf), "%s", list);
    for (tok = strtok_r(buf, ", \t", &save); tok && g_nwq < MAX_WQ; tok = strtok_r(NULL, ", \t", &save))
        snprintf(g_wq_name[g_nwq++], 32, "%s", tok);
    if (!g_nwq)
        DIE("no work queue given\n");

    g_max_xfer = read_wq_attr(g_wq_name[0], "max_transfer_size", g_max_xfer);
    g_max_batch = read_wq_attr(g_wq_name[0], "max_batch_size", g_max_batch);

    for (size_t w = 0; w < g_nwq; w++) {
        char path[64];
        snprintf(path, sizeof(path), "/sys/bus/dsa/devices/dsa%d/numa_node", atoi(g_wq_name[w] + 2));
        g_wq_node[w] = sysfs_int(path);

        snprintf(path, sizeof(path), "/dev/dsa/%s", g_wq_name[w]);
        int fd = open(path, O_RDWR);
        if (fd < 0)
            DIE("open %s failed: %s\n", path, strerror(errno));
        // Needs CAP_SYS_RAWIO on kernels carrying the INTEL-SA-01084 fix; run as root.
        g_portal[w] = mmap(NULL, PORTAL_SIZE, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
        close(fd);
        if (g_portal[w] == MAP_FAILED)
            DIE("mmap portal %s failed: %s (run as root)\n", path, strerror(errno));
    }
}

// ------------------------------------------------------------- transfer plan

static void **g_dst;
static void **g_src;
static size_t *g_len;
static size_t g_count;

// Batches are grouped per NUMA node; a work queue only drains the group belonging
// to its own socket, so a node-0 DSA never moves a node-1 GPU's data over UPI.
static size_t *g_batch_off;   // first chunk index of each batch
static size_t *g_batch_cnt;   // descriptors in each batch
static size_t g_nbatches;
static size_t g_group_first[MAX_NODE];  // first batch index of a node's group
static size_t g_group_n[MAX_NODE];      // batches in a node's group
static atomic_size_t g_cursor[MAX_NODE];
static int g_wq_group[MAX_WQ];          // which group each work queue drains
static int g_inflight = 4;

struct slot {
    struct dsa_hw_desc *sub;
    struct dsa_completion_record *comp;
    struct dsa_completion_record *bcomp;
    struct dsa_hw_desc *bdesc;
};

static struct slot g_slot[MAX_WQ][MAX_INFLIGHT];
static int g_failed;

static void slots_alloc(void) {
    for (size_t w = 0; w < g_nwq; w++) {
        for (int s = 0; s < g_inflight; s++) {
            struct slot *sl = &g_slot[w][s];
            if (posix_memalign((void **)&sl->sub, 64, g_max_batch * sizeof(*sl->sub)) ||
                posix_memalign((void **)&sl->comp, 32, g_max_batch * sizeof(*sl->comp)) ||
                posix_memalign((void **)&sl->bcomp, 32, sizeof(*sl->bcomp)) ||
                posix_memalign((void **)&sl->bdesc, 64, sizeof(*sl->bdesc)))
                DIE("descriptor allocation failed\n");
        }
    }
}

static void submit_batch(struct slot *sl, size_t b, void *portal) {
    size_t done = g_batch_off[b];
    size_t cnt = g_batch_cnt[b];

    memset(sl->sub, 0, cnt * sizeof(*sl->sub));
    for (size_t j = 0; j < cnt; j++) {
        sl->sub[j].opcode = DSA_OPCODE_MEMMOVE;
        sl->sub[j].flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR;
        sl->sub[j].completion_addr = (uint64_t)&sl->comp[j];
        sl->sub[j].src_addr = (uint64_t)g_src[done + j];
        sl->sub[j].dst_addr = (uint64_t)g_dst[done + j];
        sl->sub[j].xfer_size = (uint32_t)g_len[done + j];
        sl->comp[j].status = 0;
    }
    sl->bcomp->status = 0;

    if (cnt == 1) {
        sl->sub[0].completion_addr = (uint64_t)sl->bcomp;
        __builtin_ia32_sfence();
        movdir64b(&sl->sub[0], portal);
    } else {
        memset(sl->bdesc, 0, sizeof(*sl->bdesc));
        sl->bdesc->opcode = DSA_OPCODE_BATCH;
        sl->bdesc->flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR;
        sl->bdesc->desc_list_addr = (uint64_t)sl->sub;
        sl->bdesc->desc_count = (uint32_t)cnt;
        sl->bdesc->completion_addr = (uint64_t)sl->bcomp;
        __builtin_ia32_sfence();
        movdir64b(sl->bdesc, portal);
    }
}

static void wait_batch(struct slot *sl) {
    volatile uint8_t *st = &sl->bcomp->status;
    struct timespec t0, now;
    unsigned iter = 0;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (*st == 0) {
        _mm_pause();
        if (++iter == 4096u) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            int64_t ns = (now.tv_sec - t0.tv_sec) * 1000000000LL + (now.tv_nsec - t0.tv_nsec);
            if (ns > WAIT_TIMEOUT_NS)
                DIE("completion poll timed out\n");
            iter = 0;
        }
    }
    if (*st != DSA_COMP_SUCCESS) {
        fprintf(stderr, "[dsa_xfer] batch failed, status=0x%x\n", *st);
        g_failed = 1;
    }
}

// ------------------------------------------------------------------- workers

static pthread_barrier_t g_start, g_done;
static volatile int g_stop;

// Home group first; once it is drained the queue steals from the other socket so
// that an uneven GPU/WQ split (or a single WQ serving both sockets) still finishes.
static int next_batch(int home, size_t *out) {
    size_t b = atomic_fetch_add(&g_cursor[home], 1);

    if (b < g_group_first[home] + g_group_n[home]) {
        *out = b;
        return 1;
    }
    for (int n = 0; n < MAX_NODE; n++) {
        if (n == home || !g_group_n[n])
            continue;
        b = atomic_fetch_add(&g_cursor[n], 1);
        if (b < g_group_first[n] + g_group_n[n]) {
            *out = b;
            return 1;
        }
    }
    return 0;
}

static void drive_wq(size_t w) {
    struct slot *sl = g_slot[w];
    int grp = g_wq_group[w];
    int head = 0, tail = 0, live = 0;

    for (;;) {
        while (live < g_inflight) {
            size_t b;
            if (!next_batch(grp, &b))
                break;
            submit_batch(&sl[head], b, g_portal[w]);
            head = (head + 1) % g_inflight;
            live++;
        }
        if (!live)
            return;
        wait_batch(&sl[tail]);
        tail = (tail + 1) % g_inflight;
        live--;
    }
}

static void *worker(void *arg) {
    size_t w = (size_t)(uintptr_t)arg;
    pin_to_node(g_wq_node[w]);
    for (;;) {
        pthread_barrier_wait(&g_start);
        if (g_stop)
            return NULL;
        drive_wq(w);
        pthread_barrier_wait(&g_done);
    }
}

static double run_once(void) {
    struct timespec a, b;

    for (int n = 0; n < MAX_NODE; n++)
        atomic_store(&g_cursor[n], g_group_first[n]);
    pthread_barrier_wait(&g_start);
    clock_gettime(CLOCK_MONOTONIC, &a);
    pthread_barrier_wait(&g_done);
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

// ---------------------------------------------------------------------- main

struct gpu {
    CUcontext ctx;
    CUdeviceptr raw;
    CUdeviceptr base;  // 64 KiB aligned, what gdrcopy pins
    char *bar;         // CPU-visible BAR address of base
    char *host;
    int node;          // NUMA node the GPU hangs off
};

int main(int argc, char **argv) {
    int ngpu = 1, iters = 7, verify = 1;
    size_t chunk = 32768, mib_per_gpu = 64;
    const char *wqs = getenv("IAXL_DSA_WQS") ? getenv("IAXL_DSA_WQS") : "wq0.0";
    const char *dir = "both";

    static struct option opts[] = {{"gpus", 1, 0, 'g'},   {"wqs", 1, 0, 'w'},
                                   {"chunk", 1, 0, 'c'},  {"mib-per-gpu", 1, 0, 'm'},
                                   {"iters", 1, 0, 'i'},  {"inflight", 1, 0, 'f'},
                                   {"dir", 1, 0, 'd'},    {"no-verify", 0, 0, 'n'},
                                   {0, 0, 0, 0}};
    int o;
    while ((o = getopt_long(argc, argv, "g:w:c:m:i:f:d:n", opts, NULL)) != -1) {
        switch (o) {
        case 'g': ngpu = atoi(optarg); break;
        case 'w': wqs = optarg; break;
        case 'c': chunk = (size_t)atoll(optarg); break;
        case 'm': mib_per_gpu = (size_t)atoll(optarg); break;
        case 'i': iters = atoi(optarg); break;
        case 'f': g_inflight = atoi(optarg); break;
        case 'd': dir = optarg; break;
        case 'n': verify = 0; break;
        default: return 2;
        }
    }
    if (ngpu < 1 || ngpu > MAX_GPU)
        DIE("--gpus must be 1..%d\n", MAX_GPU);
    if (g_inflight < 1 || g_inflight > MAX_INFLIGHT)
        DIE("--inflight must be 1..%d\n", MAX_INFLIGHT);
    if (chunk % DSA_ALIGN)
        DIE("--chunk must be a multiple of %u\n", DSA_ALIGN);

    envs_init();
    wq_open_all(wqs);
    if (chunk > g_max_xfer)
        DIE("--chunk %zu exceeds max_transfer_size %zu\n", chunk, g_max_xfer);

    const size_t bytes = mib_per_gpu * 1024 * 1024;
    const size_t per_gpu = bytes / chunk;
    if (!per_gpu)
        DIE("--mib-per-gpu too small for --chunk\n");

    struct gpu gpus[MAX_GPU];
    CU_CHECK(cuInit(0));
    int ndev = 0;
    CU_CHECK(cuDeviceGetCount(&ndev));
    if (ndev < ngpu)
        DIE("only %d GPU(s) visible, need %d\n", ndev, ngpu);

    for (int g = 0; g < ngpu; g++) {
        CUdevice dev;
        CU_CHECK(cuDeviceGet(&dev, g));
        CU_CHECK(cuDevicePrimaryCtxRetain(&gpus[g].ctx, dev));
        CU_CHECK(cuCtxSetCurrent(gpus[g].ctx));
        // Over-allocate so the pinned base can be pushed up to a 64 KiB GPU page.
        CU_CHECK(cuMemAlloc(&gpus[g].raw, bytes + GD_GPU_PAGE_SIZE));
        gpus[g].base = (gpus[g].raw + GD_GPU_PAGE_SIZE - 1) & GD_GPU_PAGE_MASK;
        CU_CHECK(cuMemsetD8(gpus[g].base, (unsigned char)(0x40 + g), bytes));
        CU_CHECK(cuCtxSynchronize());

        if (dsa_gd_default_register((uint64_t)gpus[g].base, bytes) != 0)
            DIE("dsa_gd_default_register failed for GPU %d\n", g);
        void *bar = NULL;
        if (dsa_gd_default_gpu_bar_addr((uint64_t)gpus[g].base, bytes, &bar) != 0)
            DIE("dsa_gd_default_gpu_bar_addr failed for GPU %d\n", g);
        gpus[g].bar = bar;

        if (posix_memalign((void **)&gpus[g].host, 4096, bytes))
            DIE("host allocation failed\n");
        // Keep the staging buffer on the GPU's own socket, otherwise the numbers
        // measure UPI bandwidth rather than DSA bandwidth.
        char bus[32], pcipath[96];
        CU_CHECK(cuDeviceGetPCIBusId(bus, sizeof(bus), dev));
        for (char *p = bus; *p; p++)
            *p = (char)tolower((unsigned char)*p);
        snprintf(pcipath, sizeof(pcipath), "/sys/bus/pci/devices/%s/numa_node", bus);
        gpus[g].node = sysfs_int(pcipath);
        bind_pages(gpus[g].host, bytes, gpus[g].node);
        memset(gpus[g].host, 0, bytes);  // fault the pages in; DSA cannot resolve them
    }

    g_count = per_gpu * (size_t)ngpu;
    g_dst = malloc(g_count * sizeof(*g_dst));
    g_src = malloc(g_count * sizeof(*g_src));
    g_len = malloc(g_count * sizeof(*g_len));
    g_batch_off = malloc((g_count / g_max_batch + MAX_NODE + 1) * sizeof(*g_batch_off));
    g_batch_cnt = malloc((g_count / g_max_batch + MAX_NODE + 1) * sizeof(*g_batch_cnt));
    if (!g_dst || !g_src || !g_len || !g_batch_off || !g_batch_cnt)
        DIE("plan allocation failed\n");

    // Group the chunk list by the GPU's NUMA node, then cut each group into
    // batches; a batch therefore never mixes sockets.
    size_t group_start[MAX_NODE] = {0}, group_chunks[MAX_NODE] = {0};
    int seen = 0;
    for (int n = 0; n < MAX_NODE; n++) {
        int members = 0;
        for (int g = 0; g < ngpu; g++)
            if (gpus[g].node == n)
                members++;
        group_start[n] = (size_t)seen * per_gpu;
        group_chunks[n] = (size_t)members * per_gpu;
        seen += members;
    }
    if (seen != ngpu)
        DIE("could not resolve the NUMA node of every GPU\n");

    for (int n = 0; n < MAX_NODE; n++) {
        g_group_first[n] = g_nbatches;
        for (size_t off = 0; off < group_chunks[n]; off += g_max_batch) {
            size_t left = group_chunks[n] - off;
            g_batch_off[g_nbatches] = group_start[n] + off;
            g_batch_cnt[g_nbatches] = left < g_max_batch ? left : g_max_batch;
            g_nbatches++;
        }
        g_group_n[n] = g_nbatches - g_group_first[n];
    }

    // A work queue whose socket owns no GPU would idle; point it at the busiest group.
    int busiest = 0;
    for (int n = 1; n < MAX_NODE; n++)
        if (g_group_n[n] > g_group_n[busiest])
            busiest = n;
    for (size_t w = 0; w < g_nwq; w++) {
        int n = g_wq_node[w];
        g_wq_group[w] = (n >= 0 && n < MAX_NODE && g_group_n[n]) ? n : busiest;
    }

    slots_alloc();
    pthread_barrier_init(&g_start, NULL, (unsigned)g_nwq + 1);
    pthread_barrier_init(&g_done, NULL, (unsigned)g_nwq + 1);
    pthread_t th[MAX_WQ];
    for (size_t w = 0; w < g_nwq; w++)
        pthread_create(&th[w], NULL, worker, (void *)(uintptr_t)w);

    printf("[dsa_xfer] gpus=%d wqs=%zu (%s) chunk=%zu B payload=%zu MiB/GPU "
           "chunks=%zu batches=%zu inflight=%d max_batch=%zu iters=%d\n",
           ngpu, g_nwq, wqs, chunk, mib_per_gpu, g_count, g_nbatches, g_inflight, g_max_batch,
           iters);
    printf("[dsa_xfer] topology:");
    for (int n = 0; n < MAX_NODE; n++) {
        if (!g_group_n[n])
            continue;
        printf(" node%d{gpu", n);
        for (int g = 0; g < ngpu; g++)
            if (gpus[g].node == n)
                printf(" %d", g);
        printf(" |");
        for (size_t w = 0; w < g_nwq; w++)
            if (g_wq_group[w] == n)
                printf(" %s", g_wq_name[w]);
        printf(" }");
    }
    printf("\n");

    const double total_gb = (double)g_count * chunk / 1e9;
    const int do_d2h = strcmp(dir, "h2d") != 0;
    const int do_h2d = strcmp(dir, "d2h") != 0;

    for (int pass = 0; pass < 2; pass++) {
        const int h2d = pass;
        if (h2d ? !do_h2d : !do_d2h)
            continue;

        // Node-grouped, then round-robin over the GPUs of that node, so a batch
        // stays on one socket while still spanning every GPU it can reach.
        size_t k = 0;
        for (int n = 0; n < MAX_NODE; n++) {
            int members[MAX_GPU], nm = 0;
            for (int g = 0; g < ngpu; g++)
                if (gpus[g].node == n)
                    members[nm++] = g;
            for (size_t i = 0; i < per_gpu; i++) {
                for (int j = 0; j < nm; j++) {
                    int g = members[j];
                    char *dev = gpus[g].bar + i * chunk;
                    char *hst = gpus[g].host + i * chunk;
                    g_dst[k] = h2d ? dev : hst;
                    g_src[k] = h2d ? hst : dev;
                    g_len[k] = chunk;
                    k++;
                }
            }
        }
        if (h2d)
            for (int g = 0; g < ngpu; g++)
                memset(gpus[g].host, (unsigned char)(0x80 + g), bytes);

        double best = 1e30, sum = 0;
        for (int it = 0; it < iters; it++) {
            double s = run_once();
            if (s < best)
                best = s;
            sum += s;
        }
        if (g_failed)
            DIE("one or more DSA batches failed\n");

        printf("  %-4s best %8.3f GB/s (%7.3f GB/s per GPU, %6.3f us/chunk)   "
               "avg %8.3f GB/s\n",
               h2d ? "H2D" : "D2H", total_gb / best, total_gb / best / ngpu,
               best * 1e6 / (double)g_count, total_gb / (sum / iters));
        // Machine-readable duplicate for sweep scripts.
        printf("RESULT\t%s\t%d\t%zu\t%.3f\t%.3f\t%.4f\n", h2d ? "H2D" : "D2H", ngpu, g_nwq,
               total_gb / best, total_gb / (sum / iters), best * 1e6 / (double)g_count);

        if (verify) {
            for (int g = 0; g < ngpu; g++) {
                if (h2d) {
                    unsigned char got = 0, want = (unsigned char)(0x80 + g);
                    CU_CHECK(cuCtxSetCurrent(gpus[g].ctx));
                    CU_CHECK(cuMemcpyDtoH(&got, gpus[g].base + bytes - 1, 1));
                    if (got != want)
                        DIE("H2D verification failed on GPU %d: 0x%02x != 0x%02x\n", g, got, want);
                } else {
                    unsigned char want = (unsigned char)(0x40 + g);
                    if ((unsigned char)gpus[g].host[bytes - 1] != want)
                        DIE("D2H verification failed on GPU %d\n", g);
                }
            }
        }
    }
    if (verify)
        printf("  verification: passed\n");

    g_stop = 1;
    pthread_barrier_wait(&g_start);
    for (size_t w = 0; w < g_nwq; w++)
        pthread_join(th[w], NULL);
    return 0;
}
