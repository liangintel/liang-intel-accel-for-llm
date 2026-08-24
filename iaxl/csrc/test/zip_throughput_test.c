// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Measures raw compression and decompression throughput of the QAT and IAA backends, alone and
// combined, using the same shared-task-pool worker loop as kv_zip but without the torch/kv glue.
// --job splits every chunk into independent jobs the way IAXL_ZIP_JOB_SIZE does, and --cap sets
// the slot capacity, so both costs can be separated from the job size itself.
// The --pack flag adds kv_zip's per-chunk malloc+memcpy so its cost can be isolated.

#include <getopt.h>
#include <omp.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "env.h"
#include "iaa_zip.h"
#include "qat_zip.h"

typedef enum { BACKEND_QAT, BACKEND_IAA } Backend;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// bf16 samples from a normal distribution: the exponent byte is highly repetitive and the
// mantissa byte is close to random, which is what gives real KV cache its ~1.25x ratio.
static void fill_bf16_gaussian(unsigned char *buf, size_t bytes) {
    unsigned long long s = 88172645463325252ULL;
    for (size_t i = 0; i + 1 < bytes; i += 2) {
        double u = 0;
        for (int k = 0; k < 6; k++) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            u += (double)((s >> 11) & 0xffffff) / 16777216.0;
        }
        float f = (float)(u - 3.0);
        unsigned int bits;
        memcpy(&bits, &f, 4);
        unsigned short bf = (unsigned short)(bits >> 16);
        buf[i] = (unsigned char)(bf & 0xff);
        buf[i + 1] = (unsigned char)(bf >> 8);
    }
}

static int submit(Backend b, int slot, void *src, int len) {
    return b == BACKEND_QAT ? qat_zip_compress(slot, src, len) : iaa_zip_compress(slot, src, len);
}

static int submit_dec(Backend b, int slot, void *src, int len) {
    return b == BACKEND_QAT ? qat_zip_decompress(slot, src, len)
                            : iaa_zip_decompress(slot, src, len);
}

static int wait_one(Backend b, int slot, void **dst, int *len) {
    return b == BACKEND_QAT ? qat_zip_wait(slot, dst, len) : iaa_zip_wait(slot, dst, len);
}

struct Job {
    unsigned char *src;
    int len;
};

struct Config {
    int qat_inst, qat_depth;
    int iaa_inst, iaa_depth;
    int pack;
    int decompress;
    size_t chunk;
    size_t njob;
    struct Job *jobs;  // uncompressed input, sliced out of src
    struct Job *cjobs; // compressed input, one owned buffer per job
    unsigned char *src;
};

// Runs one full pass over the job list. When capture is set, each job's output is copied into
// cjobs so a later pass can decompress it.
static double run_pass(const struct Config *c, int capture, size_t *out_bytes) {
    const int workers = c->qat_inst + c->iaa_inst;
    const struct Job *jobs = c->decompress ? c->cjobs : c->jobs;
    _Atomic size_t next = 0;
    _Atomic size_t total = 0;

    const double t0 = now_sec();
#pragma omp parallel num_threads(workers)
    {
        const int t = omp_get_thread_num();
        const Backend backend = t < c->qat_inst ? BACKEND_QAT : BACKEND_IAA;
        const int depth = backend == BACKEND_QAT ? c->qat_depth : c->iaa_depth;
        const int base = (backend == BACKEND_QAT ? t : t - c->qat_inst) * depth;
        int (*post)(Backend, int, void *, int) = c->decompress ? submit_dec : submit;
        size_t local = 0;

        int active = 0;
        size_t slot_job[256];
        if (depth > (int)(sizeof(slot_job) / sizeof(slot_job[0])))
            abort();
        for (int k = 0; k < depth; k++) {
            const size_t i = atomic_fetch_add(&next, 1);
            if (i >= c->njob)
                break;
            if (post(backend, base + k, jobs[i].src, jobs[i].len) != 0)
                abort();
            slot_job[k] = i;
            active++;
        }

        int in_flight = active;
        int draining = 0;
        for (int s = 0; in_flight > 0; s = (s + 1) % active) {
            void *out;
            int out_len;
            if (wait_one(backend, base + s, &out, &out_len) != 0)
                abort();
            local += (size_t)out_len;
            if (capture) {
                unsigned char *b = malloc((size_t)out_len);
                memcpy(b, out, (size_t)out_len);
                c->cjobs[slot_job[s]].src = b;
                c->cjobs[slot_job[s]].len = out_len;
            }
            if (c->pack) {
                char *buf = malloc(sizeof(int) * 2 + (size_t)out_len);
                ((int *)buf)[0] = out_len;
                ((int *)buf)[1] = (int)c->chunk;
                memcpy(buf + sizeof(int) * 2, out, (size_t)out_len);
                free(buf);
            }
            const size_t i = draining ? c->njob : atomic_fetch_add(&next, 1);
            if (i < c->njob) {
                if (post(backend, base + s, jobs[i].src, jobs[i].len) != 0)
                    abort();
                slot_job[s] = i;
            } else {
                draining = 1;
                in_flight--;
            }
        }
        atomic_fetch_add(&total, local);
    }
    const double dt = now_sec() - t0;
    *out_bytes = atomic_load(&total);
    return dt;
}

static double total_src_bytes(const struct Config *c) {
    double bytes = 0;
    for (size_t i = 0; i < c->njob; i++)
        bytes += (double)c->jobs[i].len;
    return bytes;
}

static void report(const char *label, struct Config *c, int reps) {
    double best = 1e9, sum = 0;
    size_t produced = 0;

    if (c->decompress) {
        // Every job must be compressed by a backend that can also decompress it, so capture
        // with QAT alone when it is present.
        const int qat = c->qat_inst, iaa = c->iaa_inst;
        c->decompress = 0;
        if (qat)
            c->iaa_inst = 0;
        run_pass(c, /*capture=*/1, &produced);
        c->qat_inst = qat;
        c->iaa_inst = iaa;
        c->decompress = 1;
    }

    for (int r = 0; r < reps; r++) {
        const double dt = run_pass(c, /*capture=*/0, &produced);
        sum += dt;
        if (dt < best)
            best = dt;
    }

    const double bytes = total_src_bytes(c);
    printf("%-28s workers=%-3d best %6.2f GB/s   avg %6.2f GB/s   ratio %.3fx\n", label,
           c->qat_inst + c->iaa_inst, bytes / best / 1e9, bytes / (sum / reps) / 1e9,
           bytes / (double)produced);
    fflush(stdout);

    if (c->decompress)
        for (size_t i = 0; i < c->njob; i++)
            free(c->cjobs[i].src);
}

int main(int argc, char **argv) {
    int qat_per_dev = 2, iaa_per_dev = 4, depth = 4, reps = 5, mib = 512;
    size_t chunk = 32768, job = 0, cap = 0;
    const char *qat_dev = "0,1,2,3,4,5";
    const char *iaa_dev = "auto";

    static struct option opts[] = {{"qat-per-dev", 1, 0, 'q'}, {"iaa-per-dev", 1, 0, 'a'},
                                   {"qat-devices", 1, 0, 'Q'}, {"iaa-devices", 1, 0, 'A'},
                                   {"depth", 1, 0, 'd'},       {"chunk", 1, 0, 'c'},
                                   {"job", 1, 0, 'j'},         {"cap", 1, 0, 'C'},
                                   {"mib", 1, 0, 'm'},         {"reps", 1, 0, 'r'},
                                   {0, 0, 0, 0}};
    int o;
    while ((o = getopt_long(argc, argv, "q:a:Q:A:d:c:j:C:m:r:", opts, NULL)) != -1) {
        switch (o) {
        case 'q': qat_per_dev = atoi(optarg); break;
        case 'a': iaa_per_dev = atoi(optarg); break;
        case 'Q': qat_dev = optarg; break;
        case 'A': iaa_dev = optarg; break;
        case 'd': depth = atoi(optarg); break;
        case 'c': chunk = (size_t)atoll(optarg); break;
        case 'j': job = (size_t)atoll(optarg); break;
        case 'C': cap = (size_t)atoll(optarg); break;
        case 'm': mib = atoi(optarg); break;
        case 'r': reps = atoi(optarg); break;
        default: return 2;
        }
    }
    if (job == 0 || job > chunk)
        job = chunk;
    if (cap == 0)
        cap = chunk;

    char buf[32];
    snprintf(buf, sizeof(buf), "%zu", cap);
    setenv("IAXL_ZIP_SRC_CAP", buf, 1);
    setenv("IAXL_ZIP_DST_CAP", buf, 1);
    snprintf(buf, sizeof(buf), "%d", depth);
    setenv("IAXL_QAT_ZIP_QUEUE_DEPTH", buf, 1);
    setenv("IAXL_IAA_ZIP_QUEUE_DEPTH", buf, 1);
    snprintf(buf, sizeof(buf), "%d", qat_per_dev);
    setenv("IAXL_QAT_ZIP_INSTANCES_PER_DEVICE", buf, 1);
    snprintf(buf, sizeof(buf), "%d", iaa_per_dev);
    setenv("IAXL_IAA_ZIP_INSTANCES_PER_DEVICE", buf, 1);
    setenv("IAXL_QAT_DEVICES", qat_dev, 1);
    setenv("IAXL_IAA_DEVICES", iaa_dev, 1);
    setenv("IAXL_IAA_ZIP_PATH", "hardware", 1);
    envs_init();

    if (qat_zip_init() != 0 || iaa_zip_init() != 0) {
        fprintf(stderr, "[throughput] backend init failed\n");
        return 1;
    }

    struct Config c = {0};
    c.qat_depth = qat_zip_queue_depth();
    c.iaa_depth = iaa_zip_queue_depth();
    const int qat_inst = qat_zip_num_slots() / c.qat_depth;
    const int iaa_inst = iaa_zip_num_slots() / c.iaa_depth;
    c.chunk = chunk;

    const size_t nchunk = (size_t)mib * 1024 * 1024 / chunk;
    const size_t per_chunk = (chunk + job - 1) / job;
    c.njob = nchunk * per_chunk;

    c.src = aligned_alloc(4096, nchunk * chunk);
    c.jobs = calloc(c.njob, sizeof(*c.jobs));
    c.cjobs = calloc(c.njob, sizeof(*c.cjobs));
    if (!c.src || !c.jobs || !c.cjobs) {
        fprintf(stderr, "[throughput] source allocation failed\n");
        return 1;
    }
    fill_bf16_gaussian(c.src, nchunk * chunk);
    for (size_t i = 0, k = 0; i < nchunk; i++)
        for (size_t off = 0; off < chunk; off += job, k++) {
            c.jobs[k].src = c.src + i * chunk + off;
            c.jobs[k].len = (int)(chunk - off < job ? chunk - off : job);
        }

    printf("[throughput] qat_instances=%d iaa_instances=%d depth=%d/%d chunk=%zu B job=%zu B "
           "cap=%zu B data=%d MiB reps=%d\n\n",
           qat_inst, iaa_inst, c.qat_depth, c.iaa_depth, chunk, job, cap, mib, reps);

    for (int pack = 0; pack <= 1; pack++) {
        c.pack = pack;
        c.decompress = 0;
        printf("--- compress, %s ---\n",
               pack ? "with kv_zip pack() (malloc+memcpy)" : "accelerator only");
        c.qat_inst = qat_inst;
        c.iaa_inst = 0;
        report("QAT only", &c, reps);
        c.qat_inst = 0;
        c.iaa_inst = iaa_inst;
        report("IAA only", &c, reps);
        c.qat_inst = qat_inst;
        c.iaa_inst = iaa_inst;
        report("QAT + IAA", &c, reps);
        printf("\n");
    }

    c.pack = 0;
    c.decompress = 1;
    printf("--- decompress, accelerator only ---\n");
    c.qat_inst = qat_inst;
    c.iaa_inst = 0;
    report("QAT only", &c, reps);
    c.qat_inst = 0;
    c.iaa_inst = iaa_inst;
    report("IAA only", &c, reps);
    if (job <= 4096) {
        c.qat_inst = qat_inst;
        c.iaa_inst = iaa_inst;
        report("QAT + IAA", &c, reps);
    } else {
        printf("%-28s skipped: jobs above 4 KB may exceed the window IAA decodes\n", "QAT + IAA");
    }
    printf("\n");

    free(c.cjobs);
    free(c.jobs);
    free(c.src);
    qat_zip_shutdown();
    iaa_zip_shutdown();
    return 0;
}
