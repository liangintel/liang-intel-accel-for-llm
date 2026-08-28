// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Intel IAA (In-Memory Analytics Accelerator) DEFLATE backend built on Intel QPL.
// It mirrors the qat_zip / cpu_zip slot API: a slot maps to one QPL job, and every
// instance owns `queue_depth` jobs that can be in flight at the same time.
// The produced streams are raw DEFLATE, so they stay interchangeable with the
// QAT and zlib backends.

#include <ctype.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qpl/qpl.h"

#include "env.h"
#include "iaa_zip.h"
#include "iaxl_common.h"

#define MAX_NUMA_NODES 16
#define MAX_INSTANCES 256
#define MAX_QUEUE_DEPTH 4
#define BUSY_RETRY_LIMIT 1000000

typedef struct {
    qpl_job *job;
    uint8_t *in;
    uint8_t *out;
    int submitted;
} IaaSlot;

typedef struct {
    int numa_id;
    IaaSlot slot[MAX_QUEUE_DEPTH];
} IaaInstance;

// A NUMA node and the number of IAA devices attached to it. QPL only steers a job to a node
// (qpl_job::numa_id) and picks the device itself, so "instances per device" is realised by
// creating instances_per_device * devices instances bound to that node.
typedef struct {
    int node;
    int devices;
} IaaNode;

static IaaInstance g_inst[MAX_INSTANCES];
static int g_inst_count;
static int g_queue_depth = 1;
static int g_initialized;
static uint32_t g_src_cap;
static uint32_t g_dst_cap;
static uint32_t g_buf_cap;

// Tallies the IAA devices owned by each NUMA node. Spreading jobs over all of them is what
// makes throughput scale, so "auto" must not collapse to one node.
static int discover_iaa_nodes(IaaNode *out, int max_nodes) {
    DIR *dir = opendir("/sys/bus/dsa/devices");
    if (!dir)
        return 0;

    int n = 0;
    for (struct dirent *ent = readdir(dir); ent; ent = readdir(dir)) {
        if (strncmp(ent->d_name, "iax", 3) != 0 || !isdigit((unsigned char)ent->d_name[3]))
            continue;

        char path[320];
        snprintf(path, sizeof(path), "/sys/bus/dsa/devices/%s/numa_node", ent->d_name);
        FILE *fp = fopen(path, "r");
        if (!fp)
            continue;
        int node = -1;
        const int read = fscanf(fp, "%d", &node);
        fclose(fp);
        if (read != 1 || node < 0)
            continue;

        int i = 0;
        for (; i < n; i++)
            if (out[i].node == node)
                break;
        if (i == n) {
            if (n == max_nodes)
                continue;
            out[n].node = node;
            out[n].devices = 0;
            n++;
        }
        out[i].devices++;
    }
    closedir(dir);

    for (int i = 1; i < n; i++) {
        const IaaNode key = out[i];
        int j = i - 1;
        for (; j >= 0 && out[j].node > key.node; j--)
            out[j + 1] = out[j];
        out[j + 1] = key;
    }
    return n;
}

// Parses the comma separated NUMA node list used to steer job submission and attaches each
// node's IAA device count. "auto" (the default) uses every NUMA node that owns an IAA device.
static int select_iaa_nodes(const char *env, IaaNode *out, int max_nodes) {
    IaaNode found[MAX_NUMA_NODES];
    const int found_count = discover_iaa_nodes(found, MAX_NUMA_NODES);

    if (!env || !*env || !strcasecmp(env, "auto")) {
        IAXL_CHECK(found_count > 0, "iaa_zip: no IAA device found under /sys/bus/dsa/devices");
        const int n = found_count < max_nodes ? found_count : max_nodes;
        for (int i = 0; i < n; i++)
            out[i] = found[i];
        return n;
    }

    char buf[256];
    strncpy(buf, env, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    int n = 0;
    for (char *tok = strtok(buf, ","); tok && n < max_nodes; tok = strtok(NULL, ",")) {
        const int node = atoi(tok);
        IAXL_CHECK(node >= 0, "iaa_zip: NUMA node index must be non-negative");
        out[n].node = node;
        out[n].devices = 0;
        for (int i = 0; i < found_count; i++)
            if (found[i].node == node)
                out[n].devices = found[i].devices;
        IAXL_CHECK(out[n].devices > 0, "iaa_zip: selected NUMA node owns no IAA device");
        n++;
    }
    IAXL_CHECK(n > 0, "iaa_zip: no valid NUMA node selected");
    return n;
}

static void *alloc_aligned(size_t size) {
    return aligned_alloc(64, (size + 63) & ~(size_t)63);
}

static void slot_teardown(IaaSlot *sl) {
    if (sl->job) {
        qpl_fini_job(sl->job);
        free(sl->job);
        sl->job = NULL;
    }
    free(sl->in);
    free(sl->out);
    sl->in = NULL;
    sl->out = NULL;
    sl->submitted = 0;
}

static int slot_init(IaaSlot *sl, uint32_t job_size, int numa_id) {
    sl->job = alloc_aligned(job_size);
    sl->in = alloc_aligned(g_buf_cap);
    sl->out = alloc_aligned(g_buf_cap);
    if (!sl->job || !sl->in || !sl->out)
        return -1;
    // IAA cannot resolve page faults itself, so fault the DMA buffers in up front.
    memset(sl->in, 0, g_buf_cap);
    memset(sl->out, 0, g_buf_cap);
    if (qpl_init_job(qpl_path_hardware, sl->job) != QPL_STS_OK)
        return -1;
    sl->job->numa_id = numa_id;
    sl->submitted = 0;
    return 0;
}

int iaa_zip_init(void) {
    if (g_initialized)
        return 0;

    int instances_per_device = envs.IAXL_IAA_ZIP_INSTANCES_PER_DEVICE;
    g_queue_depth = envs.IAXL_IAA_ZIP_QUEUE_DEPTH;
    g_src_cap = (uint32_t)envs.IAXL_ZIP_SRC_CAP;
    g_dst_cap = (uint32_t)envs.IAXL_ZIP_DST_CAP;
    g_buf_cap = g_src_cap > g_dst_cap ? g_src_cap : g_dst_cap;

    if (instances_per_device < 1)
        instances_per_device = 1;
    if (g_queue_depth < 1)
        g_queue_depth = 1;
    if (g_queue_depth > MAX_QUEUE_DEPTH)
        g_queue_depth = MAX_QUEUE_DEPTH;

    IaaNode nodes[MAX_NUMA_NODES];
    const char *dev_sel = envs.IAXL_IAA_DEVICES();
    const int node_count = select_iaa_nodes(dev_sel, nodes, MAX_NUMA_NODES);

    int device_count = 0, max_per_node = 0;
    for (int n = 0; n < node_count; n++) {
        device_count += nodes[n].devices;
        const int want = nodes[n].devices * instances_per_device;
        if (want > max_per_node)
            max_per_node = want;
    }

    printf("[iaa_zip] config: numa_nodes=%s (%d node(s), %d device(s)) instances/device=%d "
           "src_cap=%u B dst_cap=%u B queue_depth=%d\n",
           dev_sel ? dev_sel : "auto", node_count, device_count, instances_per_device, g_src_cap,
           g_dst_cap, g_queue_depth);

    uint32_t job_size = 0;
    IAXL_CHECK(qpl_get_job_size(qpl_path_hardware, &job_size) == QPL_STS_OK,
               "iaa_zip: qpl_get_job_size failed");

    // Interleave nodes so that consumers taking only the first N instances still
    // spread their jobs over every NUMA node.
    for (int i = 0; i < max_per_node; i++) {
        for (int n = 0; n < node_count && g_inst_count < MAX_INSTANCES; n++) {
            if (i >= nodes[n].devices * instances_per_device)
                continue;
            IaaInstance *in = &g_inst[g_inst_count++];
            in->numa_id = nodes[n].node;
            for (int si = 0; si < g_queue_depth; si++) {
                if (slot_init(&in->slot[si], job_size, in->numa_id) != 0) {
                    fprintf(stderr, "[iaa_zip] instance %d slot %d initialization failed\n",
                            g_inst_count - 1, si);
                    iaa_zip_shutdown();
                    return -1;
                }
            }
        }
    }
    IAXL_CHECK(g_inst_count > 0, "iaa_zip: no usable instance");

    printf("[iaa_zip] ready: instances=%d slots=%d\n", g_inst_count,
           g_inst_count * g_queue_depth);
    g_initialized = 1;
    return 0;
}

int iaa_zip_num_slots(void) { return g_inst_count * g_queue_depth; }
int iaa_zip_queue_depth(void) { return g_queue_depth; }
int iaa_zip_src_cap(void) { return (int)g_src_cap; }

static IaaSlot *resolve_slot(int slot) {
    if (slot < 0 || slot >= g_inst_count * g_queue_depth)
        return NULL;
    return &g_inst[slot / g_queue_depth].slot[slot % g_queue_depth];
}

static int submit_slot(int slot, int compress, void *src, int len) {
    IaaSlot *sl = resolve_slot(slot);
    if (!sl || !src || len <= 0)
        return -1;

    const uint32_t input_cap = compress ? g_src_cap : g_dst_cap;
    const uint32_t output_cap = compress ? g_dst_cap : g_src_cap;
    if ((uint32_t)len > input_cap)
        return -1;

    memcpy(sl->in, src, (size_t)len);

    qpl_job *job = sl->job;
    job->op = compress ? qpl_op_compress : qpl_op_decompress;
    job->level = qpl_default_level;
    job->huffman_table = NULL;
    job->next_in_ptr = sl->in;
    job->available_in = (uint32_t)len;
    job->next_out_ptr = sl->out;
    job->available_out = output_cap;
    job->total_in = 0;
    job->total_out = 0;
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;
    // QPL's compress-and-verify pass rejects single-job DEFLATE streams (QPL_STS_INTL_VERIFY_ERR),
    // so it is skipped here; decompression validates the payload end to end.
    if (compress)
        job->flags |= QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;

    qpl_status status;
    int retries = 0;
    while ((status = qpl_submit_job(job)) == QPL_STS_QUEUES_ARE_BUSY_ERR) {
        if (++retries > BUSY_RETRY_LIMIT)
            return -1;
    }
    if (status != QPL_STS_OK) {
        fprintf(stderr, "[iaa_zip] %s submit failed on slot %d: qpl_status=%d\n",
                compress ? "compress" : "decompress", slot, (int)status);
        return -1;
    }

    sl->submitted = 1;
    return 0;
}

int iaa_zip_compress(int slot, void *src, int len) { return submit_slot(slot, 1, src, len); }
int iaa_zip_decompress(int slot, void *src, int len) { return submit_slot(slot, 0, src, len); }

int iaa_zip_wait(int slot, void **dest, int *len) {
    IaaSlot *sl = resolve_slot(slot);
    if (!sl || !sl->submitted)
        return -1;

    qpl_status status;
    while ((status = qpl_check_job(sl->job)) == QPL_STS_BEING_PROCESSED)
        ;
    sl->submitted = 0;
    if (status != QPL_STS_OK) {
        fprintf(stderr, "[iaa_zip] job failed on slot %d: qpl_status=%d\n", slot, (int)status);
        return -1;
    }

    if (dest)
        *dest = sl->out;
    if (len)
        *len = (int)sl->job->total_out;
    return 0;
}

void iaa_zip_shutdown(void) {
    for (int i = 0; i < g_inst_count; i++)
        for (int si = 0; si < MAX_QUEUE_DEPTH; si++)
            slot_teardown(&g_inst[i].slot[si]);
    g_inst_count = 0;
    g_queue_depth = 1;
    g_initialized = 0;
    g_src_cap = 0;
    g_dst_cap = 0;
    g_buf_cap = 0;
}
