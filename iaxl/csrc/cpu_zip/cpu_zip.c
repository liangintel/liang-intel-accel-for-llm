// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lz4.h>
#include <zlib.h>
#include <zstd.h>

#include "cpu_zip.h"
#include "env.h"

typedef enum { CODEC_DEFLATE = 0, CODEC_LZ4, CODEC_ZSTD } CpuCodec;

typedef struct {
    unsigned char *out;
    ZSTD_CCtx *cctx;
    ZSTD_DCtx *dctx;
    int len;
    int ready;
} CpuZipSlot;

static CpuZipSlot *g_slots;
static int g_initialized;
static int g_slot_count;
static int g_src_cap;
static int g_dst_cap;
static int g_buf_cap;
static CpuCodec g_codec;
static int g_level;

int cpu_zip_init(void) {
    if (g_initialized)
        return 0;

    g_slot_count = envs.IAXL_CPU_ZIP_THREADS;
    if (g_slot_count < 0)
        return -1;
    g_src_cap = envs.IAXL_ZIP_SRC_CAP;
    g_dst_cap = envs.IAXL_ZIP_DST_CAP;

    const char *codec = env_str("IAXL_CPU_ZIP_CODEC", "deflate");
    if (!strcasecmp(codec, "lz4"))
        g_codec = CODEC_LZ4;
    else if (!strcasecmp(codec, "zstd"))
        g_codec = CODEC_ZSTD;
    else if (!strcasecmp(codec, "deflate"))
        g_codec = CODEC_DEFLATE;
    else {
        printf("[cpu_zip] unknown IAXL_CPU_ZIP_CODEC=%s\n", codec);
        return -1;
    }
    // deflate: zlib level; zstd: compression level; lz4: acceleration factor (higher = faster).
    g_level = env_int("IAXL_CPU_ZIP_LEVEL", g_codec == CODEC_DEFLATE ? Z_DEFAULT_COMPRESSION : 1);

    int bound = g_codec == CODEC_LZ4    ? LZ4_compressBound(g_src_cap)
                : g_codec == CODEC_ZSTD ? (int)ZSTD_compressBound((size_t)g_src_cap)
                                        : (int)compressBound((uLong)g_src_cap);
    g_buf_cap = g_src_cap > g_dst_cap ? g_src_cap : g_dst_cap;
    if (bound > g_buf_cap)
        g_buf_cap = bound;

    if (g_slot_count == 0) {
        g_initialized = 1;
        printf("[cpu_zip] disabled: threads=0\n");
        return 0;
    }

    g_slots = calloc((size_t)g_slot_count, sizeof(*g_slots));
    if (!g_slots)
        return -1;

    for (int slot = 0; slot < g_slot_count; slot++) {
        g_slots[slot].out = malloc((size_t)g_buf_cap);
        if (!g_slots[slot].out) {
            cpu_zip_shutdown();
            return -1;
        }
        if (g_codec == CODEC_ZSTD) {
            g_slots[slot].cctx = ZSTD_createCCtx();
            g_slots[slot].dctx = ZSTD_createDCtx();
            if (!g_slots[slot].cctx || !g_slots[slot].dctx) {
                cpu_zip_shutdown();
                return -1;
            }
        }
    }

    g_initialized = 1;
    printf("[cpu_zip] config: codec=%s level=%d threads=%d slots=%d src_cap=%d B buf_cap=%d B "
           "depth=1\n",
           codec, g_level, g_slot_count, g_slot_count, g_src_cap, g_buf_cap);
    return 0;
}

int cpu_zip_num_slots(void) { return g_slot_count; }
int cpu_zip_queue_depth(void) { return 1; }
int cpu_zip_src_cap(void) { return g_src_cap; }

// Intel QPL only decodes a 4 KB history window, so shrink ours whenever an IAA worker may be the
// one to decompress this chunk. kv_zip tags each block with the answer and routes accordingly.
static int compress_window_bits(void) { return envs.IAXL_IAA_ZIP_ENABLE ? -12 : -MAX_WBITS; }

int cpu_zip_iaa_decodable(void) {
    return g_codec == CODEC_DEFLATE && compress_window_bits() >= -12;
}

int cpu_zip_compress(int slot, void *src, int len) {
    if (!g_slots || slot < 0 || slot >= g_slot_count || !src || len <= 0 || len > g_src_cap)
        return -1;

    CpuZipSlot *state = &g_slots[slot];
    state->ready = 0;
    int out_len;

    if (g_codec == CODEC_LZ4) {
        out_len = LZ4_compress_fast(src, (char *)state->out, len, g_buf_cap, g_level);
        if (out_len <= 0)
            return -1;
    } else if (g_codec == CODEC_ZSTD) {
        size_t rc = ZSTD_compressCCtx(state->cctx, state->out, (size_t)g_buf_cap, src, (size_t)len,
                                      g_level);
        if (ZSTD_isError(rc))
            return -1;
        out_len = (int)rc;
    } else {
        z_stream stream = {0};
        if (deflateInit2(&stream, g_level, Z_DEFLATED, compress_window_bits(), 8,
                         Z_DEFAULT_STRATEGY) != Z_OK)
            return -1;
        stream.next_in = src;
        stream.avail_in = (uInt)len;
        stream.next_out = state->out;
        stream.avail_out = (uInt)g_buf_cap;
        int status = deflate(&stream, Z_FINISH);
        out_len = (int)stream.total_out;
        deflateEnd(&stream);
        if (status != Z_STREAM_END)
            return -1;
    }

    state->len = out_len;
    state->ready = 1;
    return 0;
}

int cpu_zip_decompress(int slot, void *src, int len) {
    if (!g_slots || slot < 0 || slot >= g_slot_count || !src || len <= 0 || len > g_buf_cap)
        return -1;

    CpuZipSlot *state = &g_slots[slot];
    state->ready = 0;
    int out_len;

    if (g_codec == CODEC_LZ4) {
        out_len = LZ4_decompress_safe(src, (char *)state->out, len, g_src_cap);
        if (out_len < 0)
            return -1;
    } else if (g_codec == CODEC_ZSTD) {
        size_t rc = ZSTD_decompressDCtx(state->dctx, state->out, (size_t)g_src_cap, src,
                                        (size_t)len);
        if (ZSTD_isError(rc))
            return -1;
        out_len = (int)rc;
    } else {
        z_stream stream = {0};
        if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
            return -1;
        stream.next_in = src;
        stream.avail_in = (uInt)len;
        stream.next_out = state->out;
        stream.avail_out = (uInt)g_src_cap;
        int status = inflate(&stream, Z_FINISH);
        out_len = (int)stream.total_out;
        inflateEnd(&stream);
        if (status != Z_STREAM_END)
            return -1;
    }

    state->len = out_len;
    state->ready = 1;
    return 0;
}

int cpu_zip_wait(int slot, void **dest, int *len) {
    if (!g_slots || slot < 0 || slot >= g_slot_count || !g_slots[slot].ready)
        return -1;

    if (dest)
        *dest = g_slots[slot].out;
    if (len)
        *len = g_slots[slot].len;
    return 0;
}

void cpu_zip_shutdown(void) {
    if (g_slots) {
        for (int slot = 0; slot < g_slot_count; slot++) {
            free(g_slots[slot].out);
            ZSTD_freeCCtx(g_slots[slot].cctx);
            ZSTD_freeDCtx(g_slots[slot].dctx);
        }
        free(g_slots);
    }
    g_slots = NULL;
    g_initialized = 0;
    g_slot_count = 0;
    g_src_cap = 0;
    g_dst_cap = 0;
    g_buf_cap = 0;
}