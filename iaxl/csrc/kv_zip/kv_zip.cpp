// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// QAT, IAA and CPU workers share one task pool. Each worker claims another item when a request
// completes, so the faster backend naturally processes more of the batch. QAT and IAA workers keep
// multiple asynchronous requests in flight, while CPU workers run one synchronous raw-DEFLATE
// request each.
//
// All backends emit raw DEFLATE, but compatibility is one-way: IAA decodes at most a 4 KB history
// window while QAT gen4 always compresses with 32 KB and silently ignores
// CpaDcSessionSetupData.windowSize. So QAT and CPU can decompress anything, whereas IAA can only
// decompress what IAA and a 4 KB-window CPU produced.
//
// Every compressed block therefore records whether IAA can decode it. The decompression pass
// splits the batch into two pools: IAA workers claim only tagged blocks, while QAT and CPU workers
// drain the untagged ones first and then help with what is left. Capping every job at 4 KB would
// give the same compatibility, but it costs far more decompression throughput than it buys.

#include <torch/extension.h>

#include <omp.h>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "cpu_zip.h"
#include "iaa_zip.h"
#include "qat_zip.h"
#include "data_shuffle.h"
#include "lossy.h"
#include "kv_zip.h"

#define OMP_SCHEDULE dynamic

namespace kv_zip {

enum class ZipBackend { QAT, IAA, CPU };

// A chunk header is [int payload_len][int orig_size]. orig_size == 0 marks a stored block, and a
// negative payload_len marks a compressed block IAA is able to decode.
static constexpr size_t CHUNK_HEADER = sizeof(int) * 2;

static bool iaa_decodable(ZipBackend backend) {
    switch (backend) {
    case ZipBackend::IAA:
        return true;
    case ZipBackend::QAT:
        return false;
    default:
        return cpu_zip_iaa_decodable() != 0;
    }
}

static int zip_src_cap(ZipBackend backend) {
    switch (backend) {
    case ZipBackend::QAT:
        return qat_zip_src_cap();
    case ZipBackend::IAA:
        return iaa_zip_src_cap();
    default:
        return cpu_zip_src_cap();
    }
}

static int zip_compress(ZipBackend backend, int slot, void *src, int len) {
    switch (backend) {
    case ZipBackend::QAT:
        return qat_zip_compress(slot, src, len);
    case ZipBackend::IAA:
        return iaa_zip_compress(slot, src, len);
    default:
        return cpu_zip_compress(slot, src, len);
    }
}

static int zip_decompress(ZipBackend backend, int slot, void *src, int len) {
    switch (backend) {
    case ZipBackend::QAT:
        return qat_zip_decompress(slot, src, len);
    case ZipBackend::IAA:
        return iaa_zip_decompress(slot, src, len);
    default:
        return cpu_zip_decompress(slot, src, len);
    }
}

static int zip_wait(ZipBackend backend, int slot, void **dest, int *len) {
    switch (backend) {
    case ZipBackend::QAT:
        return qat_zip_wait(slot, dest, len);
    case ZipBackend::IAA:
        return iaa_zip_wait(slot, dest, len);
    default:
        return cpu_zip_wait(slot, dest, len);
    }
}

static void ensure_zip_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        IAXL_CHECK(envs.IAXL_QAT_ZIP_ENABLE || envs.IAXL_IAA_ZIP_ENABLE || envs.IAXL_CPU_ZIP_ENABLE,
                   "kv_zip: QAT, IAA and CPU zip backends are all disabled");
        if (envs.IAXL_QAT_ZIP_ENABLE)
            IAXL_CHECK(qat_zip_init() == 0, "kv_zip: qat_zip_init failed");
        if (envs.IAXL_IAA_ZIP_ENABLE)
            IAXL_CHECK(iaa_zip_init() == 0, "kv_zip: iaa_zip_init failed");
        if (envs.IAXL_CPU_ZIP_ENABLE)
            IAXL_CHECK(cpu_zip_init() == 0, "kv_zip: cpu_zip_init failed");
    });
}

// Items [0, iaa_ok) may go to any backend; items [iaa_ok, n) are the ones IAA cannot handle.
// Compression passes iaa_ok == n because every backend can compress every block.
template <class Submit, class Complete>
static void zip_pipeline(size_t n, size_t iaa_ok, Submit &&submit, Complete &&complete) {
    ensure_zip_init();
    const int qat_depth = envs.IAXL_QAT_ZIP_ENABLE ? qat_zip_queue_depth() : 1;
    const int qat_available = envs.IAXL_QAT_ZIP_ENABLE ? qat_zip_num_slots() / qat_depth : 0;
    const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
    const int iaa_depth = envs.IAXL_IAA_ZIP_ENABLE ? iaa_zip_queue_depth() : 1;
    const int iaa_available = envs.IAXL_IAA_ZIP_ENABLE ? iaa_zip_num_slots() / iaa_depth : 0;
    const int iaa_workers = envs.IAXL_IAA_ZIP_ENABLE ? envs.IAXL_IAA_INSTANCE_NUM : 0;
    const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
    const int worker_count = qat_workers + iaa_workers + cpu_workers;
    IAXL_CHECK(qat_workers <= qat_available,
               "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
    IAXL_CHECK(iaa_workers <= iaa_available,
               "kv_zip: IAXL_IAA_INSTANCE_NUM exceeds available IAA instances");
    IAXL_CHECK(worker_count == envs.IAXL_OMP_THREAD_NUM,
               "kv_zip: compression workers do not match OMP_NUM_THREADS");
    IAXL_CHECK(worker_count > 0, "kv_zip: no zip worker is configured");
    IAXL_CHECK(iaa_ok == n || qat_workers + cpu_workers > 0,
               "kv_zip: batch holds blocks that only QAT or CPU can decompress, but both are off");

    std::atomic<size_t> iaa_next{0};
    std::atomic<size_t> general_next{iaa_ok};
    const bool split = iaa_ok < n;
    // Returns n once nothing is left. QAT and CPU clear the blocks IAA cannot touch first so the
    // shared pool stays available to IAA for as long as possible.
    auto claim = [&](ZipBackend backend) -> size_t {
        if (split && backend != ZipBackend::IAA) {
            const size_t i = general_next.fetch_add(1, std::memory_order_relaxed);
            if (i < n)
                return i;
        }
        const size_t i = iaa_next.fetch_add(1, std::memory_order_relaxed);
        return i < iaa_ok ? i : n;
    };
#pragma omp parallel num_threads(worker_count)
    {
        const int t = omp_get_thread_num();
        ZipBackend backend;
        int depth;
        int base;
        if (t < qat_workers) {
            backend = ZipBackend::QAT;
            depth = qat_depth;
            base = t * qat_depth;
        } else if (t < qat_workers + iaa_workers) {
            backend = ZipBackend::IAA;
            depth = iaa_depth;
            base = (t - qat_workers) * iaa_depth;
        } else {
            backend = ZipBackend::CPU;
            depth = 1;
            base = t - qat_workers - iaa_workers;
        }
        IAXL_CHECK(omp_get_num_threads() == worker_count,
                   "kv_zip: OpenMP did not create the configured worker team");

        int active_depth = 0;
        std::vector<size_t> slot_item(static_cast<size_t>(depth));
        for (int k = 0; k < depth; k++) {
            const size_t i = claim(backend);
            if (i >= n)
                break;
            submit(backend, base + k, i);
            slot_item[k] = i;
            active_depth++;
        }

        int in_flight = active_depth;
        bool draining = false;
        for (int s = 0; in_flight > 0; s = (s + 1) % active_depth) {
            void *out;
            int out_len;
            const int status = zip_wait(backend, base + s, &out, &out_len);
            IAXL_CHECK(status == 0, "kv_zip: zip wait failed");
            complete(backend, slot_item[s], out, out_len);

            const size_t i = draining ? n : claim(backend);
            if (i < n) {
                submit(backend, base + s, i);
                slot_item[s] = i;
            } else {
                draining = true;
                in_flight--;
            }
        }
    }
}

void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress) {
    const size_t n = tensors.size();

    if (!compress || !envs.IAXL_KV_COMPRESSION) {
#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
        for (size_t i = 0; i < n; i++) {
            const auto &tensor = tensors[i];
            IAXL_CHECK(tensor.is_contiguous() && tensor.device().type() == c10::DeviceType::CPU,
                       "kv_zip: tensor must be a contiguous CPU tensor");
            const size_t nbytes = tensor.numel() * tensor.element_size();
            char *buffer = static_cast<char *>(malloc(CHUNK_HEADER + nbytes));
            IAXL_CHECK(buffer != nullptr, "kv_zip: raw cache buffer allocation failed");
            reinterpret_cast<int *>(buffer)[0] = 0;
            reinterpret_cast<int *>(buffer)[1] = 0;
            memcpy(buffer + CHUNK_HEADER, tensor.data_ptr(), nbytes);
            out_bufs[i] = buffer;
            out_sizes[i] = CHUNK_HEADER + nbytes;
            orig_sizes[i] = nbytes;
        }
        return;
    }

    auto prep = [&](size_t i, char **data, size_t *nbytes) {
        const auto &t = tensors[i];
        IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
                   "kv_zip: tensor must be a contiguous CPU tensor");
        size_t nb = t.numel() * t.element_size();
        char *p = static_cast<char *>(t.data_ptr());
        lossy_trunc(p, nb, t.element_size());
        data_shuffle(p, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
        orig_sizes[i] = nb;
        *data = p;
        *nbytes = nb;
    };

    auto pack = [&](size_t i, ZipBackend backend, const void *payload, int payload_len) {
        IAXL_CHECK(payload_len > 0, "kv_zip: compressed payload is empty");
        char *buf = static_cast<char *>(malloc(CHUNK_HEADER + payload_len));
        IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
        reinterpret_cast<int *>(buf)[0] = iaa_decodable(backend) ? -payload_len : payload_len;
        reinterpret_cast<int *>(buf)[1] = static_cast<int>(orig_sizes[i]);
        memcpy(buf + CHUNK_HEADER, payload, payload_len);
        out_bufs[i] = buf;
        out_sizes[i] = CHUNK_HEADER + payload_len;
    };

    zip_pipeline(
        n, /*iaa_ok=*/n,
        [&](ZipBackend backend, int slot, size_t i) {
            char *data;
            size_t nb;
            prep(i, &data, &nb);
            IAXL_CHECK(nb <= static_cast<size_t>(INT_MAX),
                       "kv_zip: tensor byte size exceeds zip integer length range");
            const int src_cap = zip_src_cap(backend);
            IAXL_CHECK(nb <= static_cast<size_t>(src_cap),
                       "kv_zip: tensor byte size exceeds zip source capacity");
            const int status = zip_compress(backend, slot, data, static_cast<int>(nb));
            IAXL_CHECK(status == 0, "kv_zip: zip compress failed");
        },
        [&](ZipBackend backend, size_t i, void *out, int out_len) {
            pack(i, backend, out, out_len);
        });
}

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors) {
    const size_t n = tensors.size();
    IAXL_CHECK(data_ptrs.size() == n, "kv_zip: decompression inputs must have matching lengths");

    auto copy_raw = [&](size_t i) {
        const auto &t = tensors[i];
        const size_t nb = t.numel() * t.element_size();
        memcpy(t.data_ptr(), data_ptrs[i] + CHUNK_HEADER, nb);
    };

    // IAA-decodable blocks come first so zip_pipeline can reserve the tail for QAT and CPU.
    std::vector<size_t> compressed_indices, qat_cpu_only;
    compressed_indices.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] == 0)
            continue;
        IAXL_CHECK(header[0] != 0 && header[0] != INT_MIN,
                   "kv_zip: invalid compressed payload length");
        if (header[0] < 0)
            compressed_indices.push_back(i);
        else
            qat_cpu_only.push_back(i);
    }
    const size_t iaa_ok_count = compressed_indices.size();
    compressed_indices.insert(compressed_indices.end(), qat_cpu_only.begin(), qat_cpu_only.end());

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] == 0)
            copy_raw(i);
    }

    if (compressed_indices.empty())
        return;

    auto finish = [&](size_t i, const void *out, int out_len) {
        const auto &t = tensors[i];
        const size_t nb = t.numel() * t.element_size();
        IAXL_CHECK(out_len >= 0 && static_cast<size_t>(out_len) == nb,
                   "kv_zip: decompressed size does not match tensor byte size");
        char *dst = static_cast<char *>(t.data_ptr());
        memcpy(dst, out, nb);
        data_shuffle(dst, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
    };

    zip_pipeline(
        compressed_indices.size(), iaa_ok_count,
        [&](ZipBackend backend, int slot, size_t item) {
            const size_t i = compressed_indices[item];
            const int *hdr = reinterpret_cast<const int *>(data_ptrs[i]);
            const char *payload = data_ptrs[i] + CHUNK_HEADER;
            const int payload_len = hdr[0] < 0 ? -hdr[0] : hdr[0];
            const int status =
                zip_decompress(backend, slot, const_cast<char *>(payload), payload_len);
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        [&](ZipBackend, size_t item, void *out, int out_len) {
            const size_t i = compressed_indices[item];
            finish(i, out, out_len);
        });
}

} // namespace kv_zip
