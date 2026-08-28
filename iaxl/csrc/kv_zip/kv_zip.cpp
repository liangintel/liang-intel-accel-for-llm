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
// Every compressed block therefore records in its header whether IAA can decode it, and TaskPool
// hands out work accordingly: IAA workers only get tagged blocks, while QAT and CPU workers drain
// the untagged ones first and then help with what is left. Capping every job at 4 KB would give
// the same compatibility, but it costs far more decompression throughput than it buys.

#include <torch/extension.h>

#include <omp.h>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
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

// Prefixes every cached block. orig_size == 0 marks a stored block, and a negative payload_len
// marks a compressed block IAA is able to decode.
struct ChunkHeader {
    int payload_len;
    int orig_size;
};
static constexpr size_t CHUNK_HEADER = sizeof(ChunkHeader);

static const ChunkHeader *chunk_header(const char *buf) {
    return reinterpret_cast<const ChunkHeader *>(buf);
}

static int never_iaa_decodable(void) { return 0; }
static int always_iaa_decodable(void) { return 1; }

// Backend entry points, indexed by ZipBackend.
struct ZipOps {
    int (*compress)(int slot, void *src, int len);
    int (*decompress)(int slot, void *src, int len);
    int (*wait)(int slot, void **dest, int *len);
    int (*src_cap)(void);
    int (*num_slots)(void);
    int (*queue_depth)(void);
    int (*iaa_decodable)(void);
};

static const ZipOps kZipOps[] = {
    {qat_zip_compress, qat_zip_decompress, qat_zip_wait, qat_zip_src_cap, qat_zip_num_slots,
     qat_zip_queue_depth, never_iaa_decodable},
    {iaa_zip_compress, iaa_zip_decompress, iaa_zip_wait, iaa_zip_src_cap, iaa_zip_num_slots,
     iaa_zip_queue_depth, always_iaa_decodable},
    {cpu_zip_compress, cpu_zip_decompress, cpu_zip_wait, cpu_zip_src_cap, cpu_zip_num_slots,
     cpu_zip_queue_depth, cpu_zip_iaa_decodable},
};

static const ZipOps &ops(ZipBackend backend) { return kZipOps[static_cast<int>(backend)]; }

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

// All workers pull from one pool, so a faster backend naturally takes a larger share of the batch.
// IAA may only claim items it can decode, so those are kept apart; QAT and CPU drain the rest
// first to keep the IAA-decodable list available for as long as possible.
class TaskPool {
public:
    static constexpr size_t kNone = static_cast<size_t>(-1);

    TaskPool(std::vector<size_t> iaa_ok, std::vector<size_t> iaa_undecodable)
        : iaa_ok_(std::move(iaa_ok)), others_(std::move(iaa_undecodable)) {}

    size_t size() const { return iaa_ok_.size() + others_.size(); }
    bool empty() const { return size() == 0; }
    bool has_iaa_undecodable() const { return !others_.empty(); }

    // Returns kNone once this backend has nothing left to claim.
    size_t get_next(ZipBackend backend) {
        if (backend != ZipBackend::IAA && !others_.empty()) {
            const size_t k = others_next_.fetch_add(1, std::memory_order_relaxed);
            if (k < others_.size())
                return others_[k];
        }
        const size_t k = iaa_next_.fetch_add(1, std::memory_order_relaxed);
        return k < iaa_ok_.size() ? iaa_ok_[k] : kNone;
    }

private:
    const std::vector<size_t> iaa_ok_, others_;
    std::atomic<size_t> iaa_next_{0}, others_next_{0};
};

template <class Submit, class Complete>
static void zip_pipeline(TaskPool &pool, Submit &&submit, Complete &&complete) {
    ensure_zip_init();
    const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
    const int iaa_workers = envs.IAXL_IAA_ZIP_ENABLE ? envs.IAXL_IAA_INSTANCE_NUM : 0;
    const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
    const int worker_count = qat_workers + iaa_workers + cpu_workers;
    IAXL_CHECK(qat_workers == 0 || qat_workers <= qat_zip_num_slots() / qat_zip_queue_depth(),
               "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
    IAXL_CHECK(iaa_workers == 0 || iaa_workers <= iaa_zip_num_slots() / iaa_zip_queue_depth(),
               "kv_zip: IAXL_IAA_INSTANCE_NUM exceeds available IAA instances");
    IAXL_CHECK(worker_count == envs.IAXL_OMP_THREAD_NUM,
               "kv_zip: compression workers do not match OMP_NUM_THREADS");
    IAXL_CHECK(worker_count > 0, "kv_zip: no zip worker is configured");
    IAXL_CHECK(!pool.has_iaa_undecodable() || qat_workers + cpu_workers > 0,
               "kv_zip: batch holds blocks that only QAT or CPU can decompress, but both are off");

#pragma omp parallel num_threads(worker_count)
    {
        const int t = omp_get_thread_num();
        ZipBackend backend = ZipBackend::CPU;
        int first = qat_workers + iaa_workers;
        if (t < qat_workers) {
            backend = ZipBackend::QAT;
            first = 0;
        } else if (t < qat_workers + iaa_workers) {
            backend = ZipBackend::IAA;
            first = qat_workers;
        }
        const int depth = ops(backend).queue_depth();
        const int base = (t - first) * depth;
        IAXL_CHECK(omp_get_num_threads() == worker_count,
                   "kv_zip: OpenMP did not create the configured worker team");

        int active_depth = 0;
        std::vector<size_t> slot_item(static_cast<size_t>(depth));
        for (int k = 0; k < depth; k++) {
            const size_t i = pool.get_next(backend);
            if (i == TaskPool::kNone)
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
            const int status = ops(backend).wait(base + s, &out, &out_len);
            IAXL_CHECK(status == 0, "kv_zip: zip wait failed");
            complete(backend, slot_item[s], out, out_len);

            const size_t i = draining ? TaskPool::kNone : pool.get_next(backend);
            if (i != TaskPool::kNone) {
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
            *reinterpret_cast<ChunkHeader *>(buffer) = {0, 0};
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
        *reinterpret_cast<ChunkHeader *>(buf) = {
            ops(backend).iaa_decodable() ? -payload_len : payload_len,
            static_cast<int>(orig_sizes[i])};
        memcpy(buf + CHUNK_HEADER, payload, payload_len);
        out_bufs[i] = buf;
        out_sizes[i] = CHUNK_HEADER + payload_len;
    };

    std::vector<size_t> items(n);
    std::iota(items.begin(), items.end(), size_t{0});
    TaskPool pool(std::move(items), {});
    zip_pipeline(
        pool,
        [&](ZipBackend backend, int slot, size_t i) {
            char *data;
            size_t nb;
            prep(i, &data, &nb);
            IAXL_CHECK(nb <= static_cast<size_t>(INT_MAX),
                       "kv_zip: tensor byte size exceeds zip integer length range");
            IAXL_CHECK(nb <= static_cast<size_t>(ops(backend).src_cap()),
                       "kv_zip: tensor byte size exceeds zip source capacity");
            const int status = ops(backend).compress(slot, data, static_cast<int>(nb));
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

    std::vector<size_t> iaa_ok, qat_cpu_only;
    iaa_ok.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const ChunkHeader *hdr = chunk_header(data_ptrs[i]);
        if (hdr->orig_size == 0)
            continue;
        IAXL_CHECK(hdr->payload_len != 0 && hdr->payload_len != INT_MIN,
                   "kv_zip: invalid compressed payload length");
        (hdr->payload_len < 0 ? iaa_ok : qat_cpu_only).push_back(i);
    }

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t i = 0; i < n; i++) {
        if (chunk_header(data_ptrs[i])->orig_size == 0)
            copy_raw(i);
    }

    TaskPool pool(std::move(iaa_ok), std::move(qat_cpu_only));
    if (pool.empty())
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
        pool,
        [&](ZipBackend backend, int slot, size_t i) {
            const int encoded = chunk_header(data_ptrs[i])->payload_len;
            char *payload = const_cast<char *>(data_ptrs[i] + CHUNK_HEADER);
            const int status =
                ops(backend).decompress(slot, payload, encoded < 0 ? -encoded : encoded);
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        [&](ZipBackend, size_t i, void *out, int out_len) { finish(i, out, out_len); });
}

} // namespace kv_zip
