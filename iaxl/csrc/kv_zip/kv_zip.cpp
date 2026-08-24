// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// QAT, IAA and CPU workers share one task pool. Each worker claims another item when a request
// completes, so the faster backend naturally processes more of the batch. QAT and IAA workers keep
// multiple asynchronous requests in flight, while CPU workers run one synchronous raw-DEFLATE
// request each.
//
// All backends emit raw DEFLATE, but compatibility is normally one-way: IAA decodes at most a 4 KB
// history window while QAT gen4 always compresses with 32 KB and silently ignores
// CpaDcSessionSetupData.windowSize. So QAT and CPU can decompress anything, whereas IAA can only
// decompress what IAA and a 4 KB-window CPU produced.
//
// IAXL_ZIP_JOB_SIZE removes that asymmetry. A stateless DEFLATE job can never emit a back-reference
// longer than its own input, so capping every job at 4 KB makes even QAT output IAA-decodable, and
// IAA can then join the decompression pass. Chunks larger than the job size are compressed as
// several independent jobs and stored as a segment chain, which costs some compression ratio.

#include <torch/extension.h>

#include <omp.h>
#include <algorithm>
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

// Largest history window Intel QPL can decode, and therefore the largest job whose output is
// guaranteed to be IAA-decodable no matter which backend produced it.
static constexpr int IAA_HISTORY_LIMIT = 4096;

// A chunk header is [int payload_len][int orig_size]. A negative payload_len marks a segmented
// payload: |payload_len| bytes holding a chain of [int job_payload_len][int job_orig_len][data].
static constexpr size_t CHUNK_HEADER = sizeof(int) * 2;
static constexpr size_t SEGMENT_HEADER = sizeof(int) * 2;

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

template <class Submit, class Complete>
static void zip_pipeline(size_t n, bool allow_iaa, Submit &&submit, Complete &&complete) {
    ensure_zip_init();
    const int qat_depth = envs.IAXL_QAT_ZIP_ENABLE ? qat_zip_queue_depth() : 1;
    const int qat_available = envs.IAXL_QAT_ZIP_ENABLE ? qat_zip_num_slots() / qat_depth : 0;
    const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
    const int iaa_depth = envs.IAXL_IAA_ZIP_ENABLE ? iaa_zip_queue_depth() : 1;
    const int iaa_available = envs.IAXL_IAA_ZIP_ENABLE ? iaa_zip_num_slots() / iaa_depth : 0;
    const int iaa_configured = envs.IAXL_IAA_ZIP_ENABLE ? envs.IAXL_IAA_INSTANCE_NUM : 0;
    const int iaa_workers = allow_iaa ? iaa_configured : 0;
    const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
    const int worker_count = qat_workers + iaa_workers + cpu_workers;
    IAXL_CHECK(qat_workers <= qat_available,
               "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
    IAXL_CHECK(iaa_configured <= iaa_available,
               "kv_zip: IAXL_IAA_INSTANCE_NUM exceeds available IAA instances");
    IAXL_CHECK(qat_workers + iaa_configured + cpu_workers == envs.IAXL_OMP_THREAD_NUM,
               "kv_zip: compression workers do not match OMP_NUM_THREADS");
    IAXL_CHECK(worker_count > 0, "kv_zip: no worker can decompress these streams");

    std::atomic<size_t> next{0};
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
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
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

            const size_t i = draining ? n : next.fetch_add(1, std::memory_order_relaxed);
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

    // Truncation and shuffling rewrite the tensor in place, so they run once per tensor rather than
    // once per compression job.
#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t i = 0; i < n; i++) {
        const auto &t = tensors[i];
        IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
                   "kv_zip: tensor must be a contiguous CPU tensor");
        const size_t nb = t.numel() * t.element_size();
        IAXL_CHECK(nb <= static_cast<size_t>(INT_MAX),
                   "kv_zip: tensor byte size exceeds zip integer length range");
        char *p = static_cast<char *>(t.data_ptr());
        lossy_trunc(p, nb, t.element_size());
        data_shuffle(p, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
        orig_sizes[i] = nb;
    }

    const bool segmented = envs.IAXL_ZIP_JOB_SIZE > 0;
    const size_t job = segmented ? static_cast<size_t>(envs.IAXL_ZIP_JOB_SIZE) : 0;

    struct Job {
        char *src;
        int src_len;
    };
    std::vector<Job> jobs;
    std::vector<size_t> job_begin(n + 1);
    for (size_t i = 0; i < n; i++) {
        job_begin[i] = jobs.size();
        char *p = static_cast<char *>(tensors[i].data_ptr());
        const size_t nb = orig_sizes[i];
        const size_t step = segmented ? job : nb;
        for (size_t off = 0; off < nb; off += step)
            jobs.push_back({p + off, static_cast<int>(std::min(step, nb - off))});
    }
    job_begin[n] = jobs.size();

    std::vector<char *> job_buf(jobs.size(), nullptr);
    std::vector<int> job_len(jobs.size(), 0);

    zip_pipeline(
        jobs.size(), /*allow_iaa=*/true,
        [&](ZipBackend backend, int slot, size_t k) {
            IAXL_CHECK(jobs[k].src_len <= zip_src_cap(backend),
                       "kv_zip: job byte size exceeds zip source capacity");
            const int status = zip_compress(backend, slot, jobs[k].src, jobs[k].src_len);
            IAXL_CHECK(status == 0, "kv_zip: zip compress failed");
        },
        [&](ZipBackend, size_t k, void *out, int out_len) {
            char *b = static_cast<char *>(malloc(static_cast<size_t>(out_len)));
            IAXL_CHECK(b != nullptr, "kv_zip: job buffer allocation failed");
            memcpy(b, out, static_cast<size_t>(out_len));
            job_buf[k] = b;
            job_len[k] = out_len;
        });

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t i = 0; i < n; i++) {
        const size_t first = job_begin[i], last = job_begin[i + 1];
        size_t payload = 0;
        for (size_t k = first; k < last; k++)
            payload += (segmented ? SEGMENT_HEADER : 0) + static_cast<size_t>(job_len[k]);
        IAXL_CHECK(payload <= static_cast<size_t>(INT_MAX),
                   "kv_zip: compressed payload exceeds zip integer length range");

        char *buf = static_cast<char *>(malloc(CHUNK_HEADER + payload));
        IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
        reinterpret_cast<int *>(buf)[0] =
            segmented ? -static_cast<int>(payload) : static_cast<int>(payload);
        reinterpret_cast<int *>(buf)[1] = static_cast<int>(orig_sizes[i]);

        char *w = buf + CHUNK_HEADER;
        for (size_t k = first; k < last; k++) {
            if (segmented) {
                reinterpret_cast<int *>(w)[0] = job_len[k];
                reinterpret_cast<int *>(w)[1] = jobs[k].src_len;
                w += SEGMENT_HEADER;
            }
            memcpy(w, job_buf[k], static_cast<size_t>(job_len[k]));
            w += job_len[k];
            free(job_buf[k]);
        }
        out_bufs[i] = buf;
        out_sizes[i] = CHUNK_HEADER + payload;
    }
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

    std::vector<size_t> compressed_indices;
    compressed_indices.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] != 0)
            compressed_indices.push_back(i);
    }

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] == 0)
            copy_raw(i);
    }

    if (compressed_indices.empty())
        return;

    struct Job {
        size_t item;
        const char *src;
        int src_len;
        size_t dst_off;
        int dst_len;
    };
    std::vector<Job> jobs;
    // IAA may only decompress if no job can contain a back-reference beyond its 4 KB window.
    bool iaa_decodable = true;

    for (size_t i : compressed_indices) {
        const int encoded_len = reinterpret_cast<const int *>(data_ptrs[i])[0];
        IAXL_CHECK(encoded_len != 0 && encoded_len != INT_MIN,
                   "kv_zip: invalid compressed payload length");
        const auto &t = tensors[i];
        const size_t nb = t.numel() * t.element_size();
        const char *p = data_ptrs[i] + CHUNK_HEADER;

        if (encoded_len > 0) {
            jobs.push_back({i, p, encoded_len, 0, static_cast<int>(nb)});
            if (nb > static_cast<size_t>(IAA_HISTORY_LIMIT))
                iaa_decodable = false;
            continue;
        }

        const char *end = p + static_cast<size_t>(-encoded_len);
        size_t off = 0;
        while (p < end) {
            const int src_len = reinterpret_cast<const int *>(p)[0];
            const int dst_len = reinterpret_cast<const int *>(p)[1];
            IAXL_CHECK(src_len > 0 && dst_len > 0, "kv_zip: invalid compressed segment header");
            p += SEGMENT_HEADER;
            jobs.push_back({i, p, src_len, off, dst_len});
            if (dst_len > IAA_HISTORY_LIMIT)
                iaa_decodable = false;
            p += src_len;
            off += static_cast<size_t>(dst_len);
        }
        IAXL_CHECK(p == end && off == nb, "kv_zip: compressed segment chain is inconsistent");
    }

    zip_pipeline(
        jobs.size(), /*allow_iaa=*/!envs.IAXL_QAT_ZIP_ENABLE || iaa_decodable,
        [&](ZipBackend backend, int slot, size_t k) {
            const int status =
                zip_decompress(backend, slot, const_cast<char *>(jobs[k].src), jobs[k].src_len);
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        [&](ZipBackend, size_t k, void *out, int out_len) {
            const Job &j = jobs[k];
            IAXL_CHECK(out_len == j.dst_len,
                       "kv_zip: decompressed size does not match the recorded job size");
            memcpy(static_cast<char *>(tensors[j.item].data_ptr()) + j.dst_off, out,
                   static_cast<size_t>(out_len));
        });

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t item = 0; item < compressed_indices.size(); item++) {
        const auto &t = tensors[compressed_indices[item]];
        const size_t nb = t.numel() * t.element_size();
        data_shuffle(static_cast<char *>(t.data_ptr()), nb, t.dtype() == torch::kBFloat16,
                     data_shuffle_enabled());
    }
}

} // namespace kv_zip
