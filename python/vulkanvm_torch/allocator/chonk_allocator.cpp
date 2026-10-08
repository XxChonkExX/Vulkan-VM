// chonk_allocator.cpp - Production provider: pool + DMA-BUF + HIP import
// wired into the pure slab core (chonk_slab.hpp). Retention policy env knobs
// and the PyTorch C ABI live here.

#include "chonk_allocator.hpp"

#include "../device/pool_device.hpp"
#include "../interop/hip_external_memory.hpp"
#include "chonk_slab.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace vvm_torch {
using slab::Block;  // provider callbacks use the slab block type unqualified
namespace {

size_t envSizeGB(const char* name, size_t defGB) {
    const char* p = getenv(name);
    if (p) {
        double v = atof(p);
        if (v >= 1.0) return (size_t)v;
    }
    return defGB;
}

// Bucket ladder from 1 MB to 512 GB. New blocks are rounded UP to the
// smallest bucket >= the request. One unified freelist across all block
// sizes (slab::Core best-fit) so 1 KB to 131K-context allocations share
// the same pool, with no tier-routing layer in between. Defaults to
// "auto" (the full ladder). Override with CHONK_POOL_BLOCK_SIZES_GB.
//
// Ladder layout (per user spec, no shortcuts):
//   MB scale:
//     1, 1.5, 2, 3, 4 MB                          (fine for activations, gradients)
//     6, 8, 10, 12, ..., 16382 MB (2 MB steps)   (continuous to 16 GB)
//   GB scale:
//     16, 18, 20, ..., 128 GB (2 GB steps)        (KV cache, p/scores)
//   Standard sizing:
//     128, 192, 256, 384, 512 GB                 (1.5x; huge contexts)
//
// The 1.5 MB rung catches the "odd duck" between 1 and 2 MB. From 4 MB
// upward, 2-unit MB steps give <= 2 MB of waste. From 16 GB upward,
// 2-unit GB steps give <= 2 GB of waste. Beyond 128 GB, 1.5x standard
// sizing (matches the GTT heap's power-of-2 fragment).
std::vector<size_t> buckets() {
    static std::vector<size_t> b = [] {
        std::vector<size_t> out;
        const char* p = getenv("CHONK_POOL_BLOCK_SIZES_GB");
        if (p && std::string(p) != "auto") {
            // Custom ladder, in GB, parsed like "1,2,4,8,16".
            std::string str(p);
            size_t start = 0;
            for (;;) {
                size_t end = str.find(',', start);
                std::string token = str.substr(start, end == std::string::npos ? std::string::npos : end - start);
                double gb_val = atof(token.c_str());
                if (gb_val >= 1.0) out.push_back((size_t)(gb_val * 1024.0 * 1024.0 * 1024.0));
                if (end == std::string::npos) break;
                start = end + 1;
            }
        } else {
            // Default: the full user-specified ladder.
            //   1, 1.5, 2, 3, 4 MB  (fine MB rungs)
            out.push_back(1 * 1024 * 1024);
            out.push_back(1 * 1024 * 1024 + 512 * 1024);  // 1.5 MB
            out.push_back(2 * 1024 * 1024);
            out.push_back(3 * 1024 * 1024);
            out.push_back(4 * 1024 * 1024);
            //   6, 8, 10, ..., 16382 MB  (2 MB steps, continuous to 16 GB)
            for (size_t mb = 6; mb <= 16382; mb += 2) {
                out.push_back(mb * 1024 * 1024);
            }
            //   16, 18, 20, ..., 128 GB  (2 GB steps)
            for (size_t gb = 16; gb <= 128; gb += 2) {
                out.push_back((size_t)gb * 1024 * 1024 * 1024);
            }
            //   192, 256, 384, 512 GB  (1.5x standard, beyond 128)
            out.push_back((size_t)192 * 1024 * 1024 * 1024);
            out.push_back((size_t)256 * 1024 * 1024 * 1024);
            out.push_back((size_t)384 * 1024 * 1024 * 1024);
            out.push_back((size_t)512 * 1024 * 1024 * 1024);
            fprintf(stderr,
                    "[allocator] auto-buckets: user-spec ladder, %zu rungs, 1 MB .. 512 GB "
                    "(1,1.5,2,3,4,6,8..16GB step 2MB, 16..128GB step 2GB, 192..512GB 1.5x)\n",
                    out.size());
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }();
    return b;
}

// Read CHONK_MIN_BLOCK_GB and (override) CHONK_MIN_BLOCK_MB. Default 1 MB.
static size_t computeMinBlockBytes() {
    if (const char* mb = getenv("CHONK_MIN_BLOCK_MB")) {
        double v = atof(mb);
        if (v >= 0.5) return (size_t)(v * 1024.0 * 1024.0);
    }
    return envSizeGB("CHONK_MIN_BLOCK_GB", 0.001) * 1024ull * 1024ull * 1024ull;  // 1 MB
}

// Round a request size UP to the smallest ladder rung that fits it. Falls
// back to max(minBlock, need) rounded up to the next power of two if the
// request exceeds all rungs (defensive; should not happen for our workloads).
size_t roundToBucket(size_t need) {
    for (size_t b : buckets()) {
        if (b >= need) return b;
    }
    size_t minBlock = computeMinBlockBytes();
    size_t b = std::max(minBlock, need);
    size_t p = 1;
    while (p < b) p <<= 1;
    return p;
}

FILE* g_allocLog = nullptr;

void allocLog(const char* op, void* ptr, size_t sz) {
    if (!g_allocLog) {
        const char* p = getenv("CHONK_ALLOC_LOG");
        if (p) g_allocLog = fopen(p, "w");
    }
    if (g_allocLog) {
        fprintf(g_allocLog, "%s %p %zu\n", op, ptr, sz);
        fflush(g_allocLog);
    }
}

// Production provider: Vulkan allocation -> DMA-BUF export -> HIP import.
// Keeps the vvm::Allocation handle per block (the slab core type-erases it).
struct PoolBlockProvider : slab::Core::IProvider {
    size_t minBlockBytes = computeMinBlockBytes();  // default 1 MB
    size_t escalateSlackGB = envSizeGB("CHONK_ESCALATE_SLACK_GB", 2);
    size_t minBlocksOnOOM = 2;  // pressure relief keeps 2 warm blocks (validated design)
    std::unordered_map<void*, vvm::Allocation> blockAllocs_;

    size_t warmBlocks() const { return envSizeGB("CHONK_WARM_BLOCKS", 8); }
    size_t maxBlocks() const { return envSizeGB("CHONK_MAX_BLOCKS", 24); }
    size_t freeListMax() const {
        // Default 32: measured peak concurrency at raw-32k was 18 blocks.
        // A reserve smaller than peak concurrency cannot absorb a working-set
        // spike, which is exactly when it is needed. Overridable.
        return envSizeGB("CHONK_FREE_LIST_MAX", 32);
    }

    // Block recycling (2026-10-07, 8th). The slab retires blocks constantly --
    // 24 x 4GB per run at span 2048 -- and every retirement is a
    // vkFreeMemory + dma-buf close + hipDestroyExternalMemory round trip.
    // We proved the pool side is clean (0 deferrals, 0 stranded blocks, pool
    // flat at ~38-42GB) yet amdgpu GTT never came back down: 98.5GB GTT
    // against a 38.0GB pool. That retention is driver-side and happens on the
    // destroy path, so the fix is to NOT destroy. Retired blocks keep their
    // VkDeviceMemory AND their live HIP import, so reuse needs no driver call
    // at all: hand the same Block* back with its free list reset. Churn drops
    // to zero after warmup and GTT plateaus at the concurrent-block high
    // water mark instead of climbing per span.
    //
    // Bounded by CHONK_FREE_LIST_MAX blocks so we hold a fixed warm reserve
    // rather than unbounded memory; beyond that, blocks take the real teardown
    // path so the pool can still shrink.
    std::vector<Block*> freeList_;

    // Best-fit reuse: smallest recycled block that still satisfies `need`.
    // The slab's own best-fit then treats it as any other block.
    Block* takeRecycled(size_t need) {
        Block* best = nullptr;
        size_t bestSize = SIZE_MAX;
        for (auto it = freeList_.begin(); it != freeList_.end(); ++it) {
            if ((*it)->size >= need && (*it)->size < bestSize) {
                best = *it;
                bestSize = (*it)->size;
            }
        }
        if (!best) return nullptr;
        freeList_.erase(std::remove(freeList_.begin(), freeList_.end(), best),
                        freeList_.end());
        best->liveBytes = 0;
        best->freeChunks.clear();
        best->freeChunks.push_back({0, best->size});
        return best;
    }

    // Drop recycled blocks at teardown: the slab never sees them, so nothing
    // else will ever release their VkDeviceMemory or HIP import.
    void releaseFreeList() {
        for (Block* b : freeList_) {
            // Same order as the real teardown: HIP import first, then the fd,
            // then the Vulkan allocation. After pool.shutdown() the HIP
            // context is gone and this is skipped deliberately -- the OS
            // reclaims the handle.
            if (b->extHandle && pool()) {
                hipDestroyExternalMemory(static_cast<hipExternalMemory_t>(b->extHandle));
            }
            if (b->fd >= 0) close(b->fd);
            auto it = blockAllocs_.find(b->base);
            if (it != blockAllocs_.end() && pool()) {
                pool()->deallocate(std::move(it->second));
                blockAllocs_.erase(it);
            }
            delete b;
        }
        freeList_.clear();
    }

    Block* createBlock(slab::Core& core, size_t need) override {
        vvm::UnifiedMemoryPool* p = pool();
        if (!p) return nullptr;
        // Recycling first: a warm block that already holds a live HIP import
        // costs no driver work at all, so it is always cheaper than the
        // vkAllocateMemory -> export -> import path below. This is what keeps
        // block churn (and therefore driver GTT retention) at zero after
        // warmup.
        if (Block* r = takeRecycled(need)) {
            allocLog("U", r->base, r->size);
            return r;
        }
        // Size the new block to the nearest ladder rung >= max(need, minBlock).
        // For a 1 KB request with min=1 MB, we get a 1 MB block; for a 700 MB
        // request, a 1 GB block; for 14 GB, a 16 GB block. The slab's best-fit
        // then finds the smallest free chunk across all blocks (any size),
        // so the ladder is just the set of block sizes the pool may grow to.
        size_t blockSize = roundToBucket(
            std::max(minBlockBytes,
                     (need + slab::Core::kAlign - 1) / slab::Core::kAlign * slab::Core::kAlign));
        vvm::AllocDesc desc;
        desc.size = blockSize;
        desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
        desc.mapped = true;
        desc.exportable = true;
        desc.name = "torch_segment";
        auto allocOpt = p->allocate(desc);
        if (!allocOpt) {
            // Driver refused a new block. BEFORE escalating (which asks the
            // driver for MORE memory and fails harder), serve the request from
            // the recycle reserve: those blocks hold live VkDeviceMemory and a
            // live HIP import and need no driver call at all. This is the fix
            // for the 2026-10-08 crash -- vkAllocateMemory returned
            // VK_ERROR_OUT_OF_DEVICE_MEMORY on a 4GB request at 102.8/121GB
            // GTT (nominal headroom, but no contiguous region), and the old
            // path went releaseEmptyBlocks -> escalate -> nullptr without ever
            // consulting freeList_, so a satisfied-by-reserve request died.
            if (Block* r = takeRecycled(need)) {
                fprintf(stderr,
                        "[allocator] driver OOM: served %zu bytes from recycle "
                        "reserve (%zu block(s) held)\n",
                        need, freeList_.size() + 1);
                allocLog("U", r->base, r->size);
                return r;
            }
            // Pressure relief: release fully-free blocks down to a warm
            // floor, then retry once before escalating.
            size_t released = core.releaseEmptyBlocks(minBlocksOnOOM);
            if (released > 0) {
                fprintf(stderr, "[allocator] released %zu empty block(s) under pressure; retrying %zu bytes\n",
                        released, need);
                allocOpt = p->allocate(desc);
            }
            if (!allocOpt) {
                // Escalate to the nearest configured bucket, bounded by
                // need + slack (escalating far past a driver OOM just
                // thrashes the allocator). Re-check the reserve first: if the
                // driver cannot give us the rung we want, a recycled block
                // still beats a failed vkAllocateMemory.
                size_t bucketSize = roundToBucket(need);
                size_t slack = escalateSlackGB * 1024ull * 1024ull * 1024ull;
                if (bucketSize > blockSize && bucketSize <= need + slack) {
                    if (Block* r = takeRecycled(bucketSize)) {
                        fprintf(stderr,
                                "[allocator] driver OOM (escalation): served "
                                "%zu bytes from recycle reserve\n", bucketSize);
                        allocLog("U", r->base, r->size);
                        return r;
                    }
                    fprintf(stderr, "[allocator] pressure: escalating %zu -> %zu bucket\n",
                            blockSize, bucketSize);
                    desc.size = bucketSize;
                    allocOpt = p->allocate(desc);
                }
            }
        }
        if (!allocOpt) return nullptr;
        vvm::Allocation a = std::move(*allocOpt);
        // The pool's granted size is authoritative: the escalation path
        // re-requests at bucketSize, so the Vulkan allocation may exceed the
        // original request. The HIP import and slab bookkeeping MUST describe
        // the same memory object as the Vulkan allocation.
        const size_t actualBlockSize = a.size;
        auto info = p->exportMemory(a, vvm::ExternalHandleType::OpaqueFd);
        if (!info) {
            p->deallocate(std::move(a));
            return nullptr;
        }
        int fd = info->handle.release();
        hipExternalMemory_t ext = nullptr;
        void* base = hipImportFromFd(fd, actualBlockSize, &ext);
        if (!base) {
            p->deallocate(std::move(a));
            return nullptr;
        }
        Block* b = new Block();
        b->base = base;
        b->extHandle = ext;
        b->fd = fd;
        b->size = actualBlockSize;
        b->freeChunks.push_back({0, actualBlockSize});
        blockAllocs_[base] = std::move(a);  // ownership for destroyBlock
        allocLog("B", base, actualBlockSize);
        return b;
    }

    // Blocks whose HIP teardown failed (GPU still referencing the import).
    // Retried on later releases; without this the HIP-side GTT mapping leaks
    // while the pool side already looks freed (GTT kept climbing with flat
    // pool: 119GB GTT at 54GB pool).
    struct PendingDestroy {
        hipExternalMemory_t ext = nullptr;
        int fd = -1;
        vvm::Allocation alloc;
        size_t size = 0;
    };
    std::vector<PendingDestroy> pending_;

    // Drain fully-free transient state first: retry previously-failed HIP
    // destroys. Returns number actually reclaimed.
    size_t drainPending() {
        size_t done = 0;
        if (!pool()) return 0;
        for (auto it = pending_.begin(); it != pending_.end();) {
            // Drain in-flight work before tearing down the import.
            // hipDestroyExternalMemory fails while any kernel still references
            // the memory; without this sync every retry sees the same busy
            // import and the block is stranded in pending_ forever, leaking
            // both its VkDeviceMemory (still in dedicatedAllocations_, so
            // totalUsed never falls) and its HIP-side GTT mapping. This was the
            // 80-allocations-vs-7-live-blocks gap: 73 deferred, 0 reclaimed.
            // Blocks are destroyed rarely (not per-alloc), so a full device
            // sync here is not a hot-path cost.
            hipDeviceSynchronize();
            hipError_t rc = hipDestroyExternalMemory(it->ext);
            if (rc != hipSuccess) {
                ++it;
                continue;
            }
            if (it->fd >= 0) close(it->fd);
            pool()->deallocate(std::move(it->alloc));
            allocLog("R", nullptr, it->size);
            it = pending_.erase(it);
            ++done;
        }
        if (done > 0) {
            fprintf(stderr, "[allocator] reclaimed %zu deferred block(s)\n", done);
        }
        return done;
    }

    void destroyBlock(Block* b) override {
        // After pool.shutdown() the HIP context is gone; skip HIP destruction
        // during teardown (leak the handles; the OS reclaims them).
        drainPending();
        const size_t blockSize = b->size;
        void* const base = b->base;
        // Recycle instead of destroy while we have room in the warm reserve.
        // Deliberately placed BEFORE the HIP teardown: the whole point is to
        // keep the VkDeviceMemory and the live HIP import so reuse needs no
        // driver round trip (see freeList_). Only past the cap do we take the
        // real teardown path, so the pool can still shrink.
        if (freeList_.size() < freeListMax()) {
            b->liveBytes = 0;
            b->freeChunks.clear();
            freeList_.push_back(b);
            allocLog("C", base, blockSize);
            return;
        }
        if (b->extHandle && pool()) {
            // Clear in-flight work before releasing the import, so this
            // succeeds and the block never enters pending_ (see drainPending).
            hipDeviceSynchronize();
            hipError_t rc = hipDestroyExternalMemory(
                static_cast<hipExternalMemory_t>(b->extHandle));
            if (rc != hipSuccess) {
                // GPU still holds the import (in-flight work). Do NOT close
                // the fd or free the Vulkan side yet; stash everything and
                // retry on a later release. Dropping them here leaks the
                // HIP-side GTT mapping permanently while the pool (and slab)
                // already account the block as gone.
                fprintf(stderr,
                        "[allocator] hipDestroyExternalMemory failed (%d) for "
                        "%zu bytes; deferring block teardown\n",
                        (int)rc, blockSize);
                auto it = blockAllocs_.find(base);
                PendingDestroy pd;
                pd.ext = static_cast<hipExternalMemory_t>(b->extHandle);
                pd.fd = b->fd;
                pd.size = blockSize;
                if (it != blockAllocs_.end()) {
                    pd.alloc = std::move(it->second);
                    blockAllocs_.erase(it);
                }
                b->extHandle = nullptr;
                b->fd = -1;
                pending_.push_back(std::move(pd));
                delete b;
                return;
            }
        }
        if (b->fd >= 0) close(b->fd);
        if (pool()) {
            auto it = blockAllocs_.find(base);
            if (it != blockAllocs_.end()) {
                pool()->deallocate(std::move(it->second));
                blockAllocs_.erase(it);
            }
        }
        allocLog("R", base, blockSize);
        delete b;
    }
};

PoolBlockProvider& provider() {
    static PoolBlockProvider p;
    return p;
}


slab::Core& core() {
    static slab::Core c(&provider(), provider().warmBlocks(),
                        provider().maxBlocks(), provider().minBlocksOnOOM);
    return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI (declared in chonk_allocator.hpp)
// ---------------------------------------------------------------------------

void* chonk_allocator_alloc(ssize_t size, int device, void* stream) {
    (void)device; (void)stream;
    if (size < 0) return nullptr;  // ABI boundary: reject negative sizes explicitly
    if (!pool()) return nullptr;   // pool must be initialized before install
    size_t granted = 0;
    // NO per-alloc empty-block release here. Every slab block is a dedicated
    // exportable allocation (vkAllocateMemory + dma-buf + HIP import), so
    // releasing and re-creating blocks on the hot path round-trips the kernel
    // and driver per chunk — the block churn that destabilized long runs.
    // Warm blocks are retained; pressure relief happens only inside
    // createBlock's retry path (releaseEmptyBlocks(minBlocksOnOOM)).
    void* ptr = core().alloc((size_t)size, &granted);
    if (ptr) allocLog("A", ptr, granted);
    return ptr;
}

void chonk_allocator_free(void* ptr, size_t size, void* stream) {
    (void)stream;
    if (!ptr) return;
    allocLog("F", ptr, size);
    core().free(ptr, size);
}

void vvm_torch_chonk_allocator_reset() {
    core().reset();
    // The slab only knows about blocks it still owns; recycled blocks live
    // only in the provider's free list. Without this they would keep their
    // VkDeviceMemory and HIP import alive for the whole process lifetime.
    provider().releaseFreeList();
}

size_t vvm_torch_chonk_allocator_release_empty(size_t keepFloor) {
    size_t n = core().releaseEmptyBlocks(keepFloor);
    // Retry any deferred HIP teardowns while we're already on the slow path.
    n += provider().drainPending();
    return n;
}

const char* vvm_torch_chonk_allocator_slab_stats() {
    static thread_local std::string buf;
    auto s = core().stats();
    // Per-block liveBytes is I5-relevant; emit compact summary.
    buf.clear();
    char tmp[128];
    snprintf(tmp, sizeof(tmp),
             "blocks=%zu live=%zu free=%zu cap=%zu",
             s.blocks, s.liveBytes, s.freeBytes, s.capacityBytes);
    buf = tmp;
    return buf.c_str();
}

void vvm_torch_chonk_allocator_live_histogram(std::string& out) {
    std::vector<std::pair<size_t, size_t>> hist;
    core().liveSizeHistogram(hist);
    out.clear();
    char tmp[64];
    for (const auto& kv : hist) {
        snprintf(tmp, sizeof(tmp), "%zu:%zu ", kv.first, kv.second);
        out += tmp;
    }
}

}  // namespace vvm_torch
