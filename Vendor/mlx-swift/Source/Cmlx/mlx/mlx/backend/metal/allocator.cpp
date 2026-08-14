// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/resident.h"
#include "mlx/memory.h"

#include <mach/vm_page_size.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <limits>

namespace mlx::core {

constexpr size_t resource_options =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

namespace allocator {

Allocator& allocator() {
  return metal::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto* buf = static_cast<MTL::Buffer*>(ptr_);
  assert(buf->storageMode() != MTL::StorageModePrivate);
  return buf->contents();
}

bool can_reuse_alien_buffer(void* ptr) {
  if (!ptr) {
    return true;
  }
  auto* buf = static_cast<MTL::Buffer*>(ptr);
  return buf->storageMode() != MTL::StorageModePrivate;
}

} // namespace allocator

namespace metal {

MetalAllocator::MetalAllocator(Device& d)
    : device_(d.mtl_device()),
      residency_set_(d.residency_set()),
      buffer_cache_(
          vm_page_size,
          [](MTL::Buffer* buf) { return buf->length(); },
          [this](MTL::Buffer* buf) {
            if (!buf->heap()) {
              residency_set_.erase(buf);
            }
            auto pool = metal::new_scoped_memory_pool();
            buf->release();
          }) {
  const auto& info = gpu::device_info(0);
  auto memsize = std::get<size_t>(info.at("memory_size"));
  auto max_rec_size =
      std::get<size_t>(info.at("max_recommended_working_set_size"));
  resource_limit_ = std::get<size_t>(info.at("resource_limit"));
  // Optional override (Darkbloom): MLX_RESOURCE_LIMIT lets an operator/test pin
  // the Metal resource-COUNT ceiling below the OS default (~499000). Used to
  // deterministically exercise the count-aware high-water trim, and as a safety
  // valve to force earlier cache reclamation on a box seeing the resource-limit
  // crash. The value may only LOWER the ceiling (it is clamped to the OS limit)
  // — raising it above what the hardware/OS reports would invite the very crash
  // this guards against. Strictly validated: a plain unsigned decimal that
  // consumes the whole string, is non-zero, and does not overflow; anything else
  // (empty, sign, junk, range error) is ignored and the OS limit stands.
  if (const char* rl = std::getenv("MLX_RESOURCE_LIMIT")) {
    while (*rl == ' ' || *rl == '\t') {
      ++rl;
    }
    if (*rl >= '0' && *rl <= '9') {  // unsigned decimal only (reject sign/junk)
      errno = 0;
      char* end = nullptr;
      unsigned long long v = std::strtoull(rl, &end, 10);
      bool consumed_all = end != rl && *end == '\0';
      if (consumed_all && errno != ERANGE && v > 0 &&
          v <= std::numeric_limits<size_t>::max()) {
        resource_limit_ = std::min(static_cast<size_t>(v), resource_limit_);
      }
    }
  }
  block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);
  gc_limit_ = std::min(static_cast<size_t>(0.95 * max_rec_size), block_limit_);
  max_pool_size_ = block_limit_;
  bool is_vm = std::get<std::string>(info.at("device_name")) ==
      "Apple Paravirtual device";
  if (is_vm) {
    return;
  }
  auto pool = metal::new_scoped_memory_pool();
  auto heap_desc = MTL::HeapDescriptor::alloc()->init()->autorelease();
  heap_desc->setResourceOptions(resource_options);
  heap_desc->setSize(heap_size_);
  heap_ = NS::TransferPtr(device_->newHeap(heap_desc));
  residency_set_.insert(heap_.get());
}

MetalAllocator::~MetalAllocator() = default;

size_t MetalAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
};

size_t MetalAllocator::set_memory_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, block_limit_);
  gc_limit_ = std::min(
      block_limit_,
      static_cast<size_t>(0.95 * device_->recommendedMaxWorkingSetSize()));
  return limit;
};

size_t MetalAllocator::get_memory_limit() {
  return block_limit_;
}

size_t MetalAllocator::set_wired_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, wired_limit_);
  residency_set_.resize(wired_limit_);
  return limit;
};

Buffer MetalAllocator::malloc(size_t size) {
  // Metal doesn't like empty buffers
  if (size == 0) {
    return Buffer{nullptr};
  }

  // More helpful message if maximum buffer length is exceeded
  if (size > device_->maxBufferLength()) {
    std::ostringstream msg;
    msg << "[metal::malloc] Attempting to allocate " << size
        << " bytes which is greater than"
        << " the maximum allowed buffer size of " << device_->maxBufferLength()
        << " bytes.";
    throw std::runtime_error(msg.str());
  }

  // Align up memory
  if (size > vm_page_size) {
    size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);
  }

  // Try the cache
  std::unique_lock lk(mutex_);
  MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    size_t mem_required = get_active_memory() + get_cache_memory() + size;

    auto pool = metal::new_scoped_memory_pool();

    // If we have a lot of memory pressure try to reclaim memory from the cache.
    // NOTE: release_cached_buffers takes a BYTES-to-free target; when the buffers
    // are tiny this frees only a few entries even though the COUNT is the binding
    // constraint, so the byte path alone cannot bound num_resources_ (see the
    // count-aware reclaim below).
    if (mem_required >= gc_limit_ || num_resources_ >= resource_limit_) {
      num_resources_ -=
          buffer_cache_.release_cached_buffers(mem_required - gc_limit_);
    }

    // Count-aware reclaim (Darkbloom): the Metal resource COUNT limit
    // (resource_limit_, ~iogpu.rsrc_limit/499000) is independent of byte usage.
    // Under churn with many distinct buffer shapes (varied prompt lengths,
    // growing KV caches, multiple co-resident models) freed buffers are recycled
    // into the size-keyed cache and never reused at that exact size, so the cache
    // ENTRY COUNT creeps toward the limit while byte usage stays modest — the
    // byte-driven trim above never fires (its threshold is ~physical RAM). Once
    // the count crosses a high-water mark, proactively clear the cache (pure
    // reuse pool — clearing only costs re-allocation, never correctness) so the
    // count drops back to the live working set. This makes the count limit
    // unreachable by any request mix / batching method, while the existing byte
    // limits keep total memory below physical RAM.
    if (resource_limit_ > 0 &&
        num_resources_ >= (resource_limit_ * resource_high_water_num_) /
                              resource_high_water_den_) {
      num_resources_ -= buffer_cache_.clear();
    }

    // Allocate new buffer if needed
    if (num_resources_ >= resource_limit_) {
      std::ostringstream msg;
      msg << "[metal::malloc] Resource limit (" << resource_limit_
          << ") exceeded.";
      throw std::runtime_error(msg.str());
    }
    lk.unlock();
    if (size < small_size_ && heap_) {
      buf = heap_->newBuffer(size, resource_options);
    }
    if (!buf) {
      buf = device_->newBuffer(size, resource_options);
    }
    if (!buf) {
      std::ostringstream msg;
      msg << "[malloc] Unable to allocate " << size << " bytes.";
      throw std::runtime_error(msg.str());
    }
    lk.lock();
    num_resources_++;
    if (!buf->heap()) {
      residency_set_.insert(buf);
    }
  }

  active_memory_ += buf->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);

  // Maintain the cache below the requested limit
  if (get_cache_memory() > max_pool_size_) {
    num_resources_ -= buffer_cache_.release_cached_buffers(
        get_cache_memory() - max_pool_size_);
  }

  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  num_resources_ -= buffer_cache_.clear();
}

void MetalAllocator::free(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->length();
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    num_resources_--;
    if (!buf->heap()) {
      residency_set_.erase(buf);
    }
    lk.unlock();
    auto pool = metal::new_scoped_memory_pool();
    buf->release();
  }
}

size_t MetalAllocator::size(Buffer buffer) const {
  return static_cast<MTL::Buffer*>(buffer.ptr())->length();
}

Buffer MetalAllocator::make_buffer(void* ptr, size_t size) {
  auto buf = device_->newBuffer(ptr, size, resource_options, nullptr);
  if (!buf) {
    return Buffer{nullptr};
  }
  std::unique_lock lk(mutex_);
  residency_set_.insert(buf);
  active_memory_ += buf->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);
  num_resources_++;
  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::release(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->length();
  num_resources_--;
  residency_set_.erase(buf);
  lk.unlock();
  auto pool = metal::new_scoped_memory_pool();
  buf->release();
}

MetalAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of MetalAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static MetalAllocator* allocator_ =
      new MetalAllocator(device(mlx::core::Device::gpu));
  return *allocator_;
}

} // namespace metal

size_t set_cache_limit(size_t limit) {
  return metal::allocator().set_cache_limit(limit);
}
size_t set_memory_limit(size_t limit) {
  return metal::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return metal::allocator().get_memory_limit();
}
size_t set_wired_limit(size_t limit) {
  if (limit > std::get<size_t>(
                  gpu::device_info(0).at("max_recommended_working_set_size"))) {
    throw std::invalid_argument(
        "[metal::set_wired_limit] Setting a wired limit larger than "
        "the maximum working set size is not allowed.");
  }
  return metal::allocator().set_wired_limit(limit);
}
size_t get_active_memory() {
  return metal::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return metal::allocator().get_peak_memory();
}
void reset_peak_memory() {
  metal::allocator().reset_peak_memory();
}
size_t get_cache_memory() {
  return metal::allocator().get_cache_memory();
}
size_t get_num_resources() {
  return metal::allocator().get_num_resources();
}
size_t get_resource_limit() {
  return metal::allocator().get_resource_limit();
}
void clear_cache() {
  return metal::allocator().clear_cache();
}

} // namespace mlx::core
