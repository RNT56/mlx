// Copyright © 2024 Apple Inc.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {
// TQ_HOST_PROBE_V1: env-gated host-overhead probe (roadmap G2).
// Zero behavior change unless TQ_HOST_PROBE=1 is set in the environment.
struct TQHostProbeCustomKernelStats {
  std::atomic<uint64_t> evals{0};
  std::atomic<uint64_t> compare_ns{0};
  std::atomic<uint64_t> cache_invalidations{0};
  std::atomic<uint64_t> library_builds{0};
  std::atomic<uint64_t> library_build_ns{0};
};
inline TQHostProbeCustomKernelStats& tq_host_probe_custom_kernel_stats() {
  static TQHostProbeCustomKernelStats stats;
  return stats;
}
inline void tq_host_probe_custom_kernel_emit() {
  auto& s = tq_host_probe_custom_kernel_stats();
  std::fprintf(
      stderr,
      "TQ_HOST_PROBE_V1 custom_kernel evals=%llu compare_ns=%llu cache_invalidations=%llu library_builds=%llu library_build_ns=%llu\n",
      static_cast<unsigned long long>(s.evals.load()),
      static_cast<unsigned long long>(s.compare_ns.load()),
      static_cast<unsigned long long>(s.cache_invalidations.load()),
      static_cast<unsigned long long>(s.library_builds.load()),
      static_cast<unsigned long long>(s.library_build_ns.load()));
}
inline bool tq_host_probe_custom_kernel_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("TQ_HOST_PROBE");
    bool on = value != nullptr && std::string_view(value) == "1";
    if (on) {
      std::atexit(tq_host_probe_custom_kernel_emit);
    }
    return on;
  }();
  return enabled;
}
} // namespace

struct CustomKernelCache {
  std::unordered_map<std::string, std::string> libraries;
};

static CustomKernelCache& cache() {
  static CustomKernelCache cache_;
  return cache_;
};

// TQ_T11: the CustomKernelCache above is find/emplace/assigned from eval_gpu
// concurrently across streams; guard it explicitly instead of relying on
// unsynchronized access.
static std::mutex& cache_mutex() {
  static std::mutex m;
  return m;
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // silence some warnings
  (void)is_precompiled_;
  (void)shared_memory_;

  const bool tq_probe = tq_host_probe_custom_kernel_enabled();
  if (tq_probe) {
    auto n = tq_host_probe_custom_kernel_stats().evals.fetch_add(1) + 1;
    if (n % 1000 == 0) {
      tq_host_probe_custom_kernel_emit();
    }
  }

  auto& s = stream();

  std::vector<array> copies;

  for (auto& out : outputs) {
    if (init_value_) {
      copies.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(copies.back(), out, s);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
  }

  auto check_input = [&copies, &s, this](const array& x) -> const array {
    bool no_copy = x.flags().row_contiguous;
    if (!ensure_row_contiguous_ || no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  std::vector<array> checked_inputs;
  for (const array& in : inputs) {
    checked_inputs.push_back(check_input(in));
  }

  auto& d = metal::device(s.device);

  {
    std::chrono::steady_clock::time_point tq_compare_start;
    if (tq_probe) {
      tq_compare_start = std::chrono::steady_clock::now();
    }
    std::lock_guard<std::mutex> lock(cache_mutex());
    // Clear kernels from the device library cache if needed
    auto& kernel_cache = cache();
    if (auto it = kernel_cache.libraries.find(name_);
        it != kernel_cache.libraries.end()) {
      if (it->second != source_) {
        auto& d = metal::device(s.device);
        d.clear_library(name_);
        it->second = source_;
        if (tq_probe) {
          tq_host_probe_custom_kernel_stats().cache_invalidations.fetch_add(
              1, std::memory_order_relaxed);
        }
      }
    } else {
      kernel_cache.libraries.emplace(name_, source_);
    }
    if (tq_probe) {
      auto tq_compare_elapsed =
          std::chrono::steady_clock::now() - tq_compare_start;
      tq_host_probe_custom_kernel_stats().compare_ns.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  tq_compare_elapsed)
                  .count()),
          std::memory_order_relaxed);
    }
  }

  auto lib = d.get_library(name_, [this, tq_probe] {
    std::chrono::steady_clock::time_point tq_build_start;
    if (tq_probe) {
      tq_build_start = std::chrono::steady_clock::now();
      tq_host_probe_custom_kernel_stats().library_builds.fetch_add(
          1, std::memory_order_relaxed);
    }
    auto source = metal::utils() + source_;
    if (tq_probe) {
      auto tq_build_elapsed =
          std::chrono::steady_clock::now() - tq_build_start;
      tq_host_probe_custom_kernel_stats().library_build_ns.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  tq_build_elapsed).count()),
          std::memory_order_relaxed);
    }
    return source;
  });
  auto kernel = d.get_kernel(name_, lib);

  // TQ_PIPE_PROBE_V1: env-gated occupancy/pipeline-property probe (roadmap G5).
  // Zero behavior change unless TQ_PIPELINE_PROBE=1. Prints once per distinct
  // compiled kernel name (function-constant specialization is baked into name_).
  if (const char* v = std::getenv("TQ_PIPELINE_PROBE"); v && std::string_view(v) == "1") {
    static std::mutex tq_pipe_probe_mu;
    static std::unordered_set<std::string> tq_pipe_probe_seen;
    std::lock_guard<std::mutex> lock(tq_pipe_probe_mu);
    if (tq_pipe_probe_seen.insert(name_).second) {
      std::fprintf(stderr,
        "TQ_PIPE_PROBE_V1 name=%s static_tgmem=%llu max_threads_per_tg=%llu exec_width=%llu\n",
        name_.c_str(),
        (unsigned long long)kernel->staticThreadgroupMemoryLength(),
        (unsigned long long)kernel->maxTotalThreadsPerThreadgroup(),
        (unsigned long long)kernel->threadExecutionWidth());
    }
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  int index = 0;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    compute_encoder.set_input_array(in, index);
    index++;
    if (in.ndim() > 0) {
      int ndim = in.ndim();
      if (std::get<0>(shape_info)) {
        compute_encoder.set_vector_bytes(in.shape(), ndim, index);
        index++;
      }
      if (std::get<1>(shape_info)) {
        compute_encoder.set_vector_bytes(in.strides(), ndim, index);
        index++;
      }
      if (std::get<2>(shape_info)) {
        compute_encoder.set_bytes(ndim, index);
        index++;
      }
    }
  }
  for (auto& out : outputs) {
    compute_encoder.set_output_array(out, index);
    index++;
  }

  const auto [tx, ty, tz] = threadgroup_;
  auto tg_size = tx * ty * tz;
  auto max_tg_size = kernel->maxTotalThreadsPerThreadgroup();
  if (tg_size > max_tg_size) {
    std::ostringstream msg;
    msg << "Thread group size (" << tg_size << ") is greater than "
        << " the maximum allowed threads per threadgroup (" << max_tg_size
        << ").";
    throw std::invalid_argument(msg.str());
  }

  const auto [gx, gy, gz] = grid_;
  MTL::Size group_dims =
      MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz));
  MTL::Size grid_dims = MTL::Size(gx, gy, gz);
  compute_encoder.dispatch_threads(grid_dims, group_dims);

  compute_encoder.add_temporaries(std::move(copies));
}

} // namespace mlx::core::fast
