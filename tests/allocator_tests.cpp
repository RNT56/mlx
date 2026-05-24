// Copyright © 2023 Apple Inc.

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/allocator.h"
#include "mlx/memory.h"

using namespace mlx::core;

TEST_CASE("test simple allocations") {
  {
    auto buffer = allocator::malloc(sizeof(float));
    auto fptr = static_cast<float*>(buffer.raw_ptr());
    *fptr = 0.5f;
    CHECK_EQ(*fptr, 0.5f);
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(128 * sizeof(int));
    int* ptr = static_cast<int*>(buffer.raw_ptr());
    for (int i = 0; i < 128; ++i) {
      ptr[i] = i;
    }
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(0);
    allocator::free(buffer);
  }
}

TEST_CASE("test large allocations") {
  size_t size = 1 << 30;
  for (int i = 0; i < 100; ++i) {
    auto buffer = allocator::malloc(size);
    allocator::free(buffer);
  }
}

TEST_CASE("test cached allocation keeps capacity") {
  auto old_limit = set_cache_limit(1 << 20);
  clear_cache();

  auto large = allocator::malloc(8192);
  allocator::free(large);
  auto cached = get_cache_memory();
  CHECK_GE(cached, 8192);

  auto small = allocator::malloc(6000);
  CHECK_GE(allocator::allocator().size(small), 8192);
  allocator::free(small);
  CHECK_GE(get_cache_memory(), cached);

  clear_cache();
  set_cache_limit(old_limit);
}

TEST_CASE("test concurrent cached allocations keep accounting balanced") {
  auto old_limit = set_cache_limit(4 << 20);
  clear_cache();
  auto initial_active = get_active_memory();
  std::atomic<bool> ok{true};

  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([t, &ok] {
      for (int i = 0; i < 512; ++i) {
        size_t size = 256 + ((i + t) % 64) * 31;
        auto buffer = allocator::malloc(size);
        auto ptr = static_cast<char*>(buffer.raw_ptr());
        if (ptr == nullptr || allocator::allocator().size(buffer) < size) {
          ok = false;
        } else {
          ptr[0] = static_cast<char>(t);
          ptr[size - 1] = static_cast<char>(i);
        }
        allocator::free(buffer);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  CHECK(ok);
  CHECK_EQ(get_active_memory(), initial_active);
  CHECK_GT(get_cache_memory(), 0);

  clear_cache();
  CHECK_EQ(get_cache_memory(), 0);
  set_cache_limit(old_limit);
}
