/******************************************************************************
 *
 * Copyright (C) 2026 Ittiam Systems Pvt Ltd, Bangalore
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_inter_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"

// clang-format on

#include "TestEnums.h"

static constexpr int kMaxSize = 64;
static constexpr int kTapSize = 8;
static constexpr int kMaxHeight = kMaxSize + kTapSize;

const std::vector<std::pair<int, int>>& getLumaPUBlockSizes();
const std::vector<std::pair<int, int>>& getChromaPUBlockSizes();
const std::vector<UWORD8>& getSrc8Buf();
const std::vector<IV_ARCH_T>& getTstArch();
std::string get_arch_str(IV_ARCH_T arch);

// Computes HEVC zero-column / zero-row bitmask for a given transform size
inline WORD32 ComputeZeroMask(int trans_size, int non_zero_count) {
  if (non_zero_count >= trans_size) {
    return 0;
  }
  WORD32 full_mask = (trans_size == 32)
                         ? 0xFFFFFFFFu
                         : ((static_cast<WORD32>(1u) << trans_size) - 1u);
  WORD32 nz_mask = (static_cast<WORD32>(1u) << non_zero_count) - 1u;
  return (~nz_mask) & full_mask;
}

// Deterministic pseudo-random initialization of a buffer
template <typename T>
void FillRandom(T* data, size_t count, T min_val, T max_val,
                uint32_t seed = 42) {
  std::mt19937 rng(seed);
  if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
    std::uniform_int_distribution<int> dist(static_cast<int>(min_val),
                                            static_cast<int>(max_val));
    for (size_t i = 0; i < count; ++i) {
      data[i] = static_cast<T>(dist(rng));
    }
  } else if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < count; ++i) {
      data[i] = dist(rng);
    }
  }
}

template <typename T>
void FillRandom(std::vector<T>& vec, T min_val, T max_val, uint32_t seed = 42) {
  FillRandom(vec.data(), vec.size(), min_val, max_val, seed);
}

// Populates a 2D buffer (trans_size x trans_size, stride src_strd) with random
// values in [min_val, max_val] within top-left [0, nz_rows) x [0, nz_cols),
// zeroing out all other entries.
template <typename T>
void FillRandomSubBlock(T* data, int trans_size, int src_strd, int nz_rows,
                        int nz_cols, T min_val, T max_val,
                        uint32_t seed = 42) {
  std::fill(data, data + trans_size * src_strd, static_cast<T>(0));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<T> dist(min_val, max_val);
  for (int r = 0; r < trans_size; ++r) {
    for (int c = 0; c < trans_size; ++c) {
      if (r < nz_rows && c < nz_cols) {
        data[r * src_strd + c] = dist(rng);
      }
    }
  }
}

template <typename T>
void FillRandomSubBlock(std::vector<T>& vec, int trans_size, int src_strd,
                        int nz_rows, int nz_cols, T min_val, T max_val,
                        uint32_t seed = 42) {
  FillRandomSubBlock(vec.data(), trans_size, src_strd, nz_rows, nz_cols,
                     min_val, max_val, seed);
}

// Verifies 2D output buffers row-by-row
template <typename T>
bool VerifyOutput2D(const T* ref, const T* tst, int wd, int ht, int ref_strd,
                    int tst_strd) {
  const size_t row_bytes = static_cast<size_t>(wd) * sizeof(T);
  for (int r = 0; r < ht; ++r) {
    if (std::memcmp(ref + r * ref_strd, tst + r * tst_strd, row_bytes) != 0) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool VerifyOutput2D(const T* ref, const T* tst, int wd, int ht, int strd) {
  return VerifyOutput2D(ref, tst, wd, ht, strd, strd);
}

#if __has_include(<gtest/gtest.h>)
#include <gtest/gtest.h>

// Compare outputs
template <typename T>
static void compare_output(const T* ref, const T* test, int wd, int ht,
                           int dst_strd) {
  int size_bytes = wd * sizeof(T);
  for (int i = 0; i < ht; ++i) {
    int cmp = memcmp(ref + i * dst_strd, test + i * dst_strd, size_bytes);
    ASSERT_EQ(0, cmp) << "Mismatch at row " << i << " for size " << wd << "x"
                      << ht;
  }
}

template <typename T>
static void compare_output(const std::vector<T>& ref,
                           const std::vector<T>& test, int wd, int ht,
                           int dst_strd) {
  compare_output(ref.data(), test.data(), wd, ht, dst_strd);
}
#endif
