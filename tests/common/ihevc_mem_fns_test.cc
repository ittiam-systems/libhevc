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

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

// clang-format off
#include "func_selector.h"
#include "ihevc_typedefs.h"
#include "tests_common.h"
// clang-format on

namespace {

// Param: offset (0..7 words), num_words, value, arch
using MemsetTestParam = std::tuple<int, int, UWORD16, IVD_ARCH_T>;

class Memset16BitTest : public ::testing::TestWithParam<MemsetTestParam> {
 protected:
  void SetUp() override {
    std::tie(offset, num_words, value, arch) = GetParam();
    buf_size = num_words + offset + 32;
    dst_ref.resize(buf_size, 0xDEAD);
    dst_tst.resize(buf_size, 0xDEAD);
    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  int offset;
  int num_words;
  UWORD16 value;
  IVD_ARCH_T arch;
  int buf_size;
  std::vector<UWORD16> dst_ref;
  std::vector<UWORD16> dst_tst;
  const func_selector_t* ref;
  const func_selector_t* tst;
};

TEST_P(Memset16BitTest, Run) {
  (ref->*(&func_selector_t::ihevc_memset_16bit_fptr))(dst_ref.data() + offset,
                                                      value, num_words);
  (tst->*(&func_selector_t::ihevc_memset_16bit_fptr))(dst_tst.data() + offset,
                                                      value, num_words);
  ASSERT_EQ(dst_ref, dst_tst);
}

// ---------------------------------------------------------------------------

class Memset16BitMul8Test : public ::testing::TestWithParam<MemsetTestParam> {
 protected:
  void SetUp() override {
    std::tie(offset, num_words, value, arch) = GetParam();
    buf_size = num_words + offset + 32;
    dst_ref.resize(buf_size, 0xDEAD);
    dst_tst.resize(buf_size, 0xDEAD);
    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  int offset;
  int num_words;
  UWORD16 value;
  IVD_ARCH_T arch;
  int buf_size;
  std::vector<UWORD16> dst_ref;
  std::vector<UWORD16> dst_tst;
  const func_selector_t* ref;
  const func_selector_t* tst;
};

TEST_P(Memset16BitMul8Test, Run) {
  (ref->*(&func_selector_t::ihevc_memset_16bit_mul_8_fptr))(
      dst_ref.data() + offset, value, num_words);
  (tst->*(&func_selector_t::ihevc_memset_16bit_mul_8_fptr))(
      dst_tst.data() + offset, value, num_words);
  ASSERT_EQ(dst_ref, dst_tst);
}

// ---------------------------------------------------------------------------

std::vector<MemsetTestParam> GenerateMemsetParams() {
  std::vector<MemsetTestParam> params;
  int offsets[] = {0, 1, 3, 7};
  int sizes[] = {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 127, 128};
  UWORD16 values[] = {0, 0x5555, 0xFFFF};

  for (auto off : offsets) {
    for (auto sz : sizes) {
      for (auto val : values) {
        for (auto arch : ga_tst_arch) {
          params.emplace_back(off, sz, val, arch);
        }
      }
    }
  }
  return params;
}

std::vector<MemsetTestParam> GenerateMemsetMul8Params() {
  std::vector<MemsetTestParam> params;
  int offsets[] = {0, 1, 3, 7};
  int sizes[] = {8, 16, 24, 32, 48, 64, 96, 128, 256};
  UWORD16 values[] = {0, 0x5555, 0xFFFF};

  for (auto off : offsets) {
    for (auto sz : sizes) {
      for (auto val : values) {
        for (auto arch : ga_tst_arch) {
          params.emplace_back(off, sz, val, arch);
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(MemFns, Memset16BitTest,
                         ::testing::ValuesIn(GenerateMemsetParams()));

INSTANTIATE_TEST_SUITE_P(MemFns, Memset16BitMul8Test,
                         ::testing::ValuesIn(GenerateMemsetMul8Params()));

}  // namespace
