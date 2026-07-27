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
#include <cstring>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "func_selector.h"
#include "ihevc_typedefs.h"
#include "TestCommon.h"
// clang-format on

namespace {

// ---------------------------- Test Param -----------------------------------

using PaddingTestParam = std::tuple<std::pair<int, int>, int, IVD_ARCH_T>;

std::string PrintPaddingTestParam(
    const testing::TestParamInfo<PaddingTestParam>& info) {
  int wd, ht, pad_size;
  std::pair<int, int> block_size;
  IVD_ARCH_T arch;
  std::tie(block_size, pad_size, arch) = info.param;
  std::tie(wd, ht) = block_size;
  return std::to_string(wd) + "x" + std::to_string(ht) + "_pad_" +
         std::to_string(pad_size) + "_" + get_arch_str(arch);
}

// ---------------------------- Luma Base Class ------------------------------

class PaddingLumaTest : public ::testing::TestWithParam<PaddingTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, pad_size, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    stride = wd + 2 * pad_size + 16;
    total_ht = ht + 2 * pad_size + 16;

    buf_size = stride * total_ht;
    buf_ref.resize(buf_size, 0xAA);
    buf_tst.resize(buf_size, 0xAA);

    src_offset = pad_size * stride + pad_size + 8;

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (int r = 0; r < ht; r++) {
      for (int c = 0; c < wd; c++) {
        uint8_t val = dist(rng);
        buf_ref[src_offset + r * stride + c] = val;
        buf_tst[src_offset + r * stride + c] = val;
      }
    }
  }

  int wd, ht, pad_size;
  IVD_ARCH_T arch;
  int stride, total_ht, buf_size, src_offset;
  std::vector<UWORD8> buf_ref;
  std::vector<UWORD8> buf_tst;
  const func_selector_t* ref;
  const func_selector_t* tst;
};

// --------------------------- Chroma Base Class -----------------------------

class PaddingChromaTest : public ::testing::TestWithParam<PaddingTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, pad_size, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    // For chroma, width of block is 2 * wd in bytes
    stride = 2 * wd + 2 * pad_size + 16;
    total_ht = ht + 2 * pad_size + 16;

    buf_size = stride * total_ht;
    buf_ref.resize(buf_size, 0xAA);
    buf_tst.resize(buf_size, 0xAA);

    src_offset = pad_size * stride + pad_size + 8;

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (int r = 0; r < ht; r++) {
      for (int c = 0; c < 2 * wd; c++) {
        uint8_t val = dist(rng);
        buf_ref[src_offset + r * stride + c] = val;
        buf_tst[src_offset + r * stride + c] = val;
      }
    }
  }

  int wd, ht, pad_size;
  IVD_ARCH_T arch;
  int stride, total_ht, buf_size, src_offset;
  std::vector<UWORD8> buf_ref;
  std::vector<UWORD8> buf_tst;
  const func_selector_t* ref;
  const func_selector_t* tst;
};

// ---------------------------- Test cases -----------------------------------

class PaddingLeftLumaTest : public PaddingLumaTest {};
TEST_P(PaddingLeftLumaTest, Run) {
  InitializeBuffers();
  (ref->*(&func_selector_t::ihevc_pad_left_luma_fptr))(
      buf_ref.data() + src_offset, stride, ht, pad_size);
  (tst->*(&func_selector_t::ihevc_pad_left_luma_fptr))(
      buf_tst.data() + src_offset, stride, ht, pad_size);
  ASSERT_EQ(buf_ref, buf_tst);
}

class PaddingLeftChromaTest : public PaddingChromaTest {};
TEST_P(PaddingLeftChromaTest, Run) {
  InitializeBuffers();
  (ref->*(&func_selector_t::ihevc_pad_left_chroma_fptr))(
      buf_ref.data() + src_offset, stride, ht, pad_size);
  (tst->*(&func_selector_t::ihevc_pad_left_chroma_fptr))(
      buf_tst.data() + src_offset, stride, ht, pad_size);
  ASSERT_EQ(buf_ref, buf_tst);
}

class PaddingRightLumaTest : public PaddingLumaTest {};
TEST_P(PaddingRightLumaTest, Run) {
  InitializeBuffers();
  int pad_offset = wd;
  (ref->*(&func_selector_t::ihevc_pad_right_luma_fptr))(
      buf_ref.data() + src_offset + pad_offset, stride, ht, pad_size);
  (tst->*(&func_selector_t::ihevc_pad_right_luma_fptr))(
      buf_tst.data() + src_offset + pad_offset, stride, ht, pad_size);
  ASSERT_EQ(buf_ref, buf_tst);
}

class PaddingRightChromaTest : public PaddingChromaTest {};
TEST_P(PaddingRightChromaTest, Run) {
  InitializeBuffers();
  int pad_offset = 2 * wd;
  (ref->*(&func_selector_t::ihevc_pad_right_chroma_fptr))(
      buf_ref.data() + src_offset + pad_offset, stride, ht, pad_size);
  (tst->*(&func_selector_t::ihevc_pad_right_chroma_fptr))(
      buf_tst.data() + src_offset + pad_offset, stride, ht, pad_size);
  ASSERT_EQ(buf_ref, buf_tst);
}

// ---------------------------- Instantiation --------------------------------

auto kLumaPaddingParams = ::testing::Combine(
    ::testing::ValuesIn(kPUBlockSizes),
    ::testing::Values(80), ::testing::ValuesIn(ga_tst_arch));

const std::vector<std::pair<int, int>> kChromaPUBlockSizes = []() {
  std::vector<std::pair<int, int>> ret;
  for (const auto& size : kPUBlockSizes) {
    if ((size.second / 2) % 4 == 0) {
      ret.push_back({size.first / 2, size.second / 2});
    }
  }
  return ret;
}();

auto kChromaPaddingParams = ::testing::Combine(
    ::testing::ValuesIn(kChromaPUBlockSizes),
    ::testing::Values(80), ::testing::ValuesIn(ga_tst_arch));

INSTANTIATE_TEST_SUITE_P(Padding, PaddingLeftLumaTest, kLumaPaddingParams,
                         PrintPaddingTestParam);
INSTANTIATE_TEST_SUITE_P(Padding, PaddingRightLumaTest, kLumaPaddingParams,
                         PrintPaddingTestParam);

INSTANTIATE_TEST_SUITE_P(Padding, PaddingLeftChromaTest, kChromaPaddingParams,
                         PrintPaddingTestParam);
INSTANTIATE_TEST_SUITE_P(Padding, PaddingRightChromaTest, kChromaPaddingParams,
                         PrintPaddingTestParam);

}  // namespace
