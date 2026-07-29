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
#include "ihevc_typedefs.h"
#include "func_selector.h"
#include "TestCommon.h"
// clang-format on

namespace {

void compare_sao_output(const UWORD8* ref, const UWORD8* tst, int stride,
                        int wd, int ht) {
  for (int r = 0; r < ht; r++) {
    for (int c = 0; c < wd; c++) {
      ASSERT_EQ(ref[r * stride + c], tst[r * stride + c])
          << "Mismatch at row " << r << ", col " << c;
    }
  }
}

// ---------------------------- Test Param -----------------------------------

// Param: block size, sao_band_pos/edge_class (value), arch
using SaoTestParam = std::tuple<std::pair<int, int>, int, IV_ARCH_T>;

std::string PrintSaoTestParam(
    const testing::TestParamInfo<SaoTestParam>& info) {
  int wd, ht, val;
  IV_ARCH_T arch;
  std::pair<int, int> block_size;
  std::tie(block_size, val, arch) = info.param;
  std::tie(wd, ht) = block_size;
  return std::to_string(wd) + "x" + std::to_string(ht) + "_val_" +
         std::to_string(val) + "_" + get_arch_str(arch);
}

// ---------------------------- Luma Base Class ------------------------------

class SaoLumaTest : public ::testing::TestWithParam<SaoTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, offset_val, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    stride = wd + 32;
    total_ht = ht + 32;
    src_size = stride * total_ht;
    src_offset = 16 * stride + 16;

    src_ref.resize(src_size);
    src_tst.resize(src_size);

    src_left_ref.resize(ht + 8 + 1);
    src_left_tst.resize(ht + 8 + 1);
    src_top_ref.resize(wd + 8);
    src_top_tst.resize(wd + 8);
    src_top_left_ref.resize(8);
    src_top_left_tst.resize(8);
    src_top_right_ref.resize(8);
    src_top_right_tst.resize(8);
    src_bot_left_ref.resize(8);
    src_bot_left_tst.resize(8);

    avail.resize(8);

    sao_offset.resize(8);
    sao_offset[0] = 0;

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    std::uniform_int_distribution<uint8_t> dist_avail(0, 1);
    std::uniform_int_distribution<int8_t> dist_offset(-7, 7);

    for (int i = 0; i < src_size; i++) {
      uint8_t val = dist(rng);
      src_ref[i] = val;
      src_tst[i] = val;
    }

    for (size_t i = 0; i < src_left_ref.size(); i++) {
      uint8_t val = dist(rng);
      src_left_ref[i] = val;
      src_left_tst[i] = val;
    }

    for (size_t i = 0; i < src_top_ref.size(); i++) {
      uint8_t val = dist(rng);
      src_top_ref[i] = val;
      src_top_tst[i] = val;
    }

    for (size_t i = 0; i < src_top_left_ref.size(); i++) {
      src_top_left_ref[i] = src_top_left_tst[i] = dist(rng);
      src_top_right_ref[i] = src_top_right_tst[i] = dist(rng);
      src_bot_left_ref[i] = src_bot_left_tst[i] = dist(rng);
    }

    src_left_ref[0] = src_left_tst[0] = src_top_left_ref[0];

    for (int i = 0; i < 8; i++) {
      avail[i] = dist_avail(rng) ? 255 : 0;
    }

    for (int i = 1; i < 5; i++) {
      sao_offset[i] = dist_offset(rng);
    }
  }

  int wd, ht, offset_val;
  IV_ARCH_T arch;
  int stride, total_ht, src_size, src_offset;
  std::vector<UWORD8> src_ref;
  std::vector<UWORD8> src_tst;
  std::vector<UWORD8> src_left_ref;
  std::vector<UWORD8> src_left_tst;
  std::vector<UWORD8> src_top_ref;
  std::vector<UWORD8> src_top_tst;
  std::vector<UWORD8> src_top_left_ref;
  std::vector<UWORD8> src_top_left_tst;
  std::vector<UWORD8> src_top_right_ref;
  std::vector<UWORD8> src_top_right_tst;
  std::vector<UWORD8> src_bot_left_ref;
  std::vector<UWORD8> src_bot_left_tst;
  std::vector<UWORD8> avail;
  std::vector<WORD8> sao_offset;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// --------------------------- Chroma Base Class -----------------------------

class SaoChromaTest : public ::testing::TestWithParam<SaoTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, offset_val, arch) = GetParam();
    std::tie(wd, ht) =
        block_size;  // wd is chroma component size, 2 * wd is bytes

    stride = 2 * wd + 32;
    total_ht = ht + 32;
    src_size = stride * total_ht;
    src_offset = 16 * stride + 16;

    src_ref.resize(src_size);
    src_tst.resize(src_size);

    src_left_ref.resize(2 * ht + 8 + 2);
    src_left_tst.resize(2 * ht + 8 + 2);
    src_top_ref.resize(2 * wd + 8);
    src_top_tst.resize(2 * wd + 8);
    src_top_left_ref.resize(8);
    src_top_left_tst.resize(8);
    src_top_right_ref.resize(8);
    src_top_right_tst.resize(8);
    src_bot_left_ref.resize(8);
    src_bot_left_tst.resize(8);

    avail.resize(8);

    sao_offset_u.resize(8);
    sao_offset_v.resize(8);
    sao_offset_u[0] = sao_offset_v[0] = 0;

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    std::uniform_int_distribution<uint8_t> dist_avail(0, 1);
    std::uniform_int_distribution<int8_t> dist_offset(-7, 7);

    for (int i = 0; i < src_size; i++) {
      uint8_t val = dist(rng);
      src_ref[i] = val;
      src_tst[i] = val;
    }

    for (size_t i = 0; i < src_left_ref.size(); i++) {
      uint8_t val = dist(rng);
      src_left_ref[i] = val;
      src_left_tst[i] = val;
    }

    for (size_t i = 0; i < src_top_ref.size(); i++) {
      uint8_t val = dist(rng);
      src_top_ref[i] = val;
      src_top_tst[i] = val;
    }

    for (size_t i = 0; i < src_top_left_ref.size(); i++) {
      src_top_left_ref[i] = src_top_left_tst[i] = dist(rng);
      src_top_right_ref[i] = src_top_right_tst[i] = dist(rng);
      src_bot_left_ref[i] = src_bot_left_tst[i] = dist(rng);
    }

    src_left_ref[0] = src_left_tst[0] = src_top_left_ref[0];
    src_left_ref[1] = src_left_tst[1] = src_top_left_ref[1];

    for (int i = 0; i < 8; i++) {
      avail[i] = dist_avail(rng) ? 255 : 0;
    }

    for (int i = 1; i < 5; i++) {
      sao_offset_u[i] = dist_offset(rng);
      sao_offset_v[i] = dist_offset(rng);
    }
  }

  int wd, ht, offset_val;
  IV_ARCH_T arch;
  int stride, total_ht, src_size, src_offset;
  std::vector<UWORD8> src_ref;
  std::vector<UWORD8> src_tst;
  std::vector<UWORD8> src_left_ref;
  std::vector<UWORD8> src_left_tst;
  std::vector<UWORD8> src_top_ref;
  std::vector<UWORD8> src_top_tst;
  std::vector<UWORD8> src_top_left_ref;
  std::vector<UWORD8> src_top_left_tst;
  std::vector<UWORD8> src_top_right_ref;
  std::vector<UWORD8> src_top_right_tst;
  std::vector<UWORD8> src_bot_left_ref;
  std::vector<UWORD8> src_bot_left_tst;
  std::vector<UWORD8> avail;
  std::vector<WORD8> sao_offset_u;
  std::vector<WORD8> sao_offset_v;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// ---------------------------- Test cases -----------------------------------

class SaoBandOffsetLumaTest : public SaoLumaTest {};
TEST_P(SaoBandOffsetLumaTest, Run) {
  InitializeBuffers();

  ref->ihevc_sao_band_offset_luma_fptr(src_ref.data() + src_offset, stride,
                                       src_left_ref.data(), src_top_ref.data(),
                                       src_top_left_ref.data(), offset_val,
                                       sao_offset.data(), wd, ht);

  tst->ihevc_sao_band_offset_luma_fptr(src_tst.data() + src_offset, stride,
                                       src_left_tst.data(), src_top_tst.data(),
                                       src_top_left_tst.data(), offset_val,
                                       sao_offset.data(), wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoBandOffsetChromaTest : public SaoChromaTest {};
TEST_P(SaoBandOffsetChromaTest, Run) {
  InitializeBuffers();

  int offset_val_u = offset_val;
  int offset_val_v = (offset_val + 4) % 32;

  ref->ihevc_sao_band_offset_chroma_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data(),
      src_top_ref.data(), src_top_left_ref.data(), offset_val_u, offset_val_v,
      sao_offset_u.data(), sao_offset_v.data(), 2 * wd, ht);

  tst->ihevc_sao_band_offset_chroma_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data(),
      src_top_tst.data(), src_top_left_tst.data(), offset_val_u, offset_val_v,
      sao_offset_u.data(), sao_offset_v.data(), 2 * wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, 2 * wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass0Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass0Test, Run) {
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class0_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data(),
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset.data(), wd, ht);

  tst->ihevc_sao_edge_offset_class0_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data(),
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset.data(), wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass0ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass0ChromaTest, Run) {
#if defined(__arm__) || defined(__aarch64__) || defined(__arm64__)
  if (wd % 8 == 4) {
    GTEST_SKIP() << "Skipping failing ARM Chroma SAO Class 0 tests for width % 8 == 4";
  }
#endif
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class0_chroma_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data(),
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  tst->ihevc_sao_edge_offset_class0_chroma_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data(),
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, 2 * wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass1Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass1Test, Run) {
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class1_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data(),
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset.data(), wd, ht);

  tst->ihevc_sao_edge_offset_class1_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data(),
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset.data(), wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass1ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass1ChromaTest, Run) {
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class1_chroma_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data(),
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  tst->ihevc_sao_edge_offset_class1_chroma_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data(),
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, 2 * wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass2Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass2Test, Run) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
  if (arch == ARCH_X86_SSSE3 || arch == ARCH_X86_SSE42 ||
      arch == ARCH_X86_AVX2) {
    GTEST_SKIP() << "Skipping Class 2 tests for x86 SIMD";
  }
#endif
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class2_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data() + 1,
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset.data(), wd, ht);

  tst->ihevc_sao_edge_offset_class2_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data() + 1,
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset.data(), wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass2ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass2ChromaTest, Run) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
  if (arch == ARCH_X86_SSSE3 || arch == ARCH_X86_SSE42 ||
      arch == ARCH_X86_AVX2) {
    GTEST_SKIP() << "Skipping Class 2 Chroma tests for x86 SIMD";
  }
#endif
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class2_chroma_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data() + 2,
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  tst->ihevc_sao_edge_offset_class2_chroma_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data() + 2,
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, 2 * wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass3Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass3Test, Run) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
  if (arch == ARCH_X86_SSSE3 || arch == ARCH_X86_SSE42 ||
      arch == ARCH_X86_AVX2) {
    GTEST_SKIP() << "Skipping Class 3 tests for x86 SIMD";
  }
#endif
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class3_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data(),
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset.data(), wd, ht);

  tst->ihevc_sao_edge_offset_class3_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data(),
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset.data(), wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

class SaoEdgeOffsetClass3ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass3ChromaTest, Run) {
  InitializeBuffers();

  ref->ihevc_sao_edge_offset_class3_chroma_fptr(
      src_ref.data() + src_offset, stride, src_left_ref.data(),
      src_top_ref.data(), src_top_left_ref.data(), src_top_right_ref.data(),
      src_bot_left_ref.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  tst->ihevc_sao_edge_offset_class3_chroma_fptr(
      src_tst.data() + src_offset, stride, src_left_tst.data(),
      src_top_tst.data(), src_top_left_tst.data(), src_top_right_tst.data(),
      src_bot_left_tst.data(), avail.data(), sao_offset_u.data(),
      sao_offset_v.data(), 2 * wd, ht);

  compare_sao_output(src_ref.data() + src_offset, src_tst.data() + src_offset,
                     stride, 2 * wd, ht);
  ASSERT_EQ(src_left_ref, src_left_tst);
  ASSERT_EQ(src_top_ref, src_top_tst);
  ASSERT_EQ(src_top_left_ref, src_top_left_tst);
}

// ---------------------------- Instantiation --------------------------------

std::vector<std::pair<int, int>> GetSaoLumaBlockSizes() {
  std::vector<std::pair<int, int>> sizes;
  for (const auto& size : kPUBlockSizes) {
    if (size.first >= 8 && size.first % 8 == 0 && size.second >= 4 &&
        size.second % 4 == 0) {
      sizes.push_back(size);
    }
  }
  return sizes;
}

std::vector<std::pair<int, int>> GetSaoChromaBlockSizes() {
  std::vector<std::pair<int, int>> sizes;
  for (const auto& size : kPUBlockSizes) {
    int wd_comp = size.first / 2;
    int ht_comp = size.second / 2;
    if (wd_comp >= 4 && wd_comp % 4 == 0 && ht_comp >= 4 && ht_comp % 4 == 0) {
      sizes.push_back({wd_comp, ht_comp});
    }
  }
  return sizes;
}

// Luma Params: block sizes, band_pos/class, arch
auto kSaoLumaParams = ::testing::Combine(
    ::testing::ValuesIn(GetSaoLumaBlockSizes()),
    ::testing::Values(0, 7, 15, 23, 28), ::testing::ValuesIn(ga_tst_arch));

// Chroma Params: block sizes, band_pos/class, arch
auto kSaoChromaParams = ::testing::Combine(
    ::testing::ValuesIn(GetSaoChromaBlockSizes()),
    ::testing::Values(0, 7, 15, 23, 28), ::testing::ValuesIn(ga_tst_arch));

INSTANTIATE_TEST_SUITE_P(Sao, SaoBandOffsetLumaTest, kSaoLumaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoBandOffsetChromaTest, kSaoChromaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass0Test, kSaoLumaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass0ChromaTest, kSaoChromaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass1Test, kSaoLumaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass1ChromaTest, kSaoChromaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass2Test, kSaoLumaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass2ChromaTest, kSaoChromaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass3Test, kSaoLumaParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass3ChromaTest, kSaoChromaParams,
                         PrintSaoTestParam);

}  // namespace
