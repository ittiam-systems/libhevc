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
#include "ihevc_sao_utils.h"
// clang-format on

namespace {

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

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    ref_buf = CreateSaoLumaBuffers<UWORD8>(wd, ht);
    tst_buf = ref_buf;
  }

  void VerifyBuffers() {
    compare_output(ref_buf.src.data() + ref_buf.src_offset,
                   tst_buf.src.data() + tst_buf.src_offset, wd, ht,
                   ref_buf.stride);
    ASSERT_EQ(ref_buf.src_left, tst_buf.src_left);
    ASSERT_EQ(ref_buf.src_top, tst_buf.src_top);
    ASSERT_EQ(ref_buf.src_top_left, tst_buf.src_top_left);
  }

  void RunEdgeOffset(int edge_class) {
    InitializeBuffers();
    int left_offset = (edge_class == 2) ? 1 : 0;

    auto* ref_fn = GetSaoEdgeOffsetLumaFn(ref, edge_class);
    auto* tst_fn = GetSaoEdgeOffsetLumaFn(tst, edge_class);

    ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
           ref_buf.src_left.data() + left_offset, ref_buf.src_top.data(),
           ref_buf.src_top_left.data(), ref_buf.src_top_right.data(),
           ref_buf.src_bot_left.data(), ref_buf.avail.data(),
           ref_buf.sao_offset.data(), wd, ht);

    tst_fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
           tst_buf.src_left.data() + left_offset, tst_buf.src_top.data(),
           tst_buf.src_top_left.data(), tst_buf.src_top_right.data(),
           tst_buf.src_bot_left.data(), tst_buf.avail.data(),
           tst_buf.sao_offset.data(), wd, ht);

    VerifyBuffers();
  }

  int wd, ht, offset_val;
  IV_ARCH_T arch;
  SaoLumaBuffers<UWORD8> ref_buf;
  SaoLumaBuffers<UWORD8> tst_buf;
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

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    ref_buf = CreateSaoChromaBuffers<UWORD8>(wd, ht);
    tst_buf = ref_buf;
  }

  void VerifyBuffers() {
    compare_output(ref_buf.src.data() + ref_buf.src_offset,
                   tst_buf.src.data() + tst_buf.src_offset, 2 * wd, ht,
                   ref_buf.stride);
    ASSERT_EQ(ref_buf.src_left, tst_buf.src_left);
    ASSERT_EQ(ref_buf.src_top, tst_buf.src_top);
    ASSERT_EQ(ref_buf.src_top_left, tst_buf.src_top_left);
  }

  void RunEdgeOffset(int edge_class) {
    InitializeBuffers();
    int left_offset = (edge_class == 2) ? 2 : 0;

    auto* ref_fn = GetSaoEdgeOffsetChromaFn(ref, edge_class);
    auto* tst_fn = GetSaoEdgeOffsetChromaFn(tst, edge_class);

    ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
           ref_buf.src_left.data() + left_offset, ref_buf.src_top.data(),
           ref_buf.src_top_left.data(), ref_buf.src_top_right.data(),
           ref_buf.src_bot_left.data(), ref_buf.avail.data(),
           ref_buf.sao_offset_u.data(), ref_buf.sao_offset_v.data(), 2 * wd,
           ht);

    tst_fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
           tst_buf.src_left.data() + left_offset, tst_buf.src_top.data(),
           tst_buf.src_top_left.data(), tst_buf.src_top_right.data(),
           tst_buf.src_bot_left.data(), tst_buf.avail.data(),
           tst_buf.sao_offset_u.data(), tst_buf.sao_offset_v.data(), 2 * wd,
           ht);

    VerifyBuffers();
  }

  int wd, ht, offset_val;
  IV_ARCH_T arch;
  SaoChromaBuffers<UWORD8> ref_buf;
  SaoChromaBuffers<UWORD8> tst_buf;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// ---------------------------- Test cases -----------------------------------

class SaoBandOffsetLumaTest : public SaoLumaTest {};
TEST_P(SaoBandOffsetLumaTest, Run) {
  InitializeBuffers();

  GetSaoBandOffsetLumaFn(ref)(
      ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
      ref_buf.src_left.data(), ref_buf.src_top.data(),
      ref_buf.src_top_left.data(), offset_val, ref_buf.sao_offset.data(), wd,
      ht);

  GetSaoBandOffsetLumaFn(tst)(
      tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
      tst_buf.src_left.data(), tst_buf.src_top.data(),
      tst_buf.src_top_left.data(), offset_val, tst_buf.sao_offset.data(), wd,
      ht);

  VerifyBuffers();
}

class SaoBandOffsetChromaTest : public SaoChromaTest {};
TEST_P(SaoBandOffsetChromaTest, Run) {
  InitializeBuffers();

  int offset_val_u = offset_val;
  int offset_val_v = (offset_val + 4) % 32;

  GetSaoBandOffsetChromaFn(ref)(
      ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
      ref_buf.src_left.data(), ref_buf.src_top.data(),
      ref_buf.src_top_left.data(), offset_val_u, offset_val_v,
      ref_buf.sao_offset_u.data(), ref_buf.sao_offset_v.data(), 2 * wd, ht);

  GetSaoBandOffsetChromaFn(tst)(
      tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
      tst_buf.src_left.data(), tst_buf.src_top.data(),
      tst_buf.src_top_left.data(), offset_val_u, offset_val_v,
      tst_buf.sao_offset_u.data(), tst_buf.sao_offset_v.data(), 2 * wd, ht);

  VerifyBuffers();
}

class SaoEdgeOffsetClass0Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass0Test, Run) { RunEdgeOffset(0); }

class SaoEdgeOffsetClass0ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass0ChromaTest, Run) {
#if defined(__arm__) || defined(__aarch64__) || defined(__arm64__)
  if (wd % 8 == 4) {
    GTEST_SKIP()
        << "Skipping failing ARM Chroma SAO Class 0 tests for width % 8 == 4";
  }
#endif
  RunEdgeOffset(0);
}

class SaoEdgeOffsetClass1Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass1Test, Run) { RunEdgeOffset(1); }

class SaoEdgeOffsetClass1ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass1ChromaTest, Run) { RunEdgeOffset(1); }

class SaoEdgeOffsetClass2Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass2Test, Run) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
  if (arch == ARCH_X86_SSSE3 || arch == ARCH_X86_SSE42 ||
      arch == ARCH_X86_AVX2) {
    GTEST_SKIP() << "Skipping Class 2 tests for x86 SIMD";
  }
#endif
  RunEdgeOffset(2);
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
  RunEdgeOffset(2);
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
  RunEdgeOffset(3);
}

class SaoEdgeOffsetClass3ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass3ChromaTest, Run) { RunEdgeOffset(3); }

// ---------------------------- Instantiation --------------------------------

std::vector<std::pair<int, int>> GetSaoLumaBlockSizes() {
  std::vector<std::pair<int, int>> sizes;
  for (const auto& size : getLumaPUBlockSizes()) {
    if (size.first >= 8 && size.first % 8 == 0 && size.second >= 4 &&
        size.second % 4 == 0) {
      sizes.push_back(size);
    }
  }
  return sizes;
}

std::vector<std::pair<int, int>> GetSaoChromaBlockSizes() {
  std::vector<std::pair<int, int>> sizes;
  for (const auto& size : getLumaPUBlockSizes()) {
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
    ::testing::Values(0, 7, 15, 23, 28), ::testing::ValuesIn(getTstArch()));

// Chroma Params: block sizes, band_pos/class, arch
auto kSaoChromaParams = ::testing::Combine(
    ::testing::ValuesIn(GetSaoChromaBlockSizes()),
    ::testing::Values(0, 7, 15, 23, 28), ::testing::ValuesIn(getTstArch()));

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
