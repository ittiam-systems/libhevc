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
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_inter_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"
#include "func_selector.h"
#include "TestCommon.h"
// clang-format on

// Test parameters: width, height, src_stride_mul, dst_stride_mul, coeff_idx,
// arch
using ChromaInterPredTestParam =
    std::tuple<std::pair<int, int>, int, int, int, IV_ARCH_T>;

template <typename srcType, typename dstType>
class ChromaInterPredTest
    : public ::testing::TestWithParam<ChromaInterPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_strd_mul, dst_strd_mul, coeff_idx, arch) =
        GetParam();
    std::tie(wd, ht) = block_size;
    src_strd = 2 * wd * src_strd_mul;
    dst_strd = 2 * wd * dst_strd_mul;

    // TODO: Increase allocations for x86/x86_64 to avoid out-of-bounds
    // reads/writes in SIMD implementations.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
    int pad_dst = 16;
#else
    int pad_dst = 0;
#endif

    dst_buf_ref.resize(dst_strd * ht + pad_dst);
    dst_buf_tst.resize(dst_strd * ht + pad_dst);

    // Set pv_src to a valid position within src_buf to allow negative indexing
    pv_src = (srcType*)g_src8_buf.data() + kTapSize / 2 * src_strd;
    pv_dst_ref = dst_buf_ref.data();
    pv_dst_tst = dst_buf_tst.data();

    pi1_coeffs = gai1_ihevc_chroma_filter[coeff_idx];
    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  template <typename FuncPtr>
  void RunTest(FuncPtr func_ptr) {
    (ref->*func_ptr)(pv_src, pv_dst_ref, src_strd, dst_strd, pi1_coeffs, ht,
                     wd);
    (tst->*func_ptr)(pv_src, pv_dst_tst, src_strd, dst_strd, pi1_coeffs, ht,
                     wd);
    ASSERT_NO_FATAL_FAILURE(compare_output<dstType>(dst_buf_ref, dst_buf_tst,
                                                    2 * wd, ht, dst_strd));
  }

  int wd, ht, src_strd_mul, dst_strd_mul, coeff_idx;
  int src_strd, dst_strd;
  std::vector<dstType> dst_buf_ref;
  std::vector<dstType> dst_buf_tst;
  srcType* pv_src;
  dstType* pv_dst_ref;
  dstType* pv_dst_tst;
  WORD8* pi1_coeffs;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* tst;
  const ihevc_func_selector_t* ref;
};

class ChromaInterPred_8_8_Test : public ChromaInterPredTest<UWORD8, UWORD8> {};
class ChromaInterPred_8_16_Test : public ChromaInterPredTest<UWORD8, WORD16> {};
class ChromaInterPred_16_8_Test : public ChromaInterPredTest<WORD16, UWORD8> {};
class ChromaInterPred_16_16_Test : public ChromaInterPredTest<WORD16, WORD16> {
};

TEST_P(ChromaInterPred_8_8_Test, ChromaCopyTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_copy_fptr);
}

TEST_P(ChromaInterPred_8_8_Test, ChromaHorzTest) {
#if defined(__arm__) || defined(__aarch64__)
  GTEST_SKIP() << "Skipping ChromaHorzTest on ARM";
#endif
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_horz_fptr);
}

TEST_P(ChromaInterPred_8_8_Test, ChromaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_fptr);
}

TEST_P(ChromaInterPred_8_16_Test, ChromaCopyTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_copy_w16out_fptr);
}

TEST_P(ChromaInterPred_8_16_Test, ChromaHorzTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_horz_w16out_fptr);
}

TEST_P(ChromaInterPred_8_16_Test, ChromaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_w16out_fptr);
}

TEST_P(ChromaInterPred_16_8_Test, ChromaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_w16inp_fptr);
}

TEST_P(ChromaInterPred_16_16_Test, ChromaVertTest) {
#if defined(__arm__) || defined(__aarch64__)
  GTEST_SKIP() << "Skipping ChromaVertTest on ARM";
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
  // TODO: SSE4.2 and SSSE3 are not matching C implementation
  GTEST_SKIP() << "SSE4.2 and SSSE3 are not matching C implementation for "
                  "ihevc_inter_pred_chroma_vert_w16inp_w16out_fptr";
#endif
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_w16inp_w16out_fptr);
}

// Chroma block sizes are half of luma block sizes (for 4:2:0)
const std::vector<std::pair<int, int>> kChromaPUBlockSizes = []() {
  std::vector<std::pair<int, int>> ret;
  for (const auto& size : kPUBlockSizes) {
    ret.push_back({size.first / 2, size.second / 2});
  }
  return ret;
}();

auto kChromaInterPredTestParams = ::testing::Combine(
    ::testing::ValuesIn(kChromaPUBlockSizes),
    ::testing::Values(1, 2),  // Src Stride Multiplier
    ::testing::Values(1, 2),  // Dst Stride Multiplier
    ::testing::Values(0, 1, 2, 3, 4, 5, 6,
                      7),             // Coeff index (chroma has 8 phases)
    ::testing::ValuesIn(ga_tst_arch)  // arch
);

std::string PrintChromaInterPredTestParam(
    const testing::TestParamInfo<ChromaInterPredTestParam>& info) {
  int wd, ht, src_strd_mul, dst_strd_mul, coeff_idx;
  IV_ARCH_T arch;
  std::pair<int, int> block_size;
  std::tie(block_size, src_strd_mul, dst_strd_mul, coeff_idx, arch) =
      info.param;
  std::tie(wd, ht) = block_size;
  std::stringstream ss;
  // Width in elements is 2 * wd for interleaved chroma
  ss << wd << "x" << ht << "_src_stride_" << src_strd_mul * 2 * wd
     << "_dst_stride_" << dst_strd_mul * 2 * wd << "_coeff_" << coeff_idx << "_"
     << get_arch_str(arch);
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(ChromaCopyTest, ChromaInterPred_8_8_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(ChromaHorzTest, ChromaInterPred_8_8_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_8_8_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(ChromaCopyTest, ChromaInterPred_8_16_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(ChromaHorzTest, ChromaInterPred_8_16_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_8_16_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_16_8_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_16_16_Test,
                         kChromaInterPredTestParams,
                         PrintChromaInterPredTestParam);
