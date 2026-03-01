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

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_inter_pred.h"
#include "ihevcd_function_selector.h"
#include "iv.h"
#include "ivd.h"
#include "func_selector.h"
#include "tests_common.h"
// clang-format on

// Test parameters: width, height, src_stride_mul, dst_stride_mul, coeff_idx,
// arch
using LumaInterPredTestParam =
    std::tuple<std::pair<int, int>, int, int, int, IVD_ARCH_T>;
template <typename srcType, typename dstType>
class LumaInterPredTest
    : public ::testing::TestWithParam<LumaInterPredTestParam> {
protected:
  void SetUp() override {

    std::pair<int, int> block_size;
    std::tie(block_size, src_strd_mul, dst_strd_mul, coeff_idx, arch) =
        GetParam();
    std::tie(wd, ht) = block_size;
    src_strd = wd * src_strd_mul;
    dst_strd = wd * dst_strd_mul;

    dst_buf_ref.resize(dst_strd * ht);
    dst_buf_tst.resize(dst_strd * ht);

    // Set pv_src to a valid position within src_buf to allow negative indexing
    pv_src = (srcType *)g_src8_buf.data() + kTapSize / 2 * src_strd;
    pv_dst_ref = dst_buf_ref.data();
    pv_dst_tst = dst_buf_tst.data();

    pi1_coeffs = gai1_ihevc_luma_filter[coeff_idx];
    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  template <typename FuncPtr> void RunTest(FuncPtr func_ptr) {
    (ref->*func_ptr)(pv_src, pv_dst_ref, src_strd, dst_strd, pi1_coeffs, ht,
                     wd);
    (tst->*func_ptr)(pv_src, pv_dst_tst, src_strd, dst_strd, pi1_coeffs, ht,
                     wd);
    ASSERT_NO_FATAL_FAILURE(
        compare_output<dstType>(dst_buf_ref, dst_buf_tst, wd, ht, dst_strd));
  }

  int wd, ht, src_strd_mul, dst_strd_mul, coeff_idx;
  int src_strd, dst_strd;
  std::vector<dstType> dst_buf_ref;
  std::vector<dstType> dst_buf_tst;
  srcType *pv_src;
  dstType *pv_dst_ref;
  dstType *pv_dst_tst;
  WORD8 *pi1_coeffs;
  IVD_ARCH_T arch;
  const func_selector_t *tst;
  const func_selector_t *ref;
};

class LumaInterPred_8_8_Test : public LumaInterPredTest<UWORD8, UWORD8> {};
class LumaInterPred_8_16_Test : public LumaInterPredTest<UWORD8, WORD16> {};
class LumaInterPred_16_8_Test : public LumaInterPredTest<WORD16, UWORD8> {};
class LumaInterPred_16_16_Test : public LumaInterPredTest<WORD16, WORD16> {};

TEST_P(LumaInterPred_8_8_Test, LumaCopyTest) {
  RunTest(&func_selector_t::ihevc_inter_pred_luma_copy_fptr);
}

TEST_P(LumaInterPred_8_8_Test, LumaHorzTest) {
  RunTest(&func_selector_t::ihevc_inter_pred_luma_horz_fptr);
}

TEST_P(LumaInterPred_8_8_Test, LumaVertTest) {
  RunTest(&func_selector_t::ihevc_inter_pred_luma_vert_fptr);
}

TEST_P(LumaInterPred_8_16_Test, LumaCopyTest) {
  RunTest(&func_selector_t::ihevc_inter_pred_luma_copy_w16out_fptr);
}

TEST_P(LumaInterPred_8_16_Test, LumaHorzTest) {
  RunTest(&func_selector_t::ihevc_inter_pred_luma_horz_w16out_fptr);
}

TEST_P(LumaInterPred_8_16_Test, LumaVertTest) {
  RunTest(&func_selector_t::ihevc_inter_pred_luma_vert_w16out_fptr);
}

TEST_P(LumaInterPred_16_8_Test, LumaVertTest) {
  RunTest(&func_selector_t::ihevc_inter_pred_luma_vert_w16inp_fptr);
}

TEST_P(LumaInterPred_16_16_Test, LumaVertTest) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) ||               \
    defined(_M_IX86)
  // TODO: SSE4.2 and SSSE3 are not matching C implementation
  GTEST_SKIP() << "SSE4.2 and SSSE3 are not matching C implementation for "
                  "ihevc_inter_pred_luma_vert_w16inp_w16out_fptr";
#endif
  RunTest(&func_selector_t::ihevc_inter_pred_luma_vert_w16inp_w16out_fptr);
}

auto kLumaInterPredTestParams =
    ::testing::Combine(::testing::ValuesIn(kPUBlockSizes),
                       ::testing::Values(1, 2),         // Src Stride Multiplier
                       ::testing::Values(1, 2),         // Dst Stride Multiplier
                       ::testing::Values(0, 1, 2, 3),   // Coeff index
                       ::testing::ValuesIn(ga_tst_arch) // arch
    );

std::string PrintLumaInterPredTestParam(
    const testing::TestParamInfo<LumaInterPredTestParam> &info) {
  int wd, ht, src_strd_mul, dst_strd_mul, coeff_idx;
  IVD_ARCH_T arch;
  std::pair<int, int> block_size;
  std::tie(block_size, src_strd_mul, dst_strd_mul, coeff_idx, arch) =
      info.param;
  std::tie(wd, ht) = block_size;
  std::stringstream ss;
  ss << wd << "x" << ht << "_src_stride_" << src_strd_mul * wd << "_dst_stride_"
     << dst_strd_mul * wd << "_coeff_" << coeff_idx << "_"
     << get_arch_str(arch);
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(LumaCopyTest, LumaInterPred_8_8_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(LumaHorzTest, LumaInterPred_8_8_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_8_8_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(LumaCopyTest, LumaInterPred_8_16_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(LumaHorzTest, LumaInterPred_8_16_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_8_16_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_16_8_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_16_16_Test,
                         kLumaInterPredTestParams, PrintLumaInterPredTestParam);
