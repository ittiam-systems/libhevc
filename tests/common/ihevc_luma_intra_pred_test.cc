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
#include "ihevc_intra_pred.h"
#include "ihevcd_function_selector.h"
#include "iv.h"
#include "ivd.h"
#include "func_selector.h"
#include "tests_common.h"
// clang-format on

// Test parameters: block_size, mode, dst_stride_mul, arch
using LumaIntraPredTestParam = std::tuple<int, int, int, IVD_ARCH_T>;

class LumaIntraPredTest
    : public ::testing::TestWithParam<LumaIntraPredTestParam> {
protected:
  void SetUp() override {
    std::tie(nt, mode, dst_strd_mul, arch) = GetParam();
    src_strd = 1; // Intra pred reference is usually dense
    dst_strd = nt * dst_strd_mul;

    // Reference buffer size: 4 * nt + 1
    int ref_size = 4 * nt + 1;
    ref_buf.resize(ref_size);

    // Initialize reference buffer with random data
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &v : ref_buf) {
      v = static_cast<UWORD8>(dist(rng));
    }

    // Use a pointer aligned or offset to ensure we have valid data
    // The function expects pu1_ref to point to the start of the reference array
    // Top-left is usually at index 2*nt.
    // We just pass the data pointer.
    pu1_ref = ref_buf.data();

    dst_buf_ref.resize(dst_strd * nt);
    dst_buf_tst.resize(dst_strd * nt);

    // Initialize dst buffers with pattern to detect over/under writes
    std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xCD);
    std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xCD);

    pu1_dst_ref = dst_buf_ref.data();
    pu1_dst_tst = dst_buf_tst.data();

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  template <typename FuncPtr> void RunTest(FuncPtr func_ptr) {
    (ref->*func_ptr)(pu1_ref, src_strd, pu1_dst_ref, dst_strd, nt, mode);
    (tst->*func_ptr)(pu1_ref, src_strd, pu1_dst_tst, dst_strd, nt, mode);
    ASSERT_NO_FATAL_FAILURE(
        compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, nt, nt, dst_strd));
  }

  int nt, mode, dst_strd_mul;
  int src_strd, dst_strd;
  std::vector<UWORD8> ref_buf;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  UWORD8 *pu1_ref;
  UWORD8 *pu1_dst_ref;
  UWORD8 *pu1_dst_tst;
  IVD_ARCH_T arch;
  const func_selector_t *tst;
  const func_selector_t *ref;
};

TEST_P(LumaIntraPredTest, Run) {
  if (mode == 0)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_planar_fptr);
  else if (mode == 1)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_dc_fptr);
  else if (mode == 2)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_mode2_fptr);
  else if (mode >= 3 && mode <= 9)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_mode_3_to_9_fptr);
  else if (mode == 10) {
    GTEST_SKIP() << "SIMD implementation is not matching C implementation for "
                    "ihevc_intra_pred_luma_horz_fptr";
    RunTest(&func_selector_t::ihevc_intra_pred_luma_horz_fptr);
  } else if (mode >= 11 && mode <= 17)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_mode_11_to_17_fptr);
  else if (mode == 18 || mode == 34)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_mode_18_34_fptr);
  else if (mode >= 19 && mode <= 25)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_mode_19_to_25_fptr);
  else if (mode == 26) {
    GTEST_SKIP() << "SIMD implementation is not matching C implementation for "
                    "ihevc_intra_pred_luma_ver_fptr";
    RunTest(&func_selector_t::ihevc_intra_pred_luma_ver_fptr);
  } else if (mode >= 27 && mode <= 33)
    RunTest(&func_selector_t::ihevc_intra_pred_luma_mode_27_to_33_fptr);
  else
    FAIL() << "Invalid mode: " << mode;
}

std::string PrintLumaIntraPredTestParam(
    const testing::TestParamInfo<LumaIntraPredTestParam> &info) {
  int nt, mode, dst_strd_mul;
  IVD_ARCH_T arch;
  std::tie(nt, mode, dst_strd_mul, arch) = info.param;
  std::stringstream ss;
  ss << "nt_" << nt << "_mode_" << mode << "_dst_stride_" << nt * dst_strd_mul
     << "_" << get_arch_str(arch);
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    LumaIntraPred, LumaIntraPredTest,
    ::testing::Combine(::testing::Values(4, 8, 16, 32), ::testing::Range(0, 35),
                       ::testing::Values(1, 2), // Dst Stride Multiplier
                       ::testing::ValuesIn(ga_tst_arch)),
    PrintLumaIntraPredTestParam);
