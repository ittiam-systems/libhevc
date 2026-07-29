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
#include "ihevc_chroma_intra_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"
#include "func_selector.h"
#include "TestCommon.h"
// clang-format on

// Test parameters: block_size, mode, dst_stride_mul, arch
using ChromaIntraPredTestParam = std::tuple<int, int, int, IV_ARCH_T>;

class ChromaIntraPredTest
    : public ::testing::TestWithParam<ChromaIntraPredTestParam> {
 protected:
  void SetUp() override {
    std::tie(nt, mode, dst_strd_mul, arch) = GetParam();
    src_strd = 1;  // Intra pred reference is usually dense
    dst_strd = 2 * nt * dst_strd_mul;

    // TODO: Increase allocations for x86/x86_64 to avoid out-of-bounds
    // reads/writes in SIMD implementations.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
    int pad_ref = 16;
    int pad_dst = 16;
#else
    int pad_ref = 0;
    int pad_dst = 0;
#endif

    // Reference buffer size: 2 * (4 * nt + 1) = 8 * nt + 2
    int ref_size = 8 * nt + 2;
    ref_buf.resize(ref_size + pad_ref);

    // Initialize reference buffer with random data
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : ref_buf) {
      v = static_cast<UWORD8>(dist(rng));
    }

    pu1_ref = ref_buf.data();

    dst_buf_ref.resize(dst_strd * nt + pad_dst);
    dst_buf_tst.resize(dst_strd * nt + pad_dst);

    // Initialize dst buffers with pattern to detect over/under writes
    std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xCD);
    std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xCD);

    pu1_dst_ref = dst_buf_ref.data();
    pu1_dst_tst = dst_buf_tst.data();

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  template <typename FuncPtr>
  void RunTest(FuncPtr func_ptr) {
    (ref->*func_ptr)(pu1_ref, src_strd, pu1_dst_ref, dst_strd, nt, mode);
    (tst->*func_ptr)(pu1_ref, src_strd, pu1_dst_tst, dst_strd, nt, mode);
    ASSERT_NO_FATAL_FAILURE(
        compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, 2 * nt, nt, dst_strd));
  }

  int nt, mode, dst_strd_mul;
  int src_strd, dst_strd;
  std::vector<UWORD8> ref_buf;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  UWORD8* pu1_ref;
  UWORD8* pu1_dst_ref;
  UWORD8* pu1_dst_tst;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* tst;
  const ihevc_func_selector_t* ref;
};

TEST_P(ChromaIntraPredTest, Run) {
  if (mode == 0)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_planar_fptr);
  else if (mode == 1)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_dc_fptr);
  else if (mode == 2)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_mode2_fptr);
  else if (mode >= 3 && mode <= 9)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_mode_3_to_9_fptr);
  else if (mode == 10)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_horz_fptr);
  else if (mode >= 11 && mode <= 17)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_mode_11_to_17_fptr);
  else if (mode == 18 || mode == 34)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_mode_18_34_fptr);
  else if (mode >= 19 && mode <= 25)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_mode_19_to_25_fptr);
  else if (mode == 26)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_ver_fptr);
  else if (mode >= 27 && mode <= 33)
    RunTest(&ihevc_func_selector_t::ihevc_intra_pred_chroma_mode_27_to_33_fptr);
  else
    FAIL() << "Invalid mode: " << mode;
}

std::string PrintChromaIntraPredTestParam(
    const testing::TestParamInfo<ChromaIntraPredTestParam>& info) {
  int nt, mode, dst_strd_mul;
  IV_ARCH_T arch;
  std::tie(nt, mode, dst_strd_mul, arch) = info.param;
  std::stringstream ss;
  ss << "nt_" << nt << "_mode_" << mode << "_dst_stride_"
     << 2 * nt * dst_strd_mul << "_" << get_arch_str(arch);
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    ChromaIntraPred, ChromaIntraPredTest,
    ::testing::Combine(::testing::Values(4, 8, 16), ::testing::Range(0, 35),
                       ::testing::Values(1, 2),  // Dst Stride Multiplier
                       ::testing::ValuesIn(ga_tst_arch)),
    PrintChromaIntraPredTestParam);
