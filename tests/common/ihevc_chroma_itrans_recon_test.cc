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
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// clang-format off
#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_structs.h"
#include "ihevc_typedefs.h"
#include "TestCommon.h"
// clang-format on

namespace {

// Test parameters: trans_size, arch, non_zero_rows, non_zero_cols
using ITransReconTestParam = std::tuple<int, IV_ARCH_T, int, int>;

class ChromaITransReconTest
    : public ::testing::TestWithParam<ITransReconTestParam> {
 protected:
  void SetUp() override {
    std::tie(trans_size, arch, num_non_zero_rows, num_non_zero_cols) =
        GetParam();

    src_strd = trans_size;
    pred_strd = 2 * trans_size;
    dst_strd = 2 * trans_size;

    pi2_src.resize(trans_size * trans_size);
    pi2_tmp.resize(trans_size * trans_size);
    pu1_pred.resize(pred_strd * trans_size);
    pu1_dst_ref.resize(dst_strd * trans_size);
    pu1_dst_tst.resize(dst_strd * trans_size);

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  template <typename FuncPtr>
  void RunTest(FuncPtr func_ptr) {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int16_t> coeff_dist_full(-32768, 32767);
    std::uniform_int_distribution<uint8_t> pixel_dist(0, 255);

    std::fill(pi2_src.begin(), pi2_src.end(), 0);
    for (int i = 0; i < trans_size; i++) {
      for (int j = 0; j < trans_size; j++) {
        if (i < num_non_zero_rows && j < num_non_zero_cols) {
          pi2_src[i * src_strd + j] = coeff_dist_full(rng);
        }
      }
    }

    for (auto& v : pu1_pred) {
      v = pixel_dist(rng);
    }

    std::fill(pu1_dst_ref.begin(), pu1_dst_ref.end(), 0xAA);
    std::fill(pu1_dst_tst.begin(), pu1_dst_tst.end(), 0xAA);

    WORD32 non_zero_rows_mask = 0;
    for (int i = 0; i < num_non_zero_rows && i < trans_size; i++) {
      non_zero_rows_mask |= (1u << i);
    }

    WORD32 non_zero_cols_mask = 0;
    for (int j = 0; j < num_non_zero_cols && j < trans_size; j++) {
      non_zero_cols_mask |= (1u << j);
    }

    WORD32 mask = (trans_size == 32)
                      ? 0xFFFFFFFFu
                      : ((static_cast<WORD32>(1u) << trans_size) - 1u);
    WORD32 zero_cols = (~non_zero_cols_mask) & mask;
    WORD32 zero_rows = (~non_zero_rows_mask) & mask;

    // 1. Reference path (generic C)
    (ref->*func_ptr)(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(),
                     pu1_dst_ref.data(), src_strd, pred_strd, dst_strd,
                     zero_cols, zero_rows);

    // 2. Test path (SIMD)
    (tst->*func_ptr)(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(),
                     pu1_dst_tst.data(), src_strd, pred_strd, dst_strd,
                     zero_cols, zero_rows);

    ASSERT_NO_FATAL_FAILURE(compare_output<UWORD8>(
        pu1_dst_ref, pu1_dst_tst, 2 * trans_size, trans_size, dst_strd));
  }

  int trans_size;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;

  WORD32 src_strd;
  WORD32 pred_strd;
  WORD32 dst_strd;
  WORD32 num_non_zero_rows;
  WORD32 num_non_zero_cols;
  std::vector<WORD16> pi2_src;
  std::vector<WORD16> pi2_tmp;
  std::vector<UWORD8> pu1_pred;
  std::vector<UWORD8> pu1_dst_ref;
  std::vector<UWORD8> pu1_dst_tst;
};

TEST_P(ChromaITransReconTest, Run) {
  if (trans_size == 4) {
    RunTest(&ihevc_func_selector_t::ihevc_chroma_itrans_recon_4x4_fptr);
  } else if (trans_size == 8) {
    RunTest(&ihevc_func_selector_t::ihevc_chroma_itrans_recon_8x8_fptr);
  } else if (trans_size == 16) {
    RunTest(&ihevc_func_selector_t::ihevc_chroma_itrans_recon_16x16_fptr);
  } else if (trans_size == 32) {
    RunTest(&ihevc_func_selector_t::ihevc_chroma_itrans_recon_32x32_fptr);
  }
}

std::string PrintChromaITransReconTestParam(
    const testing::TestParamInfo<ITransReconTestParam>& info) {
  WORD32 trans_size, non_zero_rows, non_zero_cols;
  IV_ARCH_T arch;
  std::tie(trans_size, arch, non_zero_rows, non_zero_cols) = info.param;
  std::stringstream ss;
  ss << "size_" << trans_size << "_nzr_" << non_zero_rows << "_nzc_"
     << non_zero_cols << "_" << get_arch_str(arch);
  return ss.str();
}

std::vector<ITransReconTestParam> GenerateChromaITransReconTestParams() {
  std::vector<ITransReconTestParam> params;
  const WORD32 nz_options[] = {1, 2, 4, 8, 16, 32};
  const int sizes[] = {4, 8, 16, 32};

  for (int size : sizes) {
    for (auto arch : ga_tst_arch) {
      for (int nnzr : nz_options) {
        if (nnzr > size) continue;
        for (int nnzc : nz_options) {
          if (nnzc > size) continue;
          params.emplace_back(size, arch, nnzr, nnzc);
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(
    ChromaITransRecon, ChromaITransReconTest,
    ::testing::ValuesIn(GenerateChromaITransReconTestParams()),
    PrintChromaITransReconTestParam);

}  // namespace
