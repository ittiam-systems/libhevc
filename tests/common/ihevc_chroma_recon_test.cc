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
#include "tests_common.h"
// clang-format on

namespace {

// Test parameters: trans_size, arch, non_zero_cols, offset (0 for U, 1 for V)
using ChromaReconTestParam = std::tuple<int, IVD_ARCH_T, int, int>;

class ChromaReconTest : public ::testing::TestWithParam<ChromaReconTestParam> {
 protected:
  void SetUp() override {
    std::tie(trans_size, arch, num_non_zero_cols, offset) = GetParam();

    src_strd = trans_size;
    pred_strd = trans_size * 2;
    dst_strd = trans_size * 2;

    pi2_src.resize(trans_size * trans_size);
    pu1_pred.resize(trans_size * trans_size * 2);
    pu1_dst_ref.resize(trans_size * trans_size * 2);
    pu1_dst_tst.resize(trans_size * trans_size * 2);

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  template <typename FuncPtr>
  void RunTest(FuncPtr func_ptr) {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int16_t> coeff_dist(-512, 511);
    std::uniform_int_distribution<uint8_t> pixel_dist(0, 255);

    // Populate pi2_src so that columns [0, num_non_zero_cols) are non-zero.
    std::fill(pi2_src.begin(), pi2_src.end(), 0);
    for (int i = 0; i < trans_size; i++) {
      for (int j = 0; j < trans_size; j++) {
        if (j < num_non_zero_cols) {
          pi2_src[i * src_strd + j] = coeff_dist(rng);
        }
      }
    }

    for (auto& v : pu1_pred) {
      v = pixel_dist(rng);
    }

    WORD32 non_zero_cols_mask = 0;
    for (int j = 0; j < num_non_zero_cols && j < trans_size; j++) {
      non_zero_cols_mask |= (1u << j);
    }

    WORD32 mask = (trans_size == 32)
                      ? 0xFFFFFFFFu
                      : ((static_cast<WORD32>(1u) << trans_size) - 1u);
    WORD32 zero_cols = (~non_zero_cols_mask) & mask;

    std::fill(pu1_dst_ref.begin(), pu1_dst_ref.end(), 0xAA);
    (ref->*func_ptr)(pi2_src.data(), pu1_pred.data() + offset,
                     pu1_dst_ref.data() + offset, src_strd, pred_strd, dst_strd,
                     zero_cols);

    std::fill(pu1_dst_tst.begin(), pu1_dst_tst.end(), 0xAA);
    (tst->*func_ptr)(pi2_src.data(), pu1_pred.data() + offset,
                     pu1_dst_tst.data() + offset, src_strd, pred_strd, dst_strd,
                     zero_cols);

    ASSERT_NO_FATAL_FAILURE(compare_output<UWORD8>(
        pu1_dst_ref, pu1_dst_tst, trans_size * 2, trans_size, dst_strd));
  }

  int trans_size;
  int offset;
  IVD_ARCH_T arch;
  const func_selector_t* ref;
  const func_selector_t* tst;

  WORD32 src_strd;
  WORD32 pred_strd;
  WORD32 dst_strd;
  WORD32 num_non_zero_cols;
  std::vector<WORD16> pi2_src;
  std::vector<UWORD8> pu1_pred;
  std::vector<UWORD8> pu1_dst_ref;
  std::vector<UWORD8> pu1_dst_tst;
};

TEST_P(ChromaReconTest, Run) {
  if (trans_size == 4) {
    RunTest(&func_selector_t::ihevc_chroma_recon_4x4_fptr);
  } else if (trans_size == 8) {
    RunTest(&func_selector_t::ihevc_chroma_recon_8x8_fptr);
  } else if (trans_size == 16) {
    RunTest(&func_selector_t::ihevc_chroma_recon_16x16_fptr);
  } else if (trans_size == 32) {
    RunTest(&func_selector_t::ihevc_chroma_recon_32x32_fptr);
  }
}

std::string PrintChromaReconTestParam(
    const testing::TestParamInfo<ChromaReconTestParam>& info) {
  WORD32 trans_size, non_zero_cols, offset;
  IVD_ARCH_T arch;
  std::tie(trans_size, arch, non_zero_cols, offset) = info.param;
  std::stringstream ss;
  ss << "size_" << trans_size << "_nzc_" << non_zero_cols << "_component_"
     << (offset == 0 ? "U" : "V") << "_" << get_arch_str(arch);
  return ss.str();
}

std::vector<ChromaReconTestParam> GenerateChromaReconTestParams() {
  std::vector<ChromaReconTestParam> params;
  const WORD32 nz_options[] = {1, 2, 4, 8, 16, 32};
  const int offsets[] = {0, 1};

  auto add_params_for_size = [&](int size) {
    for (auto arch : ga_tst_arch) {
      for (WORD32 nnzc : nz_options) {
        if (nnzc > size) continue;
        for (int offset : offsets) {
          params.emplace_back(size, arch, nnzc, offset);
        }
      }
    }
  };

  add_params_for_size(4);
  add_params_for_size(8);
  add_params_for_size(16);
  add_params_for_size(32);

  return params;
}

INSTANTIATE_TEST_SUITE_P(ChromaRecon, ChromaReconTest,
                         ::testing::ValuesIn(GenerateChromaReconTestParams()),
                         PrintChromaReconTestParam);

}  // namespace
