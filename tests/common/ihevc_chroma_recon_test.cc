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
#include "ihevc_itrans_utils.h"
// clang-format on

namespace {

// Test parameters: trans_size, arch, non_zero_cols, offset (0 for U, 1 for V)
using ChromaReconTestParam = std::tuple<int, IV_ARCH_T, int, int>;

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

  void RunTest() {
    FillRandomSubBlock(pi2_src.data(), trans_size, src_strd, trans_size,
                       num_non_zero_cols, static_cast<WORD16>(-512),
                       static_cast<WORD16>(511), 0);
    FillRandom(pu1_pred, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 1);

    WORD32 zero_cols = ComputeZeroMask(trans_size, num_non_zero_cols);

    ChromaReconFn ref_fn = GetChromaReconFn(ref, trans_size);
    ChromaReconFn tst_fn = GetChromaReconFn(tst, trans_size);

    std::fill(pu1_dst_ref.begin(), pu1_dst_ref.end(), 0xAA);
    ref_fn(pi2_src.data(), pu1_pred.data() + offset,
           pu1_dst_ref.data() + offset, src_strd, pred_strd, dst_strd,
           zero_cols);

    std::fill(pu1_dst_tst.begin(), pu1_dst_tst.end(), 0xAA);
    tst_fn(pi2_src.data(), pu1_pred.data() + offset,
           pu1_dst_tst.data() + offset, src_strd, pred_strd, dst_strd,
           zero_cols);

    ASSERT_NO_FATAL_FAILURE(compare_output<UWORD8>(
        pu1_dst_ref, pu1_dst_tst, trans_size * 2, trans_size, dst_strd));
  }

  int trans_size;
  int offset;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;

  WORD32 src_strd;
  WORD32 pred_strd;
  WORD32 dst_strd;
  WORD32 num_non_zero_cols;
  std::vector<WORD16> pi2_src;
  std::vector<UWORD8> pu1_pred;
  std::vector<UWORD8> pu1_dst_ref;
  std::vector<UWORD8> pu1_dst_tst;
};

TEST_P(ChromaReconTest, Run) { RunTest(); }

std::string PrintChromaReconTestParam(
    const testing::TestParamInfo<ChromaReconTestParam>& info) {
  WORD32 trans_size, non_zero_cols, offset;
  IV_ARCH_T arch;
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
    for (auto arch : getTstArch()) {
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
