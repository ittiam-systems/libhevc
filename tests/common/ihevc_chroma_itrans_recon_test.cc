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
#include "ihevc_macros.h"
#include "ihevc_typedefs.h"
#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_itrans_utils.h"
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

  void RunTest() {
    FillRandomSubBlock(pi2_src.data(), trans_size, src_strd, num_non_zero_rows,
                       num_non_zero_cols, static_cast<WORD16>(-32768),
                       static_cast<WORD16>(32767), 0);
    FillRandom(pu1_pred, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 1);

    std::fill(pu1_dst_ref.begin(), pu1_dst_ref.end(), 0xAA);
    std::fill(pu1_dst_tst.begin(), pu1_dst_tst.end(), 0xAA);

    WORD32 zero_cols = ComputeZeroMask(trans_size, num_non_zero_cols);
    WORD32 zero_rows = ComputeZeroMask(trans_size, num_non_zero_rows);

    ChromaITransReconFn ref_fn = GetChromaITransReconFn(ref, trans_size);
    ChromaITransReconFn tst_fn = GetChromaITransReconFn(tst, trans_size);

    // 1. Reference path (generic C)
    ref_fn(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst_ref.data(),
           src_strd, pred_strd, dst_strd, zero_cols, zero_rows);

    // 2. Test path (SIMD)
    tst_fn(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst_tst.data(),
           src_strd, pred_strd, dst_strd, zero_cols, zero_rows);

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

TEST_P(ChromaITransReconTest, Run) { RunTest(); }

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
    for (auto arch : getTstArch()) {
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
