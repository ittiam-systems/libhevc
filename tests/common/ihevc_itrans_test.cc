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
#include "ihevc_macros.h"
#include "ihevc_typedefs.h"
#include "TestCommon.h"
// clang-format on

namespace {

// Test parameters: trans_size, ttype (0: normal, 1: ttype1), shift,
// non_zero_cols, arch
using ITransTestParam = std::tuple<int, int, int, int, IV_ARCH_T>;

class ITransTest : public ::testing::TestWithParam<ITransTestParam> {
 protected:
  void SetUp() override {
    std::tie(trans_size, ttype, shift, num_non_zero_cols, arch) = GetParam();

    src_strd = trans_size;
    dst_strd = trans_size;

    pi2_src.resize(trans_size * trans_size);
    pi2_dst_ref.resize(trans_size * trans_size);
    pi2_dst_tst.resize(trans_size * trans_size);

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void RunTest() {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int16_t> coeff_dist(-32768, 32767);

    std::fill(pi2_src.begin(), pi2_src.end(), 0);
    for (int i = 0; i < trans_size; i++) {
      for (int j = 0; j < trans_size; j++) {
        if (j < num_non_zero_cols) {
          pi2_src[i * src_strd + j] = coeff_dist(rng);
        }
      }
    }

    WORD32 non_zero_cols_mask = 0;
    for (int j = 0; j < num_non_zero_cols; j++) {
      non_zero_cols_mask |= (1u << j);
    }

    WORD32 mask = (trans_size == 32)
                      ? 0xFFFFFFFFu
                      : ((static_cast<WORD32>(1u) << trans_size) - 1u);
    WORD32 zero_cols = (~non_zero_cols_mask) & mask;

    // 1. Reference path from selector (generic C)
    if (trans_size == 4) {
      if (ttype == 1) {
        (ref->*(&ihevc_func_selector_t::ihevc_itrans_4x4_ttype1_fptr))(
            pi2_src.data(), pi2_dst_ref.data(), src_strd, dst_strd, shift,
            zero_cols);
      } else {
        (ref->*(&ihevc_func_selector_t::ihevc_itrans_4x4_fptr))(
            pi2_src.data(), pi2_dst_ref.data(), src_strd, dst_strd, shift,
            zero_cols);
      }
    } else if (trans_size == 8) {
      (ref->*(&ihevc_func_selector_t::ihevc_itrans_8x8_fptr))(
          pi2_src.data(), pi2_dst_ref.data(), src_strd, dst_strd, shift,
          zero_cols);
    } else if (trans_size == 16) {
      (ref->*(&ihevc_func_selector_t::ihevc_itrans_16x16_fptr))(
          pi2_src.data(), pi2_dst_ref.data(), src_strd, dst_strd, shift,
          zero_cols);
    } else if (trans_size == 32) {
      (ref->*(&ihevc_func_selector_t::ihevc_itrans_32x32_fptr))(
          pi2_src.data(), pi2_dst_ref.data(), src_strd, dst_strd, shift,
          zero_cols);
    }

    // 2. Test path from selector (SIMD, which might fall back to C)
    if (trans_size == 4) {
      if (ttype == 1) {
        (tst->*(&ihevc_func_selector_t::ihevc_itrans_4x4_ttype1_fptr))(
            pi2_src.data(), pi2_dst_tst.data(), src_strd, dst_strd, shift,
            zero_cols);
      } else {
        (tst->*(&ihevc_func_selector_t::ihevc_itrans_4x4_fptr))(
            pi2_src.data(), pi2_dst_tst.data(), src_strd, dst_strd, shift,
            zero_cols);
      }
    } else if (trans_size == 8) {
      (tst->*(&ihevc_func_selector_t::ihevc_itrans_8x8_fptr))(
          pi2_src.data(), pi2_dst_tst.data(), src_strd, dst_strd, shift,
          zero_cols);
    } else if (trans_size == 16) {
      (tst->*(&ihevc_func_selector_t::ihevc_itrans_16x16_fptr))(
          pi2_src.data(), pi2_dst_tst.data(), src_strd, dst_strd, shift,
          zero_cols);
    } else if (trans_size == 32) {
      (tst->*(&ihevc_func_selector_t::ihevc_itrans_32x32_fptr))(
          pi2_src.data(), pi2_dst_tst.data(), src_strd, dst_strd, shift,
          zero_cols);
    }

    ASSERT_NO_FATAL_FAILURE(compare_output<WORD16>(
        pi2_dst_ref, pi2_dst_tst, trans_size, trans_size, dst_strd));
  }

  int trans_size;
  int ttype;
  int shift;
  int num_non_zero_cols;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;

  WORD32 src_strd;
  WORD32 dst_strd;
  std::vector<WORD16> pi2_src;
  std::vector<WORD16> pi2_dst_ref;
  std::vector<WORD16> pi2_dst_tst;
};

TEST_P(ITransTest, Run) { RunTest(); }

std::string PrintITransTestParam(
    const testing::TestParamInfo<ITransTestParam>& info) {
  int trans_size, ttype, shift, non_zero_cols;
  IV_ARCH_T arch;
  std::tie(trans_size, ttype, shift, non_zero_cols, arch) = info.param;
  std::stringstream ss;
  ss << "size_" << trans_size << "_ttype_" << ttype << "_shift_" << shift
     << "_nzc_" << non_zero_cols << "_" << get_arch_str(arch);
  return ss.str();
}

std::vector<ITransTestParam> GenerateITransTestParams() {
  std::vector<ITransTestParam> params;
  const int shift_options[] = {7, 12};
  const int nz_options[] = {1, 2, 4, 8, 16, 32};
  const int sizes[] = {4, 8, 16, 32};

  for (int size : sizes) {
    const int ttypes[] = {0, 1};
    int num_ttypes = (size == 4) ? 2 : 1;
    for (int t = 0; t < num_ttypes; t++) {
      int ttype = ttypes[t];
      for (int shift : shift_options) {
        for (auto arch : getTstArch()) {
          for (int nzc : nz_options) {
            if (nzc > size) continue;
            params.emplace_back(size, ttype, shift, nzc, arch);
          }
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(ITrans, ITransTest,
                         ::testing::ValuesIn(GenerateITransTestParams()),
                         PrintITransTestParam);

}  // namespace
