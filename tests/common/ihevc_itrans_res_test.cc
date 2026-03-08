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
#include "ihevc_itrans_res.h"
#include "ihevcd_function_selector.h"
#include "iv.h"
#include "ivd.h"
#include "func_selector.h"
#include "tests_common.h"
// clang-format on

// Test parameters: trans_size, ttype (0: normal, 1: ttype1), arch
using ITransResTestParam = std::tuple<int, int, IVD_ARCH_T>;

class ITransResTest : public ::testing::TestWithParam<ITransResTestParam> {
protected:
  void SetUp() override {
    std::tie(trans_size, ttype, arch) = GetParam();
    src_strd = trans_size;
    dst_strd = trans_size;

    // Input buffer (coefficients)
    src_buf.resize(trans_size * trans_size);

    // Temporary buffer (intermediate 16-bit data)
    // Size needed: width * height * 16 bits = width * height * 2 bytes
    // pi2_tmp is WORD16*, so we need width * height elements
    tmp_buf.resize(trans_size * trans_size);

    // Output buffers
    dst_buf_ref.resize(trans_size * trans_size);
    dst_buf_tst.resize(trans_size * trans_size);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-32768, 32767);

    for (auto &v : src_buf) {
      v = static_cast<WORD16>(dist(rng));
    }

    // Fill dst buffers with pattern
    std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xCDCD);
    std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xCDCD);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  template <typename FuncPtr> void RunTest(FuncPtr func_ptr) {
    if (ref && ref->*func_ptr) {
      (ref->*func_ptr)(src_buf.data(), tmp_buf.data(), dst_buf_ref.data(),
                       src_strd, dst_strd, 0, 0);
    } else {
      GTEST_SKIP() << "Reference function not available";
    }
    if (tst && tst->*func_ptr) {
      (tst->*func_ptr)(src_buf.data(), tmp_buf.data(), dst_buf_tst.data(),
                       src_strd, dst_strd, 0, 0);
    } else {
      GTEST_SKIP() << "Test function not available";
    }

    ASSERT_NO_FATAL_FAILURE(compare_output<WORD16>(
        dst_buf_ref, dst_buf_tst, trans_size, trans_size, dst_strd));
  }

  int trans_size;
  int ttype;
  IVD_ARCH_T arch;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf;
  std::vector<WORD16> tmp_buf;
  std::vector<WORD16> dst_buf_ref;
  std::vector<WORD16> dst_buf_tst;
  const func_selector_t *tst;
  const func_selector_t *ref;
};

TEST_P(ITransResTest, Run) {
  if (trans_size == 4) {
    if (ttype == 1) {
      RunTest(&func_selector_t::ihevc_itrans_res_4x4_ttype1_fptr);
    } else {
      RunTest(&func_selector_t::ihevc_itrans_res_4x4_fptr);
    }
  } else if (trans_size == 8) {
    RunTest(&func_selector_t::ihevc_itrans_res_8x8_fptr);
  } else if (trans_size == 16) {
    RunTest(&func_selector_t::ihevc_itrans_res_16x16_fptr);
  } else if (trans_size == 32) {
    RunTest(&func_selector_t::ihevc_itrans_res_32x32_fptr);
  }
}

std::string PrintITransResTestParam(
    const testing::TestParamInfo<ITransResTestParam> &info) {
  int trans_size, ttype;
  IVD_ARCH_T arch;
  std::tie(trans_size, ttype, arch) = info.param;
  std::stringstream ss;
  ss << "size_" << trans_size << "_ttype_" << ttype << "_"
     << get_arch_str(arch);
  return ss.str();
}

// Instantiate tests
// Size 4: ttype 0 and 1
INSTANTIATE_TEST_SUITE_P(ITransRes4x4, ITransResTest,
                         ::testing::Combine(::testing::Values(4),
                                            ::testing::Values(0, 1),
                                            ::testing::ValuesIn(ga_tst_arch)),
                         PrintITransResTestParam);

// Size 8, 16, 32: ttype 0
INSTANTIATE_TEST_SUITE_P(ITransResOther, ITransResTest,
                         ::testing::Combine(::testing::Values(8, 16, 32),
                                            ::testing::Values(0),
                                            ::testing::ValuesIn(ga_tst_arch)),
                         PrintITransResTestParam);
