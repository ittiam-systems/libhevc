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
#include "ihevc_function_selector.h"
#include "iv.h"
#include "func_selector.h"
#include "TestCommon.h"
#include "ihevc_itrans_utils.h"
// clang-format on

// Test parameters: trans_size, ttype (0: normal, 1: ttype1), arch
using ITransResTestParam = std::tuple<int, int, IV_ARCH_T>;

class ITransResTest : public ::testing::TestWithParam<ITransResTestParam> {
 protected:
  void SetUp() override {
    std::tie(trans_size, ttype, arch) = GetParam();
    src_strd = trans_size;
    dst_strd = trans_size;

    // Input buffer (coefficients)
    src_buf.resize(trans_size * trans_size);

    // Temporary buffer (intermediate 16-bit data)
    tmp_buf.resize(trans_size * trans_size);

    // Output buffers
    dst_buf_ref.resize(trans_size * trans_size);
    dst_buf_tst.resize(trans_size * trans_size);

    FillRandom(src_buf, static_cast<WORD16>(-32768), static_cast<WORD16>(32767),
               12345);

    // Fill dst buffers with pattern
    std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xCDCD);
    std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xCDCD);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  void RunTest(UWORD8 bit_depth) {
    ITransResFn ref_fn = GetITransResFn(ref, trans_size, ttype);
    ITransResFn tst_fn = GetITransResFn(tst, trans_size, ttype);

    ref_fn(src_buf.data(), tmp_buf.data(), dst_buf_ref.data(), src_strd,
           dst_strd, 0, 0, bit_depth);
    tst_fn(src_buf.data(), tmp_buf.data(), dst_buf_tst.data(), src_strd,
           dst_strd, 0, 0, bit_depth);

    ASSERT_NO_FATAL_FAILURE(compare_output<WORD16>(
        dst_buf_ref, dst_buf_tst, trans_size, trans_size, dst_strd));
  }

  int trans_size;
  int ttype;
  IV_ARCH_T arch;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf;
  std::vector<WORD16> tmp_buf;
  std::vector<WORD16> dst_buf_ref;
  std::vector<WORD16> dst_buf_tst;
  const ihevc_func_selector_t* tst;
  const ihevc_func_selector_t* ref;
};

TEST_P(ITransResTest, Run) {
  for (UWORD8 bit_depth : {8, 10}) {
    RunTest(bit_depth);
  }
}

std::string PrintITransResTestParam(
    const testing::TestParamInfo<ITransResTestParam> &info) {
  int trans_size, ttype;
  IV_ARCH_T arch;
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
                                            ::testing::ValuesIn(getTstArch())),
                         PrintITransResTestParam);

// Size 8, 16, 32: ttype 0
INSTANTIATE_TEST_SUITE_P(ITransResOther, ITransResTest,
                         ::testing::Combine(::testing::Values(8, 16, 32),
                                            ::testing::Values(0),
                                            ::testing::ValuesIn(getTstArch())),
                         PrintITransResTestParam);
