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

#include "ihevc_intra_pred_utils.h"

// Test parameters: block_size, mode, dst_stride_mul, arch
using ChromaIntraPredTestParam = std::tuple<int, int, int, IV_ARCH_T>;

class ChromaIntraPredTest
    : public ::testing::TestWithParam<ChromaIntraPredTestParam> {
 protected:
  void SetUp() override {
    std::tie(nt, mode, dst_strd_mul, arch) = GetParam();
    buf_ref = CreateIntraPredChromaBuffers<UWORD8>(nt, dst_strd_mul, 8, 12345);
    buf_tst = buf_ref;

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  int nt, mode, dst_strd_mul;
  IntraPredChromaBuffers<UWORD8> buf_ref;
  IntraPredChromaBuffers<UWORD8> buf_tst;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* tst;
  const ihevc_func_selector_t* ref;
};

TEST_P(ChromaIntraPredTest, Run) {
  ChromaIntraPredFn ref_fn = GetChromaIntraPredFn(ref, mode);
  ChromaIntraPredFn tst_fn = GetChromaIntraPredFn(tst, mode);
  ASSERT_NE(ref_fn, nullptr);
  ASSERT_NE(tst_fn, nullptr);

  ref_fn(buf_ref.ref.data(), buf_ref.src_strd, buf_ref.dst.data(),
         buf_ref.dst_strd, nt, mode);
  tst_fn(buf_tst.ref.data(), buf_tst.src_strd, buf_tst.dst.data(),
         buf_tst.dst_strd, nt, mode);
  ASSERT_NO_FATAL_FAILURE(compare_output<UWORD8>(buf_ref.dst, buf_tst.dst,
                                                 2 * nt, nt, buf_ref.dst_strd));
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
    ::testing::Combine(::testing::Values(4, 8, 16, 32), ::testing::Range(0, 35),
                       ::testing::Values(1, 2),  // Dst Stride Multiplier
                       ::testing::ValuesIn(getTstArch())),
    PrintChromaIntraPredTestParam);
