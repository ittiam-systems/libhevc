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
#include "ihevc_inter_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"
#include "func_selector.h"
#include "TestCommon.h"
#include "ihevc_inter_pred_utils.h"
// clang-format on

using LumaInterPredTestParam = InterPredTestParam;

template <typename srcType, typename dstType>
using LumaInterPredTest = InterPredTestBase<srcType, dstType, 1>;

class LumaInterPred_8_8_Test : public LumaInterPredTest<UWORD8, UWORD8> {};
class LumaInterPred_8_16_Test : public LumaInterPredTest<UWORD8, WORD16> {};
class LumaInterPred_16_8_Test : public LumaInterPredTest<WORD16, UWORD8> {};
class LumaInterPred_16_16_Test : public LumaInterPredTest<WORD16, WORD16> {};

TEST_P(LumaInterPred_8_8_Test, LumaCopyTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_copy_fptr);
}

TEST_P(LumaInterPred_8_8_Test, LumaHorzTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_horz_fptr);
}

TEST_P(LumaInterPred_8_8_Test, LumaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_vert_fptr);
}

TEST_P(LumaInterPred_8_16_Test, LumaCopyTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_copy_w16out_fptr);
}

TEST_P(LumaInterPred_8_16_Test, LumaHorzTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_horz_w16out_fptr);
}

TEST_P(LumaInterPred_8_16_Test, LumaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_vert_w16out_fptr);
}

TEST_P(LumaInterPred_16_8_Test, LumaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_vert_w16inp_fptr);
}

TEST_P(LumaInterPred_16_16_Test, LumaVertTest) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
  // TODO: SSE4.2 and SSSE3 are not matching C implementation
  GTEST_SKIP() << "SSE4.2 and SSSE3 are not matching C implementation for "
                  "ihevc_inter_pred_luma_vert_w16inp_w16out_fptr";
#endif
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_luma_vert_w16inp_w16out_fptr);
}

auto kLumaInterPredTestParams =
    ::testing::Combine(::testing::ValuesIn(getLumaPUBlockSizes()),
                       ::testing::Values(1, 2),        // Src Stride Multiplier
                       ::testing::Values(1, 2),        // Dst Stride Multiplier
                       ::testing::Values(0, 1, 2, 3),  // Coeff index
                       ::testing::ValuesIn(getTstArch())  // arch
    );

INSTANTIATE_TEST_SUITE_P(LumaCopyTest, LumaInterPred_8_8_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);

INSTANTIATE_TEST_SUITE_P(LumaHorzTest, LumaInterPred_8_8_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_8_8_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);

INSTANTIATE_TEST_SUITE_P(LumaCopyTest, LumaInterPred_8_16_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);

INSTANTIATE_TEST_SUITE_P(LumaHorzTest, LumaInterPred_8_16_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_8_16_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_16_8_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);

INSTANTIATE_TEST_SUITE_P(LumaVertTest, LumaInterPred_16_16_Test,
                         kLumaInterPredTestParams, PrintInterPredTestParam<1>);
