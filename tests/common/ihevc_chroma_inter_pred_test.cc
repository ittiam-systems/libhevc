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
#include "ihevc_inter_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"
#include "func_selector.h"
#include "TestCommon.h"
#include "ihevc_inter_pred_utils.h"
// clang-format on

using ChromaInterPredTestParam = InterPredTestParam;

template <typename srcType, typename dstType>
using ChromaInterPredTest = InterPredTestBase<srcType, dstType, 2>;

class ChromaInterPred_8_8_Test : public ChromaInterPredTest<UWORD8, UWORD8> {};
class ChromaInterPred_8_16_Test : public ChromaInterPredTest<UWORD8, WORD16> {};
class ChromaInterPred_16_8_Test : public ChromaInterPredTest<WORD16, UWORD8> {};
class ChromaInterPred_16_16_Test : public ChromaInterPredTest<WORD16, WORD16> {
};

TEST_P(ChromaInterPred_8_8_Test, ChromaCopyTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_copy_fptr);
}

TEST_P(ChromaInterPred_8_8_Test, ChromaHorzTest) {
#if defined(__arm__) || defined(__aarch64__)
  GTEST_SKIP() << "Skipping ChromaHorzTest on ARM";
#endif
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_horz_fptr);
}

TEST_P(ChromaInterPred_8_8_Test, ChromaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_fptr);
}

TEST_P(ChromaInterPred_8_16_Test, ChromaCopyTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_copy_w16out_fptr);
}

TEST_P(ChromaInterPred_8_16_Test, ChromaHorzTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_horz_w16out_fptr);
}

TEST_P(ChromaInterPred_8_16_Test, ChromaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_w16out_fptr);
}

TEST_P(ChromaInterPred_16_8_Test, ChromaVertTest) {
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_w16inp_fptr);
}

TEST_P(ChromaInterPred_16_16_Test, ChromaVertTest) {
#if defined(__arm__) || defined(__aarch64__)
  GTEST_SKIP() << "Skipping ChromaVertTest on ARM";
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
  // TODO: SSE4.2 and SSSE3 are not matching C implementation
  GTEST_SKIP() << "SSE4.2 and SSSE3 are not matching C implementation for "
                  "ihevc_inter_pred_chroma_vert_w16inp_w16out_fptr";
#endif
  RunTest(&ihevc_func_selector_t::ihevc_inter_pred_chroma_vert_w16inp_w16out_fptr);
}

auto kChromaInterPredTestParams = ::testing::Combine(
    ::testing::ValuesIn(getChromaPUBlockSizes()),
    ::testing::Values(1, 2),  // Src Stride Multiplier
    ::testing::Values(1, 2),  // Dst Stride Multiplier
    ::testing::Values(0, 1, 2, 3, 4, 5, 6,
                      7),              // Coeff index (chroma has 8 phases)
    ::testing::ValuesIn(getTstArch())  // arch
);

INSTANTIATE_TEST_SUITE_P(ChromaCopyTest, ChromaInterPred_8_8_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);

INSTANTIATE_TEST_SUITE_P(ChromaHorzTest, ChromaInterPred_8_8_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_8_8_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);

INSTANTIATE_TEST_SUITE_P(ChromaCopyTest, ChromaInterPred_8_16_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);

INSTANTIATE_TEST_SUITE_P(ChromaHorzTest, ChromaInterPred_8_16_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_8_16_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_16_8_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);

INSTANTIATE_TEST_SUITE_P(ChromaVertTest, ChromaInterPred_16_16_Test,
                         kChromaInterPredTestParams,
                         PrintInterPredTestParam<2>);
