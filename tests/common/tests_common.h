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
#ifndef __TESTS_COMMON_H__
#define __TESTS_COMMON_H__
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
#include "ihevcd_function_selector.h"
#include "iv.h"
#include "ivd.h"
// clang-format on

static constexpr int kMaxSize = 64;
static constexpr int kTapSize = 8;
static constexpr int kMaxHeight = kMaxSize + kTapSize;

extern const std::vector<std::pair<int, int>> kPUBlockSizes;
extern const std::vector<UWORD8> g_src8_buf;
extern const std::vector<IVD_ARCH_T> ga_tst_arch;

// Compare outputs
template <typename T>
static void compare_output(const std::vector<T> &ref, const std::vector<T> &test,
                    int wd, int ht, int dst_strd) {
  int size_bytes = wd * sizeof(T);
  for (int i = 0; i < ht; ++i) {
    int cmp = memcmp(ref.data() + i * dst_strd, test.data() + i * dst_strd,
                     size_bytes);
    ASSERT_EQ(0, cmp) << "Mismatch at row " << i << " for size " << wd << "x"
                      << ht;
  }
}

std::string get_arch_str(IVD_ARCH_T arch);
#endif /* __TESTS_COMMON_H__ */