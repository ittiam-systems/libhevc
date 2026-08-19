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

#include "TestCommon.h"
// clang-format on

const std::vector<std::pair<int, int>>& getLumaPUBlockSizes() {
  static const std::vector<std::pair<int, int>> kPUBlockSizes = {
      // clang-format off
      {4, 4},
      {8, 8}, {8, 4}, {4, 8},
      {16, 16}, {16, 12}, {16, 8}, {16, 4}, {12, 16}, {8, 16}, {4, 16},
      {32, 32}, {32, 24}, {32, 16}, {32, 8}, {24, 32}, {16, 32}, {8, 32},
      {64, 64}, {64, 48}, {64, 32}, {64, 16}, {48, 64}, {32, 64}, {16, 64},
      // clang-format on
  };
  return kPUBlockSizes;
}

const std::vector<std::pair<int, int>>& getChromaPUBlockSizes() {
  static const std::vector<std::pair<int, int>> kChromaPUBlockSizes = []() {
    std::vector<std::pair<int, int>> ret;
    for (const auto& size : getLumaPUBlockSizes()) {
      ret.push_back({size.first / 2, size.second / 2});
    }
    return ret;
  }();
  return kChromaPUBlockSizes;
}

const std::vector<UWORD8>& getSrc8Buf() {
  static const std::vector<UWORD8> g_src8_buf = []() {
    std::vector<UWORD8> buf(kMaxSize * kMaxHeight * 4);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : buf) v = static_cast<UWORD8>(dist(rng));
    return buf;
  }();
  return g_src8_buf;
}

std::string get_arch_str(IV_ARCH_T arch) {
  std::string arch_str;
  switch (arch) {
    case ARCH_X86_GENERIC:
      arch_str = "GENERIC";
      break;
    case ARCH_X86_SSSE3:
      arch_str = "SSSE3";
      break;
    case ARCH_X86_SSE42:
      arch_str = "SSE42";
      break;
    case ARCH_X86_AVX2:
      arch_str = "AVX2";
      break;
    case ARCH_ARMV8_GENERIC:
      arch_str = "ARMV8";
      break;
    case ARCH_ARM_A9Q:
      arch_str = "A9Q";
      break;
    default:
      arch_str = "UNKNOWN";
      break;
  }
  return arch_str;
}

const std::vector<IV_ARCH_T>& getTstArch() {
  static const std::vector<IV_ARCH_T> ga_tst_arch = {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
      ARCH_X86_SSSE3,
      ARCH_X86_SSE42,
#ifndef DISABLE_AVX2
      ARCH_X86_AVX2,
#endif  // DISABLE_AVX2
#elif defined(__aarch64__)
      ARCH_ARMV8_GENERIC,
#elif defined(__arm__)
      ARCH_ARM_A9Q,
#endif
  };
  return ga_tst_arch;
}