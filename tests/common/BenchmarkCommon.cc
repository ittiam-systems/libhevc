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

#include "BenchmarkCommon.h"

#include <benchmark/benchmark.h>

const std::vector<IV_ARCH_T>& GetBenchmarkArchitectures() {
  static const std::vector<IV_ARCH_T> kArches = []() {
    std::vector<IV_ARCH_T> arches = {ARCH_NA};
    for (auto a : getTstArch()) {
      arches.push_back(a);
    }
    return arches;
  }();
  return kArches;
}

const std::vector<std::pair<int, int>>& GetBenchmarkLumaPUSizes() {
  static const std::vector<std::pair<int, int>> kSizes = {
      {8, 8}, {16, 16}, {32, 32}, {64, 64}, {16, 8}, {32, 16}, {64, 32},
  };
  return kSizes;
}

const std::vector<std::pair<int, int>>& GetBenchmarkChromaPUSizes() {
  static const std::vector<std::pair<int, int>> kSizes = {
      {4, 4}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 8}, {32, 16},
  };
  return kSizes;
}

int main(int argc, char** argv) {
  RegisterAllBenchmarks();
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
