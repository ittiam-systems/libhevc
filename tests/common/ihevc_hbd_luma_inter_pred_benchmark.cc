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

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "TestCommon.h"
#include "ihevc_inter_pred_utils.h"

namespace {

bool HasHbdLumaInterPredFn(IV_ARCH_T arch, InterPredOp op) {
  if (!IsW16Inp(op) && !IsW16Out(op)) {
    return GetHbdLumaInterPredFn(arch, op) != nullptr;
  } else if (!IsW16Inp(op) && IsW16Out(op)) {
    return GetHbdLumaInterPredW16outFn(arch, op) != nullptr;
  } else if (IsW16Inp(op) && !IsW16Out(op)) {
    return GetHbdLumaInterPredW16inpFn(arch, op) != nullptr;
  } else {
    return GetHbdLumaInterPredW16inpW16outFn(arch, op) != nullptr;
  }
}

void BM_HbdLumaInterPred(benchmark::State& state, InterPredConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const InterPredOp op = config.op;
  const IV_ARCH_T arch = config.arch;
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  WORD8* pi1_coeff = gai1_ihevc_luma_filter[config.coeff_idx];
  const bool verify = (arch != ARCH_NA);

  if (!IsW16Inp(op) && !IsW16Out(op)) {
    RunHbdInterPredBenchmarkCase<UWORD16, UWORD16, 1>(
        state, GetHbdLumaInterPredFn(arch, op),
        GetHbdLumaInterPredFn(ARCH_NA, op), wd, ht, pi1_coeff, bit_depth,
        verify);
  } else if (!IsW16Inp(op) && IsW16Out(op)) {
    RunHbdInterPredBenchmarkCase<UWORD16, WORD16, 1>(
        state, GetHbdLumaInterPredW16outFn(arch, op),
        GetHbdLumaInterPredW16outFn(ARCH_NA, op), wd, ht, pi1_coeff, bit_depth,
        verify);
  } else if (IsW16Inp(op) && !IsW16Out(op)) {
    RunHbdInterPredBenchmarkCase<WORD16, UWORD16, 1>(
        state, GetHbdLumaInterPredW16inpFn(arch, op),
        GetHbdLumaInterPredW16inpFn(ARCH_NA, op), wd, ht, pi1_coeff, bit_depth,
        verify);
  } else {
    RunHbdInterPredBenchmarkCase<WORD16, WORD16, 1>(
        state, GetHbdLumaInterPredW16inpW16outFn(arch, op),
        GetHbdLumaInterPredW16inpW16outFn(ARCH_NA, op), wd, ht, pi1_coeff,
        bit_depth, verify);
  }
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  for (int bit_depth : {8, 10}) {
    // Register HBD Luma Inter Prediction benchmarks
    for (const auto& op_info : GetInterPredOps()) {
      for (const auto& size : GetInterPredLumaBenchmarkSizes()) {
        const int wd = size.first;
        const int ht = size.second;
        for (auto arch : arches) {
          if (arch != ARCH_NA && !HasHbdLumaInterPredFn(arch, op_info.op)) {
            continue;
          }
          std::string arch_name = GetArchName(arch);
          std::string name =
              "BM_HbdLumaInterPred/" + std::string(op_info.name) + "/" +
              std::to_string(wd) + "x" + std::to_string(ht) + "/" +
              std::to_string(bit_depth) + "bit/" + arch_name;
          InterPredConfig cfg{wd,
                              ht,
                              op_info.op,
                              arch,
                              bit_depth,
                              /*coeff_idx=*/1};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_HbdLumaInterPred(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
