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

bool HasHbdChromaInterPredFn(IV_ARCH_T arch, InterPredOp op) {
  if (!IsW16Inp(op) && !IsW16Out(op)) {
    return GetHbdChromaInterPredFn(arch, op) != nullptr;
  } else if (!IsW16Inp(op) && IsW16Out(op)) {
    return GetHbdChromaInterPredW16outFn(arch, op) != nullptr;
  } else if (IsW16Inp(op) && !IsW16Out(op)) {
    return GetHbdChromaInterPredW16inpFn(arch, op) != nullptr;
  } else {
    return GetHbdChromaInterPredW16inpW16outFn(arch, op) != nullptr;
  }
}

void BM_HbdChromaInterPred(benchmark::State& state, InterPredConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const InterPredOp op = config.op;
  const IV_ARCH_T arch = config.arch;
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  WORD8* pi1_coeff = gai1_ihevc_chroma_filter[config.coeff_idx];
  const bool verify = (arch != ARCH_NA);

  if (!IsW16Inp(op) && !IsW16Out(op)) {
    RunHbdInterPredBenchmarkCase<UWORD16, UWORD16, 2>(
        state, GetHbdChromaInterPredFn(arch, op),
        GetHbdChromaInterPredFn(ARCH_NA, op), wd, ht, pi1_coeff, bit_depth,
        verify);
  } else if (!IsW16Inp(op) && IsW16Out(op)) {
    RunHbdInterPredBenchmarkCase<UWORD16, WORD16, 2>(
        state, GetHbdChromaInterPredW16outFn(arch, op),
        GetHbdChromaInterPredW16outFn(ARCH_NA, op), wd, ht, pi1_coeff,
        bit_depth, verify);
  } else if (IsW16Inp(op) && !IsW16Out(op)) {
    RunHbdInterPredBenchmarkCase<WORD16, UWORD16, 2>(
        state, GetHbdChromaInterPredW16inpFn(arch, op),
        GetHbdChromaInterPredW16inpFn(ARCH_NA, op), wd, ht, pi1_coeff,
        bit_depth, verify);
  } else {
    RunHbdInterPredBenchmarkCase<WORD16, WORD16, 2>(
        state, GetHbdChromaInterPredW16inpW16outFn(arch, op),
        GetHbdChromaInterPredW16inpW16outFn(ARCH_NA, op), wd, ht, pi1_coeff,
        bit_depth, verify);
  }
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  for (int bit_depth : {8, 10}) {
    // Register HBD Chroma Inter Prediction benchmarks
    for (const auto& op_info : GetInterPredOps()) {
      for (const auto& size : GetInterPredChromaBenchmarkSizes()) {
        const int wd = size.first;
        const int ht = size.second;
        for (auto arch : arches) {
          if (arch != ARCH_NA && !HasHbdChromaInterPredFn(arch, op_info.op)) {
            continue;
          }
          std::string arch_name = GetArchName(arch);
          std::string name =
              "BM_HbdChromaInterPred/" + std::string(op_info.name) + "/" +
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
            BM_HbdChromaInterPred(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
