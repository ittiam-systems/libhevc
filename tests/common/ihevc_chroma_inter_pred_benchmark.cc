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
#include "func_selector.h"
#include "ihevc_inter_pred_utils.h"

namespace {

void BM_ChromaInterPred(benchmark::State& state, InterPredConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const InterPredOp op = config.op;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  WORD8* pi1_coeff = gai1_ihevc_chroma_filter[config.coeff_idx];
  const bool verify = (arch != ARCH_NA);

  if (!IsW16Inp(op) && !IsW16Out(op)) {
    RunInterPredBenchmarkCase<UWORD8, UWORD8, 2>(
        state, GetChromaInterPredFn(selector, op),
        GetChromaInterPredFn(get_ref_func_ptr(), op), wd, ht, pi1_coeff,
        verify);
  } else if (!IsW16Inp(op) && IsW16Out(op)) {
    RunInterPredBenchmarkCase<UWORD8, WORD16, 2>(
        state, GetChromaInterPredW16outFn(selector, op),
        GetChromaInterPredW16outFn(get_ref_func_ptr(), op), wd, ht, pi1_coeff,
        verify);
  } else if (IsW16Inp(op) && !IsW16Out(op)) {
    RunInterPredBenchmarkCase<WORD16, UWORD8, 2>(
        state, GetChromaInterPredW16inpFn(selector, op),
        GetChromaInterPredW16inpFn(get_ref_func_ptr(), op), wd, ht, pi1_coeff,
        verify);
  } else {
    RunInterPredBenchmarkCase<WORD16, WORD16, 2>(
        state, GetChromaInterPredW16inpW16outFn(selector, op),
        GetChromaInterPredW16inpW16outFn(get_ref_func_ptr(), op), wd, ht,
        pi1_coeff, verify);
  }
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  // Register Chroma Inter Prediction benchmarks
  for (const auto& op_info : GetInterPredOps()) {
    for (const auto& size : GetInterPredChromaBenchmarkSizes()) {
      const int wd = size.first;
      const int ht = size.second;
      for (auto arch : arches) {
        if (!get_func_ptr(arch)) continue;
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_ChromaInterPred/" + std::string(op_info.name) +
                           "/" + std::to_string(wd) + "x" + std::to_string(ht) +
                           "/" + arch_name;
        InterPredConfig cfg{wd,
                            ht,
                            op_info.op,
                            arch,
                            /*bit_depth=*/8,
                            /*coeff_idx=*/1};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_ChromaInterPred(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }
}
