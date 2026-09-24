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
#include "ihevc_intra_pred_utils.h"

namespace {

void BM_LumaIntraPred(benchmark::State& state, IntraPredConfig config) {
  const int nt = config.nt;
  const int mode = config.mode;
  const int mode_or_flag =
      (mode == 10 || mode == 26) ? config.disable_boundary_filter : mode;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  LumaIntraPredFn fn = GetLumaIntraPredFn(selector, mode);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  IntraPredLumaBuffers<UWORD8> buf =
      CreateIntraPredLumaBuffers<UWORD8>(nt, config.dst_strd_mul);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    LumaIntraPredFn ref_fn = GetLumaIntraPredFn(get_ref_func_ptr(), mode);
    if (ref_fn) {
      IntraPredLumaBuffers<UWORD8> ref_buf = buf;
      IntraPredLumaBuffers<UWORD8> tst_buf = buf;
      ref_fn(ref_buf.ref.data(), ref_buf.src_strd, ref_buf.dst.data(),
             ref_buf.dst_strd, nt, mode_or_flag);
      fn(tst_buf.ref.data(), tst_buf.src_strd, tst_buf.dst.data(),
         tst_buf.dst_strd, nt, mode_or_flag);
      if (!VerifyOutput2D(ref_buf.dst.data(), tst_buf.dst.data(), nt, nt,
                          ref_buf.dst_strd, tst_buf.dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(buf.ref.data(), buf.src_strd, buf.dst.data(), buf.dst_strd, nt,
       mode_or_flag);
    benchmark::DoNotOptimize(buf.dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * nt * nt);
  state.SetBytesProcessed(state.iterations() * nt * nt * sizeof(UWORD8));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  // Register Luma Intra Prediction benchmarks
  for (const auto& mode_info : GetIntraPredBenchmarkModes()) {
    for (int nt : GetIntraPredBenchmarkSizes()) {
      for (auto arch : arches) {
        const ihevc_func_selector_t* sel = get_func_ptr(arch);
        if (!sel || !GetLumaIntraPredFn(sel, mode_info.mode)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_LumaIntraPred/" + std::string(mode_info.name) +
                           "/" + std::to_string(nt) + "x" + std::to_string(nt) +
                           "/" + arch_name;
        IntraPredConfig cfg{nt, mode_info.mode, /*dst_strd_mul=*/1,
                            /*disable_boundary_filter=*/0, arch};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_LumaIntraPred(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }
}
