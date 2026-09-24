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
#include "ihevc_intra_pred_utils.h"

namespace {

void BM_HbdChromaIntraPred(benchmark::State& state, IntraPredConfig config) {
  const int nt = config.nt;
  const int mode = config.mode;
  const IV_ARCH_T arch = config.arch;

  HbdChromaIntraPredFn fn = GetHbdChromaIntraPredFn(arch, mode);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  IntraPredChromaBuffers<UWORD16> buf = CreateIntraPredChromaBuffers<UWORD16>(
      nt, config.dst_strd_mul, config.bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    HbdChromaIntraPredFn ref_fn = GetHbdChromaIntraPredFn(ARCH_NA, mode);
    if (ref_fn) {
      IntraPredChromaBuffers<UWORD16> ref_buf = buf;
      IntraPredChromaBuffers<UWORD16> tst_buf = buf;
      ref_fn(ref_buf.ref.data(), ref_buf.src_strd, ref_buf.dst.data(),
             ref_buf.dst_strd, nt, mode);
      fn(tst_buf.ref.data(), tst_buf.src_strd, tst_buf.dst.data(),
         tst_buf.dst_strd, nt, mode);
      if (!VerifyOutput2D(ref_buf.dst.data(), tst_buf.dst.data(), 2 * nt, nt,
                          ref_buf.dst_strd, tst_buf.dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(buf.ref.data(), buf.src_strd, buf.dst.data(), buf.dst_strd, nt, mode);
    benchmark::DoNotOptimize(buf.dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * 2 * nt * nt);
  state.SetBytesProcessed(state.iterations() * 2 * nt * nt * sizeof(UWORD16));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  for (int bit_depth : {8, 10}) {
    // Register HBD Chroma Intra Prediction benchmarks
    for (const auto& mode_info : GetIntraPredBenchmarkModes()) {
      for (int nt : GetIntraPredBenchmarkSizes()) {
        for (auto arch : arches) {
          if (arch != ARCH_NA &&
              !GetHbdChromaIntraPredFn(arch, mode_info.mode)) {
            continue;
          }
          std::string arch_name = GetArchName(arch);
          std::string name =
              "BM_HbdChromaIntraPred/" + std::string(mode_info.name) + "/" +
              std::to_string(nt) + "x" + std::to_string(nt) + "/" +
              std::to_string(bit_depth) + "bit/" + arch_name;
          IntraPredConfig cfg{nt,
                              mode_info.mode,
                              /*dst_strd_mul=*/1,
                              /*disable_boundary_filter=*/0,
                              arch,
                              bit_depth};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_HbdChromaIntraPred(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
