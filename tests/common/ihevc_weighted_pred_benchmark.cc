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
#include <random>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "ihevc_defs.h"
#include "ihevc_function_selector.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_typedefs.h"
#include "iv.h"
}

#include "BenchmarkCommon.h"
#include "TestCommon.h"
#include "func_selector.h"

namespace {

enum class WeightedPredType {
  kUniLuma,
  kUniChroma,
  kBiLuma,
  kBiChroma,
  kBiDefaultLuma,
  kBiDefaultChroma,
};

struct WeightedPredBenchConfig {
  int wd;
  int ht;
  WeightedPredType type;
  IV_ARCH_T arch;
};

void BM_WeightedPred(benchmark::State& state, WeightedPredBenchConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const WeightedPredType type = config.type;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  const bool is_chroma = (type == WeightedPredType::kUniChroma ||
                          type == WeightedPredType::kBiChroma ||
                          type == WeightedPredType::kBiDefaultChroma);
  const int cols = is_chroma ? (2 * wd) : wd;
  const int src_strd = cols;
  const int dst_strd = cols;

  std::vector<WORD16> src1(src_strd * ht + 16);
  std::vector<WORD16> src2(src_strd * ht + 16);
  FillRandom(src1, static_cast<WORD16>(-8192), static_cast<WORD16>(8191), 42);
  FillRandom(src2, static_cast<WORD16>(-8192), static_cast<WORD16>(8191), 43);

  std::vector<UWORD8> dst(dst_strd * ht + 16, 0xAA);
  std::vector<UWORD8> ref_dst(dst_strd * ht + 16, 0xAA);

  const int wgt0 = 64;
  const int wgt1 = 64;
  const int off0 = 0;
  const int off1 = 0;
  const int shift = 7;
  const int lvl_shift1 = 0;
  const int lvl_shift2 = 0;

  const ihevc_func_selector_t* ref_sel = get_ref_func_ptr();

  auto run_wp_bench = [&](auto fn, auto ref_fn, auto invoke) {
    if (!fn) {
      state.SkipWithError("Target function pointer is null");
      return;
    }
    if (arch != ARCH_NA && ref_fn) {
      invoke(ref_fn, ref_dst.data());
      invoke(fn, dst.data());
      if (!VerifyOutput2D(ref_dst.data(), dst.data(), cols, ht, dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
    for (auto _ : state) {
      invoke(fn, dst.data());
      benchmark::DoNotOptimize(dst.data());
      benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * cols * ht);
    state.SetBytesProcessed(state.iterations() * cols * ht * sizeof(UWORD8));
  };

  switch (type) {
    case WeightedPredType::kUniLuma:
      run_wp_bench(selector->ihevc_weighted_pred_uni_fptr,
                   ref_sel->ihevc_weighted_pred_uni_fptr,
                   [&](auto f, UWORD8* d) {
                     f(src1.data(), d, src_strd, dst_strd, wgt0, off0, shift,
                       lvl_shift1, ht, wd);
                   });
      break;
    case WeightedPredType::kUniChroma:
      run_wp_bench(selector->ihevc_weighted_pred_chroma_uni_fptr,
                   ref_sel->ihevc_weighted_pred_chroma_uni_fptr,
                   [&](auto f, UWORD8* d) {
                     f(src1.data(), d, src_strd, dst_strd, wgt0, wgt1, off0,
                       off1, shift, lvl_shift1, ht, wd);
                   });
      break;
    case WeightedPredType::kBiLuma:
      run_wp_bench(selector->ihevc_weighted_pred_bi_fptr,
                   ref_sel->ihevc_weighted_pred_bi_fptr,
                   [&](auto f, UWORD8* d) {
                     f(src1.data(), src2.data(), d, src_strd, src_strd,
                       dst_strd, wgt0, off0, wgt1, off1, shift, lvl_shift1,
                       lvl_shift2, ht, wd);
                   });
      break;
    case WeightedPredType::kBiChroma:
      run_wp_bench(selector->ihevc_weighted_pred_chroma_bi_fptr,
                   ref_sel->ihevc_weighted_pred_chroma_bi_fptr,
                   [&](auto f, UWORD8* d) {
                     f(src1.data(), src2.data(), d, src_strd, src_strd,
                       dst_strd, wgt0, wgt0, off0, off0, wgt1, wgt1, off1, off1,
                       shift, lvl_shift1, lvl_shift2, ht, wd);
                   });
      break;
    case WeightedPredType::kBiDefaultLuma:
      run_wp_bench(selector->ihevc_weighted_pred_bi_default_fptr,
                   ref_sel->ihevc_weighted_pred_bi_default_fptr,
                   [&](auto f, UWORD8* d) {
                     f(src1.data(), src2.data(), d, src_strd, src_strd,
                       dst_strd, lvl_shift1, lvl_shift2, ht, wd);
                   });
      break;
    case WeightedPredType::kBiDefaultChroma:
      run_wp_bench(selector->ihevc_weighted_pred_chroma_bi_default_fptr,
                   ref_sel->ihevc_weighted_pred_chroma_bi_default_fptr,
                   [&](auto f, UWORD8* d) {
                     f(src1.data(), src2.data(), d, src_strd, src_strd,
                       dst_strd, lvl_shift1, lvl_shift2, ht, wd);
                   });
      break;
  }
}

}  // namespace

void RegisterAllBenchmarks() {
  struct WpOpInfo {
    WeightedPredType type;
    const char* name;
    bool is_chroma;
  };

  const WpOpInfo ops[] = {
      {WeightedPredType::kUniLuma, "uni_luma", false},
      {WeightedPredType::kUniChroma, "uni_chroma", true},
      {WeightedPredType::kBiLuma, "bi_luma", false},
      {WeightedPredType::kBiChroma, "bi_chroma", true},
      {WeightedPredType::kBiDefaultLuma, "bi_default_luma", false},
      {WeightedPredType::kBiDefaultChroma, "bi_default_chroma", true},
  };

  for (const auto& op : ops) {
    const auto& sizes =
        op.is_chroma ? GetBenchmarkChromaPUSizes() : GetBenchmarkLumaPUSizes();
    for (const auto& size : sizes) {
      const int wd = size.first;
      const int ht = size.second;
      for (auto arch : GetBenchmarkArchitectures()) {
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_WeightedPred/" + std::string(op.name) + "/" +
                           std::to_string(wd) + "x" + std::to_string(ht) + "/" +
                           arch_name;

        WeightedPredBenchConfig cfg{wd, ht, op.type, arch};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_WeightedPred(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }
}
