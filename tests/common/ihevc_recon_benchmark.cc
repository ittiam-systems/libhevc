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
#include <vector>

extern "C" {
#include "ihevc_defs.h"
#include "ihevc_function_selector.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_recon.h"
#include "ihevc_structs.h"
#include "ihevc_typedefs.h"
#include "iv.h"
}

#include "BenchmarkCommon.h"
#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_itrans_utils.h"

namespace {

struct ReconBenchConfig {
  int trans_size;
  int ttype;  // 0: DCT, 1: DST (4x4 only)
  int non_zero_cols;
  IV_ARCH_T arch;
};

void BM_Recon(benchmark::State& state, ReconBenchConfig config) {
  const int trans_size = config.trans_size;
  const int ttype = config.ttype;
  const int nz_cols = config.non_zero_cols;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ReconFn fn = GetReconFn(selector, trans_size, ttype);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  const WORD32 src_strd = trans_size;
  const WORD32 pred_strd = trans_size;
  const WORD32 dst_strd = trans_size;

  std::vector<WORD16> pi2_src(trans_size * trans_size);
  std::vector<UWORD8> pu1_pred(trans_size * trans_size);
  std::vector<UWORD8> pu1_dst(trans_size * trans_size);

  FillRandomSubBlock(pi2_src.data(), src_strd, trans_size, trans_size, nz_cols,
                     static_cast<WORD16>(-512), static_cast<WORD16>(511));
  FillRandom(pu1_pred, static_cast<UWORD8>(0), static_cast<UWORD8>(255));
  std::fill(pu1_dst.begin(), pu1_dst.end(), 0xAA);

  WORD32 zero_cols = ComputeZeroMask(trans_size, nz_cols);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ReconFn ref_fn = GetReconFn(ref_selector, trans_size, ttype);
    if (ref_fn) {
      std::vector<UWORD8> ref_dst(dst_strd * trans_size, 0xAA);
      ref_fn(pi2_src.data(), pu1_pred.data(), ref_dst.data(), src_strd,
             pred_strd, dst_strd, zero_cols);
      fn(pi2_src.data(), pu1_pred.data(), pu1_dst.data(), src_strd, pred_strd,
         dst_strd, zero_cols);
      if (std::memcmp(ref_dst.data(), pu1_dst.data(),
                      trans_size * trans_size * sizeof(UWORD8)) != 0) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(pi2_src.data(), pu1_pred.data(), pu1_dst.data(), src_strd, pred_strd,
       dst_strd, zero_cols);
    benchmark::DoNotOptimize(pu1_dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * trans_size * trans_size);
  state.SetBytesProcessed(state.iterations() * trans_size * trans_size *
                          sizeof(UWORD8));
}

}  // namespace

void RegisterAllBenchmarks() {
  struct TestCase {
    int size;
    int ttype;
    std::vector<int> nz_cols;
  };

  const TestCase test_cases[] = {
      {4, 0, {1, 2, 4}},   {4, 1, {1, 2, 4}},    {8, 0, {2, 4, 8}},
      {16, 0, {4, 8, 16}}, {32, 0, {8, 16, 32}},
  };

  for (const auto& tc : test_cases) {
    for (int nz : tc.nz_cols) {
      for (auto arch : GetBenchmarkArchitectures()) {
        const ihevc_func_selector_t* sel = get_func_ptr(arch);
        if (!sel || !GetReconFn(sel, tc.size, tc.ttype)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_Recon/" + std::to_string(tc.size) + "x" +
                           std::to_string(tc.size) +
                           (tc.ttype == 1 ? "_dst" : "_dct") + "/nz_" +
                           std::to_string(nz) + "/" + arch_name;

        ReconBenchConfig cfg{tc.size, tc.ttype, nz, arch};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_Recon(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }
}
