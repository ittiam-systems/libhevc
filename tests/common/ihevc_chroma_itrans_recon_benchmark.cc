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
#include <memory>
#include <random>
#include <string>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_function_selector.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_structs.h"
#include "iv.h"
}
// clang-format on

#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_itrans_utils.h"

namespace {

void BM_ChromaITransRecon(benchmark::State& state, ITransBenchConfig config) {
  const int trans_size = config.trans_size;
  const int nz_cols = config.non_zero_cols;
  const int nz_rows = config.non_zero_rows;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ChromaITransReconFn fn = GetChromaITransReconFn(selector, trans_size);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  // Padding and temporary buffer allocation
  const int pad_pred = 16;
  const int pad_tmp = 16;
  const size_t tmp_size = 3 * trans_size * trans_size + pad_tmp;

  const WORD32 src_strd = trans_size;
  const WORD32 pred_strd = 2 * trans_size;
  const WORD32 dst_strd = 2 * trans_size;

  std::vector<WORD16> pi2_src(trans_size * trans_size);
  std::vector<WORD16> pi2_tmp(tmp_size);
  std::vector<UWORD8> pu1_pred(pred_strd * trans_size + pad_pred);
  std::vector<UWORD8> pu1_dst(dst_strd * trans_size);

  WORD32 zero_cols = 0;
  WORD32 zero_rows = 0;
  GenerateITransInput(trans_size, /*ttype=*/0, nz_cols, nz_rows, pi2_src.data(),
                      static_cast<UWORD8*>(nullptr), &zero_cols, &zero_rows);

  FillRandom(pu1_pred, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 1337);
  std::fill(pu1_dst.begin(), pu1_dst.end(), 0xAA);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ChromaITransReconFn ref_fn =
        GetChromaITransReconFn(ref_selector, trans_size);
    if (ref_fn) {
      std::vector<WORD16> ref_tmp(tmp_size);
      std::vector<UWORD8> ref_dst(dst_strd * trans_size, 0xAA);
      ref_fn(pi2_src.data(), ref_tmp.data(), pu1_pred.data(), ref_dst.data(),
             src_strd, pred_strd, dst_strd, zero_cols, zero_rows);
      fn(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst.data(),
         src_strd, pred_strd, dst_strd, zero_cols, zero_rows);
      if (!VerifyOutput2D(ref_dst.data(), pu1_dst.data(), 2 * trans_size,
                          trans_size, dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst.data(),
       src_strd, pred_strd, dst_strd, zero_cols, zero_rows);
    benchmark::DoNotOptimize(pu1_dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * trans_size * trans_size);
  state.SetBytesProcessed(state.iterations() * trans_size * trans_size *
                          sizeof(UWORD8));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  for (const auto& tc : GetITransTestCases()) {
    if (tc.ttype != 0) continue;  // Chroma uses DCT only
    for (const auto& nz : tc.nz_regions) {
      for (auto arch : arches) {
        const ihevc_func_selector_t* sel = get_func_ptr(arch);
        if (!sel || !GetChromaITransReconFn(sel, tc.size)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_ChromaITransRecon/" + std::to_string(tc.size) +
                           "x" + std::to_string(tc.size) + "/nz_" +
                           std::to_string(nz.cols) + "x" +
                           std::to_string(nz.rows) + "/" + arch_name;

        ITransBenchConfig cfg{tc.size, /*ttype=*/0, nz.cols, nz.rows, arch};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_ChromaITransRecon(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }
}
