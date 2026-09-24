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

extern "C" {
#include "ihevc_defs.h"
#include "ihevc_function_selector.h"
#include "ihevc_itrans_res.h"
#include "ihevc_macros.h"
#include "ihevc_structs.h"
#include "ihevc_typedefs.h"
#include "iv.h"
}

#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_itrans_utils.h"

namespace {

void BM_ITransRes(benchmark::State& state, ITransBenchConfig config) {
  const int trans_size = config.trans_size;
  const int ttype = config.ttype;
  const int nz_cols = config.non_zero_cols;
  const int nz_rows = config.non_zero_rows;
  const IV_ARCH_T arch = config.arch;
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ITransResFn fn = GetITransResFn(selector, trans_size, ttype);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  // Worst-case padding and temporary buffer size for safe SIMD loads/stores
  // across all platforms.
  const int pad_dst = 8;
  const int pad_tmp = 8;
  const size_t tmp_size = 3 * trans_size * trans_size + pad_tmp;

  std::vector<WORD16> pi2_coeffs(trans_size * trans_size);
  std::vector<WORD16> pi2_tmp(tmp_size);
  std::vector<WORD16> pi2_dst(trans_size * trans_size + pad_dst);
  std::vector<UWORD8> pu1_pred_dummy(trans_size * trans_size);

  WORD32 zero_cols = 0;
  WORD32 zero_rows = 0;
  GenerateITransInput(trans_size, ttype, nz_cols, nz_rows, pi2_coeffs.data(),
                      pu1_pred_dummy.data(), &zero_cols, &zero_rows);

  const WORD32 src_strd = trans_size;
  const WORD32 dst_strd = trans_size;

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ITransResFn ref_fn = GetITransResFn(ref_selector, trans_size, ttype);
    if (ref_fn) {
      std::vector<WORD16> ref_tmp(tmp_size);
      std::vector<WORD16> ref_dst(trans_size * trans_size + pad_dst);
      ref_fn(pi2_coeffs.data(), ref_tmp.data(), ref_dst.data(), src_strd,
             dst_strd, zero_cols, zero_rows, bit_depth);
      fn(pi2_coeffs.data(), pi2_tmp.data(), pi2_dst.data(), src_strd, dst_strd,
         zero_cols, zero_rows, bit_depth);
      if (std::memcmp(ref_dst.data(), pi2_dst.data(),
                      trans_size * trans_size * sizeof(WORD16)) != 0) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(pi2_coeffs.data(), pi2_tmp.data(), pi2_dst.data(), src_strd, dst_strd,
       zero_cols, zero_rows, bit_depth);
    benchmark::DoNotOptimize(pi2_dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * trans_size * trans_size);
  state.SetBytesProcessed(state.iterations() * trans_size * trans_size *
                          sizeof(WORD16));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  for (const auto& tc : GetITransTestCases()) {
    for (const auto& nz : tc.nz_regions) {
      for (auto arch : arches) {
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_ITransRes/" + std::to_string(tc.size) + "x" +
                           std::to_string(tc.size) +
                           (tc.ttype == 1 ? "_dst" : "_dct") + "/nz_" +
                           std::to_string(nz.cols) + "x" +
                           std::to_string(nz.rows) + "/" + arch_name;

        ITransBenchConfig cfg{tc.size, tc.ttype, nz.cols, nz.rows, arch, 8};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_ITransRes(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }
}
