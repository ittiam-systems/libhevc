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

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevc_chroma_recon.h"
#include "ihevc_function_selector.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_structs.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_itrans_utils.h"

namespace {

struct ChromaReconBenchConfig {
  int trans_size;
  int non_zero_cols;
  int offset;  // 0 for U, 1 for V
  IV_ARCH_T arch;
};

void BM_ChromaRecon(benchmark::State& state, ChromaReconBenchConfig config) {
  const int trans_size = config.trans_size;
  const int nz_cols = config.non_zero_cols;
  const int offset = config.offset;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ChromaReconFn fn = GetChromaReconFn(selector, trans_size);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  const WORD32 src_strd = trans_size;
  const WORD32 pred_strd = 2 * trans_size;
  const WORD32 dst_strd = 2 * trans_size;

  std::vector<WORD16> pi2_src(trans_size * trans_size);
  std::vector<UWORD8> pu1_pred(pred_strd * trans_size);
  std::vector<UWORD8> pu1_dst(dst_strd * trans_size);

  FillRandomSubBlock(pi2_src.data(), src_strd, trans_size, trans_size, nz_cols,
                     static_cast<WORD16>(-512), static_cast<WORD16>(511));
  FillRandom(pu1_pred, static_cast<UWORD8>(0), static_cast<UWORD8>(255));
  std::fill(pu1_dst.begin(), pu1_dst.end(), 0xAA);

  WORD32 zero_cols = ComputeZeroMask(trans_size, nz_cols);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ChromaReconFn ref_fn = GetChromaReconFn(ref_selector, trans_size);
    if (ref_fn) {
      std::vector<UWORD8> ref_dst(dst_strd * trans_size, 0xAA);
      ref_fn(pi2_src.data(), pu1_pred.data() + offset, ref_dst.data() + offset,
             src_strd, pred_strd, dst_strd, zero_cols);
      fn(pi2_src.data(), pu1_pred.data() + offset, pu1_dst.data() + offset,
         src_strd, pred_strd, dst_strd, zero_cols);
      if (!VerifyOutput2D(ref_dst.data(), pu1_dst.data(), 2 * trans_size,
                          trans_size, dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(pi2_src.data(), pu1_pred.data() + offset, pu1_dst.data() + offset,
       src_strd, pred_strd, dst_strd, zero_cols);
    benchmark::DoNotOptimize(pu1_dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * trans_size * trans_size);
  state.SetBytesProcessed(state.iterations() * trans_size * trans_size *
                          sizeof(UWORD8));
}

}  // namespace

void RegisterAllBenchmarks() {
  const int sizes[] = {4, 8, 16, 32};
  const int offsets[] = {0, 1};

  for (int size : sizes) {
    std::vector<int> nz_options;
    if (size == 4) {
      nz_options = {1, 2, 4};
    } else if (size == 8) {
      nz_options = {2, 4, 8};
    } else if (size == 16) {
      nz_options = {4, 8, 16};
    } else {
      nz_options = {8, 16, 32};
    }

    for (int nz : nz_options) {
      for (int offset : offsets) {
        for (auto arch : GetBenchmarkArchitectures()) {
          const ihevc_func_selector_t* sel = get_func_ptr(arch);
          if (!sel || !GetChromaReconFn(sel, size)) {
            continue;
          }
          std::string arch_name = GetArchName(arch);
          std::string name = "BM_ChromaRecon/" + std::to_string(size) + "x" +
                             std::to_string(size) + "/nz_" +
                             std::to_string(nz) + "/" +
                             (offset == 0 ? "U" : "V") + "/" + arch_name;

          ChromaReconBenchConfig cfg{size, nz, offset, arch};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_ChromaRecon(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
