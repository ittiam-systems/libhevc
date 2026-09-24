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
#include "ihevc_deblk_utils.h"

namespace {

void BM_DeblkLuma(benchmark::State& state, DeblkLumaConfig config) {
  const bool is_vert = config.is_vert;
  const int bs = config.bs;
  const int qp_p = config.qp_p;
  const int qp_q = config.qp_q;
  const int beta_offset = config.beta_offset;
  const int tc_offset = config.tc_offset;
  const int filter_p = config.filter_p;
  const int filter_q = config.filter_q;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  DeblkLumaFn fn = GetLumaTargetFn(selector, is_vert);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  std::vector<UWORD8> src(kDeblkBufSize);
  InitializeDeblkBuffer(src.data(), kDeblkBufSize);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    DeblkLumaFn ref_fn = GetLumaTargetFn(ref_selector, is_vert);
    if (ref_fn) {
      std::vector<UWORD8> ref_src(src);
      std::vector<UWORD8> tst_src(src);
      ref_fn(ref_src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q,
             beta_offset, tc_offset, filter_p, filter_q);
      fn(tst_src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q,
         beta_offset, tc_offset, filter_p, filter_q);
      if (std::memcmp(ref_src.data(), tst_src.data(), kDeblkBufSize) != 0) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q, beta_offset,
       tc_offset, filter_p, filter_q);
    benchmark::DoNotOptimize(src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * 32);
  state.SetBytesProcessed(state.iterations() * 32 * sizeof(UWORD8));
}

void BM_DeblkChroma(benchmark::State& state, DeblkChromaConfig config) {
  const bool is_vert = config.is_vert;
  const int qp_p = config.qp_p;
  const int qp_q = config.qp_q;
  const int qp_offset_u = config.qp_offset_u;
  const int qp_offset_v = config.qp_offset_v;
  const int tc_offset = config.tc_offset;
  const int filter_p = config.filter_p;
  const int filter_q = config.filter_q;
  const int chroma_fmt_idc = config.chroma_fmt_idc;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  DeblkChromaFn fn = GetChromaTargetFn(selector, is_vert);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  std::vector<UWORD8> src(kDeblkBufSize);
  InitializeDeblkBuffer(src.data(), kDeblkBufSize);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    DeblkChromaFn ref_fn = GetChromaTargetFn(ref_selector, is_vert);
    if (ref_fn) {
      std::vector<UWORD8> ref_src(src);
      std::vector<UWORD8> tst_src(src);
      ref_fn(ref_src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q,
             qp_offset_u, qp_offset_v, tc_offset, filter_p, filter_q,
             static_cast<WORD8>(chroma_fmt_idc));
      fn(tst_src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q,
         qp_offset_u, qp_offset_v, tc_offset, filter_p, filter_q,
         static_cast<WORD8>(chroma_fmt_idc));
      if (std::memcmp(ref_src.data(), tst_src.data(), kDeblkBufSize) != 0) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q, qp_offset_u,
       qp_offset_v, tc_offset, filter_p, filter_q,
       static_cast<WORD8>(chroma_fmt_idc));
    benchmark::DoNotOptimize(src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * 32);
  state.SetBytesProcessed(state.iterations() * 32 * sizeof(UWORD8));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  // Register Luma deblocking benchmarks
  for (bool is_vert : {true, false}) {
    for (int bs : {1, 2}) {
      for (int qp : {28, 44}) {
        for (const auto& filter_pair :
             {std::make_pair(1, 1), std::make_pair(1, 0)}) {
          for (auto arch : arches) {
            std::string arch_name = GetArchName(arch);
            std::string name =
                "BM_DeblkLuma/" + std::string(is_vert ? "vert" : "horz") +
                "/bs_" + std::to_string(bs) + "/qp_" + std::to_string(qp) +
                "/fP_" + std::to_string(filter_pair.first) + "_fQ_" +
                std::to_string(filter_pair.second) + "/" + arch_name;

            DeblkLumaConfig cfg{is_vert,
                                bs,
                                qp,
                                qp,
                                /*beta_offset=*/0,
                                /*tc_offset=*/0,
                                filter_pair.first,
                                filter_pair.second,
                                arch};
            benchmark::RegisterBenchmark(
                name.c_str(),
                [cfg](benchmark::State& st) { BM_DeblkLuma(st, cfg); })
                ->Unit(benchmark::kNanosecond);
          }
        }
      }
    }
  }

  // Register Chroma deblocking benchmarks
  for (bool is_vert : {true, false}) {
    for (int fmt : {1, 3}) {  // 1: YUV420, 3: YUV444
      for (int qp : {28, 44}) {
        for (auto arch : arches) {
          std::string arch_name = GetArchName(arch);
          std::string name = "BM_DeblkChroma/" +
                             std::string(is_vert ? "vert" : "horz") + "/fmt_" +
                             (fmt == 1 ? "420" : "444") + "/qp_" +
                             std::to_string(qp) + "/" + arch_name;

          DeblkChromaConfig cfg{is_vert,           qp,  qp,
                                /*qp_offset_u=*/0,
                                /*qp_offset_v=*/0,
                                /*tc_offset=*/0,
                                /*filter_p=*/1,
                                /*filter_q=*/1,    fmt, arch};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_DeblkChroma(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
