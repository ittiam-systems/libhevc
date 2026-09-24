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
#include "ihevc_deblk_utils.h"

namespace {

void BM_HbdDeblkLuma(benchmark::State& state, DeblkLumaConfig config) {
  const bool is_vert = config.is_vert;
  const int bs = config.bs;
  const int qp_p = config.qp_p;
  const int qp_q = config.qp_q;
  const int beta_offset = config.beta_offset;
  const int tc_offset = config.tc_offset;
  const int filter_p = config.filter_p;
  const int filter_q = config.filter_q;
  const IV_ARCH_T arch = config.arch;
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  HbdDeblkLumaFn fn = GetHbdLumaTargetFn(arch, is_vert);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  std::vector<UWORD16> src(kDeblkBufSize);
  InitializeDeblkBuffer(src.data(), kDeblkBufSize, bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    HbdDeblkLumaFn ref_fn = GetHbdLumaTargetFn(ARCH_NA, is_vert);
    if (ref_fn) {
      std::vector<UWORD16> ref_src(src);
      std::vector<UWORD16> tst_src(src);
      ref_fn(ref_src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q,
             beta_offset, tc_offset, filter_p, filter_q, bit_depth);
      fn(tst_src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q,
         beta_offset, tc_offset, filter_p, filter_q, bit_depth);
      if (std::memcmp(ref_src.data(), tst_src.data(),
                      kDeblkBufSize * sizeof(UWORD16)) != 0) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q, beta_offset,
       tc_offset, filter_p, filter_q, bit_depth);
    benchmark::DoNotOptimize(src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * 32);
  state.SetBytesProcessed(state.iterations() * 32 * sizeof(UWORD16));
}

void BM_HbdDeblkChroma(benchmark::State& state, DeblkChromaConfig config) {
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
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  HbdDeblkChromaFn fn = GetHbdChromaTargetFn(arch, is_vert);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  std::vector<UWORD16> src(kDeblkBufSize);
  InitializeDeblkBuffer(src.data(), kDeblkBufSize, bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    HbdDeblkChromaFn ref_fn = GetHbdChromaTargetFn(ARCH_NA, is_vert);
    if (ref_fn) {
      std::vector<UWORD16> ref_src(src);
      std::vector<UWORD16> tst_src(src);
      ref_fn(ref_src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q,
             qp_offset_u, qp_offset_v, tc_offset, filter_p, filter_q, bit_depth,
             static_cast<WORD8>(chroma_fmt_idc));
      fn(tst_src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q,
         qp_offset_u, qp_offset_v, tc_offset, filter_p, filter_q, bit_depth,
         static_cast<WORD8>(chroma_fmt_idc));
      if (std::memcmp(ref_src.data(), tst_src.data(),
                      kDeblkBufSize * sizeof(UWORD16)) != 0) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q, qp_offset_u,
       qp_offset_v, tc_offset, filter_p, filter_q, bit_depth,
       static_cast<WORD8>(chroma_fmt_idc));
    benchmark::DoNotOptimize(src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * 32);
  state.SetBytesProcessed(state.iterations() * 32 * sizeof(UWORD16));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  // Register HBD Luma deblocking benchmarks
  for (int bit_depth : {8, 10}) {
    for (bool is_vert : {true, false}) {
      for (int bs : {1, 2}) {
        for (int qp : {28, 44}) {
          for (const auto& filter_pair :
               {std::make_pair(1, 1), std::make_pair(1, 0)}) {
            for (auto arch : arches) {
              if (arch != ARCH_NA && !GetHbdLumaTargetFn(arch, is_vert)) {
                continue;
              }
              std::string arch_name = GetArchName(arch);
              std::string name =
                  "BM_HbdDeblkLuma/" + std::string(is_vert ? "vert" : "horz") +
                  "/bs_" + std::to_string(bs) + "/qp_" + std::to_string(qp) +
                  "/fP_" + std::to_string(filter_pair.first) + "_fQ_" +
                  std::to_string(filter_pair.second) + "/" +
                  std::to_string(bit_depth) + "bit/" + arch_name;

              DeblkLumaConfig cfg{is_vert,
                                  bs,
                                  qp,
                                  qp,
                                  /*beta_offset=*/0,
                                  /*tc_offset=*/0,
                                  filter_pair.first,
                                  filter_pair.second,
                                  arch,
                                  bit_depth};
              benchmark::RegisterBenchmark(
                  name.c_str(),
                  [cfg](benchmark::State& st) { BM_HbdDeblkLuma(st, cfg); })
                  ->Unit(benchmark::kNanosecond);
            }
          }
        }
      }
    }
  }

  // Register HBD Chroma deblocking benchmarks
  for (int bit_depth : {8, 10}) {
    for (bool is_vert : {true, false}) {
      for (int fmt : {1, 3}) {  // 1: YUV420, 3: YUV444
        for (int qp : {28, 44}) {
          for (auto arch : arches) {
            if (arch != ARCH_NA && !GetHbdChromaTargetFn(arch, is_vert)) {
              continue;
            }
            std::string arch_name = GetArchName(arch);
            std::string name = "BM_HbdDeblkChroma/" +
                               std::string(is_vert ? "vert" : "horz") +
                               "/fmt_" + (fmt == 1 ? "420" : "444") + "/qp_" +
                               std::to_string(qp) + "/" +
                               std::to_string(bit_depth) + "bit/" + arch_name;

            DeblkChromaConfig cfg{is_vert,           qp,  qp,
                                  /*qp_offset_u=*/0,
                                  /*qp_offset_v=*/0,
                                  /*tc_offset=*/0,
                                  /*filter_p=*/1,
                                  /*filter_q=*/1,    fmt, arch, bit_depth};
            benchmark::RegisterBenchmark(
                name.c_str(),
                [cfg](benchmark::State& st) { BM_HbdDeblkChroma(st, cfg); })
                ->Unit(benchmark::kNanosecond);
          }
        }
      }
    }
  }
}
