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
#include "ihevc_sao_utils.h"

namespace {

void BM_SaoBandOffsetLuma(benchmark::State& state, SaoBandConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const int sao_band_pos = config.sao_band_pos;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ihevc_sao_band_offset_luma_ft* fn = GetSaoBandOffsetLumaFn(selector);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoLumaBuffers<UWORD8> buf = CreateSaoLumaBuffers<UWORD8>(wd, ht);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ihevc_sao_band_offset_luma_ft* ref_fn =
        GetSaoBandOffsetLumaFn(ref_selector);
    if (ref_fn) {
      SaoLumaBuffers<UWORD8> ref_buf = buf;
      SaoLumaBuffers<UWORD8> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), sao_band_pos,
             ref_buf.sao_offset.data(), wd, ht);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), sao_band_pos, tst_buf.sao_offset.data(),
         wd, ht);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size()) != 0 ||
          ref_buf.src_left != tst_buf.src_left ||
          ref_buf.src_top != tst_buf.src_top ||
          ref_buf.src_top_left != tst_buf.src_top_left) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(),
       buf.src_top.data(), buf.src_top_left.data(), sao_band_pos,
       buf.sao_offset.data(), wd, ht);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * wd * ht);
  state.SetBytesProcessed(state.iterations() * wd * ht * sizeof(UWORD8));
}

void BM_SaoBandOffsetChroma(benchmark::State& state, SaoBandConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const int interleaved_wd = 2 * wd;
  const int sao_band_pos_u = config.sao_band_pos;
  const int sao_band_pos_v = (config.sao_band_pos + 4) % 32;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ihevc_sao_band_offset_chroma_ft* fn = GetSaoBandOffsetChromaFn(selector);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoChromaBuffers<UWORD8> buf = CreateSaoChromaBuffers<UWORD8>(wd, ht);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ihevc_sao_band_offset_chroma_ft* ref_fn =
        GetSaoBandOffsetChromaFn(ref_selector);
    if (ref_fn) {
      SaoChromaBuffers<UWORD8> ref_buf = buf;
      SaoChromaBuffers<UWORD8> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), sao_band_pos_u, sao_band_pos_v,
             ref_buf.sao_offset_u.data(), ref_buf.sao_offset_v.data(),
             interleaved_wd, ht);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), sao_band_pos_u, sao_band_pos_v,
         tst_buf.sao_offset_u.data(), tst_buf.sao_offset_v.data(),
         interleaved_wd, ht);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size()) != 0 ||
          ref_buf.src_left != tst_buf.src_left ||
          ref_buf.src_top != tst_buf.src_top ||
          ref_buf.src_top_left != tst_buf.src_top_left) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(),
       buf.src_top.data(), buf.src_top_left.data(), sao_band_pos_u,
       sao_band_pos_v, buf.sao_offset_u.data(), buf.sao_offset_v.data(),
       interleaved_wd, ht);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * interleaved_wd * ht);
  state.SetBytesProcessed(state.iterations() * interleaved_wd * ht *
                          sizeof(UWORD8));
}

void BM_SaoEdgeOffsetLuma(benchmark::State& state, SaoEdgeConfig config) {
  const int edge_class = config.edge_class;
  const int wd = config.wd;
  const int ht = config.ht;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ihevc_sao_edge_offset_class0_ft* fn =
      GetSaoEdgeOffsetLumaFn(selector, edge_class);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoLumaBuffers<UWORD8> buf = CreateSaoLumaBuffers<UWORD8>(wd, ht);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ihevc_sao_edge_offset_class0_ft* ref_fn =
        GetSaoEdgeOffsetLumaFn(ref_selector, edge_class);
    if (ref_fn) {
      SaoLumaBuffers<UWORD8> ref_buf = buf;
      SaoLumaBuffers<UWORD8> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), ref_buf.src_top_right.data(),
             ref_buf.src_bot_left.data(), ref_buf.avail.data(),
             ref_buf.sao_offset.data(), wd, ht);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), tst_buf.src_top_right.data(),
         tst_buf.src_bot_left.data(), tst_buf.avail.data(),
         tst_buf.sao_offset.data(), wd, ht);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size()) != 0 ||
          ref_buf.src_left != tst_buf.src_left ||
          ref_buf.src_top != tst_buf.src_top ||
          ref_buf.src_top_left != tst_buf.src_top_left) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(),
       buf.src_top.data(), buf.src_top_left.data(), buf.src_top_right.data(),
       buf.src_bot_left.data(), buf.avail.data(), buf.sao_offset.data(), wd,
       ht);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * wd * ht);
  state.SetBytesProcessed(state.iterations() * wd * ht * sizeof(UWORD8));
}

void BM_SaoEdgeOffsetChroma(benchmark::State& state, SaoEdgeConfig config) {
  const int edge_class = config.edge_class;
  const int wd = config.wd;
  const int ht = config.ht;
  const int interleaved_wd = 2 * wd;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  ihevc_sao_edge_offset_class0_chroma_ft* fn =
      GetSaoEdgeOffsetChromaFn(selector, edge_class);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoChromaBuffers<UWORD8> buf = CreateSaoChromaBuffers<UWORD8>(wd, ht);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    const ihevc_func_selector_t* ref_selector = get_ref_func_ptr();
    ihevc_sao_edge_offset_class0_chroma_ft* ref_fn =
        GetSaoEdgeOffsetChromaFn(ref_selector, edge_class);
    if (ref_fn) {
      SaoChromaBuffers<UWORD8> ref_buf = buf;
      SaoChromaBuffers<UWORD8> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), ref_buf.src_top_right.data(),
             ref_buf.src_bot_left.data(), ref_buf.avail.data(),
             ref_buf.sao_offset_u.data(), ref_buf.sao_offset_v.data(),
             interleaved_wd, ht);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), tst_buf.src_top_right.data(),
         tst_buf.src_bot_left.data(), tst_buf.avail.data(),
         tst_buf.sao_offset_u.data(), tst_buf.sao_offset_v.data(),
         interleaved_wd, ht);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size()) != 0 ||
          ref_buf.src_left != tst_buf.src_left ||
          ref_buf.src_top != tst_buf.src_top ||
          ref_buf.src_top_left != tst_buf.src_top_left) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(),
       buf.src_top.data(), buf.src_top_left.data(), buf.src_top_right.data(),
       buf.src_bot_left.data(), buf.avail.data(), buf.sao_offset_u.data(),
       buf.sao_offset_v.data(), interleaved_wd, ht);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * interleaved_wd * ht);
  state.SetBytesProcessed(state.iterations() * interleaved_wd * ht *
                          sizeof(UWORD8));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  // Register SAO Band Offset Luma benchmarks
  for (const auto& size : GetSaoLumaBenchmarkSizes()) {
    const int wd = size.first;
    const int ht = size.second;
    for (auto arch : arches) {
      std::string arch_name = GetArchName(arch);
      std::string name = "BM_SaoBandOffsetLuma/" + std::to_string(wd) + "x" +
                         std::to_string(ht) + "/" + arch_name;
      SaoBandConfig cfg{wd, ht, /*sao_band_pos=*/0, arch};
      benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
        BM_SaoBandOffsetLuma(st, cfg);
      })->Unit(benchmark::kNanosecond);
    }
  }

  // Register SAO Band Offset Chroma benchmarks
  for (const auto& size : GetSaoChromaBenchmarkSizes()) {
    const int wd = size.first;
    const int ht = size.second;
    for (auto arch : arches) {
      std::string arch_name = GetArchName(arch);
      std::string name = "BM_SaoBandOffsetChroma/" + std::to_string(wd) + "x" +
                         std::to_string(ht) + "/" + arch_name;
      SaoBandConfig cfg{wd, ht, /*sao_band_pos=*/0, arch};
      benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
        BM_SaoBandOffsetChroma(st, cfg);
      })->Unit(benchmark::kNanosecond);
    }
  }

  // Register SAO Edge Offset Luma benchmarks
  for (int edge_class = 0; edge_class < 4; ++edge_class) {
    for (const auto& size : GetSaoLumaBenchmarkSizes()) {
      const int wd = size.first;
      const int ht = size.second;
      for (auto arch : arches) {
        std::string arch_name = GetArchName(arch);
        std::string name =
            "BM_SaoEdgeOffsetLuma/class" + std::to_string(edge_class) + "/" +
            std::to_string(wd) + "x" + std::to_string(ht) + "/" + arch_name;
        SaoEdgeConfig cfg{edge_class, wd, ht, arch};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_SaoEdgeOffsetLuma(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }

  // Register SAO Edge Offset Chroma benchmarks
  for (int edge_class = 0; edge_class < 4; ++edge_class) {
    for (const auto& size : GetSaoChromaBenchmarkSizes()) {
      const int wd = size.first;
      const int ht = size.second;
      for (auto arch : arches) {
        std::string arch_name = GetArchName(arch);
        std::string name =
            "BM_SaoEdgeOffsetChroma/class" + std::to_string(edge_class) + "/" +
            std::to_string(wd) + "x" + std::to_string(ht) + "/" + arch_name;
        SaoEdgeConfig cfg{edge_class, wd, ht, arch};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_SaoEdgeOffsetChroma(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }
  }
}
