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
#include "ihevc_sao_utils.h"

namespace {

void BM_HbdSaoBandOffsetLuma(benchmark::State& state, SaoBandConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const int sao_band_pos = config.sao_band_pos;
  const IV_ARCH_T arch = config.arch;
  const UWORD32 bit_depth = static_cast<UWORD32>(config.bit_depth);

  ihevc_hbd_sao_band_offset_luma_ft* fn = GetHbdSaoBandOffsetLumaFn(arch);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoLumaBuffers<UWORD16> buf =
      CreateSaoLumaBuffers<UWORD16>(wd, ht, config.bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    ihevc_hbd_sao_band_offset_luma_ft* ref_fn =
        GetHbdSaoBandOffsetLumaFn(ARCH_NA);
    if (ref_fn) {
      SaoLumaBuffers<UWORD16> ref_buf = buf;
      SaoLumaBuffers<UWORD16> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), sao_band_pos,
             ref_buf.sao_offset.data(), wd, ht, bit_depth);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), sao_band_pos, tst_buf.sao_offset.data(),
         wd, ht, bit_depth);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size() * sizeof(UWORD16)) != 0 ||
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
       buf.sao_offset.data(), wd, ht, bit_depth);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * wd * ht);
  state.SetBytesProcessed(state.iterations() * wd * ht * sizeof(UWORD16));
}

void BM_HbdSaoBandOffsetChroma(benchmark::State& state, SaoBandConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const int interleaved_wd = 2 * wd;
  const int sao_band_pos_u = config.sao_band_pos;
  const int sao_band_pos_v = (config.sao_band_pos + 4) % 32;
  const IV_ARCH_T arch = config.arch;
  const UWORD32 bit_depth = static_cast<UWORD32>(config.bit_depth);

  ihevc_hbd_sao_band_offset_chroma_ft* fn = GetHbdSaoBandOffsetChromaFn(arch);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoChromaBuffers<UWORD16> buf =
      CreateSaoChromaBuffers<UWORD16>(wd, ht, config.bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    ihevc_hbd_sao_band_offset_chroma_ft* ref_fn =
        GetHbdSaoBandOffsetChromaFn(ARCH_NA);
    if (ref_fn) {
      SaoChromaBuffers<UWORD16> ref_buf = buf;
      SaoChromaBuffers<UWORD16> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), sao_band_pos_u, sao_band_pos_v,
             ref_buf.sao_offset_u.data(), ref_buf.sao_offset_v.data(),
             interleaved_wd, ht, bit_depth);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), sao_band_pos_u, sao_band_pos_v,
         tst_buf.sao_offset_u.data(), tst_buf.sao_offset_v.data(),
         interleaved_wd, ht, bit_depth);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size() * sizeof(UWORD16)) != 0 ||
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
       interleaved_wd, ht, bit_depth);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * interleaved_wd * ht);
  state.SetBytesProcessed(state.iterations() * interleaved_wd * ht *
                          sizeof(UWORD16));
}

void BM_HbdSaoEdgeOffsetLuma(benchmark::State& state, SaoEdgeConfig config) {
  const int edge_class = config.edge_class;
  const int wd = config.wd;
  const int ht = config.ht;
  const IV_ARCH_T arch = config.arch;
  const UWORD32 bit_depth = static_cast<UWORD32>(config.bit_depth);

  ihevc_hbd_sao_edge_offset_class0_ft* fn =
      GetHbdSaoEdgeOffsetLumaFn(arch, edge_class);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoLumaBuffers<UWORD16> buf =
      CreateSaoLumaBuffers<UWORD16>(wd, ht, config.bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    ihevc_hbd_sao_edge_offset_class0_ft* ref_fn =
        GetHbdSaoEdgeOffsetLumaFn(ARCH_NA, edge_class);
    if (ref_fn) {
      SaoLumaBuffers<UWORD16> ref_buf = buf;
      SaoLumaBuffers<UWORD16> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), ref_buf.src_top_right.data(),
             ref_buf.src_bot_left.data(), ref_buf.avail.data(),
             ref_buf.sao_offset.data(), wd, ht, bit_depth);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), tst_buf.src_top_right.data(),
         tst_buf.src_bot_left.data(), tst_buf.avail.data(),
         tst_buf.sao_offset.data(), wd, ht, bit_depth);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size() * sizeof(UWORD16)) != 0 ||
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
       buf.src_bot_left.data(), buf.avail.data(), buf.sao_offset.data(), wd, ht,
       bit_depth);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * wd * ht);
  state.SetBytesProcessed(state.iterations() * wd * ht * sizeof(UWORD16));
}

void BM_HbdSaoEdgeOffsetChroma(benchmark::State& state, SaoEdgeConfig config) {
  const int edge_class = config.edge_class;
  const int wd = config.wd;
  const int ht = config.ht;
  const int interleaved_wd = 2 * wd;
  const IV_ARCH_T arch = config.arch;
  const UWORD32 bit_depth = static_cast<UWORD32>(config.bit_depth);

  ihevc_hbd_sao_edge_offset_class0_chroma_ft* fn =
      GetHbdSaoEdgeOffsetChromaFn(arch, edge_class);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  SaoChromaBuffers<UWORD16> buf =
      CreateSaoChromaBuffers<UWORD16>(wd, ht, config.bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    ihevc_hbd_sao_edge_offset_class0_chroma_ft* ref_fn =
        GetHbdSaoEdgeOffsetChromaFn(ARCH_NA, edge_class);
    if (ref_fn) {
      SaoChromaBuffers<UWORD16> ref_buf = buf;
      SaoChromaBuffers<UWORD16> tst_buf = buf;
      ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.stride,
             ref_buf.src_left.data(), ref_buf.src_top.data(),
             ref_buf.src_top_left.data(), ref_buf.src_top_right.data(),
             ref_buf.src_bot_left.data(), ref_buf.avail.data(),
             ref_buf.sao_offset_u.data(), ref_buf.sao_offset_v.data(),
             interleaved_wd, ht, bit_depth);
      fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.stride,
         tst_buf.src_left.data(), tst_buf.src_top.data(),
         tst_buf.src_top_left.data(), tst_buf.src_top_right.data(),
         tst_buf.src_bot_left.data(), tst_buf.avail.data(),
         tst_buf.sao_offset_u.data(), tst_buf.sao_offset_v.data(),
         interleaved_wd, ht, bit_depth);
      if (std::memcmp(ref_buf.src.data(), tst_buf.src.data(),
                      ref_buf.src.size() * sizeof(UWORD16)) != 0 ||
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
       buf.sao_offset_v.data(), interleaved_wd, ht, bit_depth);
    benchmark::DoNotOptimize(buf.src.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * interleaved_wd * ht);
  state.SetBytesProcessed(state.iterations() * interleaved_wd * ht *
                          sizeof(UWORD16));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  for (int bit_depth : {8, 10}) {
    // Register HBD SAO Band Offset Luma benchmarks
    for (const auto& size : GetSaoLumaBenchmarkSizes()) {
      const int wd = size.first;
      const int ht = size.second;
      for (auto arch : arches) {
        if (arch != ARCH_NA && !GetHbdSaoBandOffsetLumaFn(arch)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_HbdSaoBandOffsetLuma/" + std::to_string(wd) +
                           "x" + std::to_string(ht) + "/" +
                           std::to_string(bit_depth) + "bit/" + arch_name;
        SaoBandConfig cfg{wd, ht, /*sao_band_pos=*/0, arch, bit_depth};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_HbdSaoBandOffsetLuma(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }

    // Register HBD SAO Band Offset Chroma benchmarks
    for (const auto& size : GetSaoChromaBenchmarkSizes()) {
      const int wd = size.first;
      const int ht = size.second;
      for (auto arch : arches) {
        if (arch != ARCH_NA && !GetHbdSaoBandOffsetChromaFn(arch)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_HbdSaoBandOffsetChroma/" + std::to_string(wd) +
                           "x" + std::to_string(ht) + "/" +
                           std::to_string(bit_depth) + "bit/" + arch_name;
        SaoBandConfig cfg{wd, ht, /*sao_band_pos=*/0, arch, bit_depth};
        benchmark::RegisterBenchmark(name.c_str(), [cfg](benchmark::State& st) {
          BM_HbdSaoBandOffsetChroma(st, cfg);
        })->Unit(benchmark::kNanosecond);
      }
    }

    // Register HBD SAO Edge Offset Luma benchmarks
    for (int edge_class = 0; edge_class < 4; ++edge_class) {
      for (const auto& size : GetSaoLumaBenchmarkSizes()) {
        const int wd = size.first;
        const int ht = size.second;
        for (auto arch : arches) {
          if (arch != ARCH_NA && !GetHbdSaoEdgeOffsetLumaFn(arch, edge_class)) {
            continue;
          }
          std::string arch_name = GetArchName(arch);
          std::string name =
              "BM_HbdSaoEdgeOffsetLuma/class" + std::to_string(edge_class) +
              "/" + std::to_string(wd) + "x" + std::to_string(ht) + "/" +
              std::to_string(bit_depth) + "bit/" + arch_name;
          SaoEdgeConfig cfg{edge_class, wd, ht, arch, bit_depth};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_HbdSaoEdgeOffsetLuma(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }

    // Register HBD SAO Edge Offset Chroma benchmarks
    for (int edge_class = 0; edge_class < 4; ++edge_class) {
      for (const auto& size : GetSaoChromaBenchmarkSizes()) {
        const int wd = size.first;
        const int ht = size.second;
        for (auto arch : arches) {
          if (arch != ARCH_NA &&
              !GetHbdSaoEdgeOffsetChromaFn(arch, edge_class)) {
            continue;
          }
          std::string arch_name = GetArchName(arch);
          std::string name =
              "BM_HbdSaoEdgeOffsetChroma/class" + std::to_string(edge_class) +
              "/" + std::to_string(wd) + "x" + std::to_string(ht) + "/" +
              std::to_string(bit_depth) + "bit/" + arch_name;
          SaoEdgeConfig cfg{edge_class, wd, ht, arch, bit_depth};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_HbdSaoEdgeOffsetChroma(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
