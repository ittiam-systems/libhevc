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
#include "ihevc_padding.h"
#include "ihevc_platform_macros.h"
#include "ihevc_typedefs.h"
#include "iv.h"
}

#include "BenchmarkCommon.h"
#include "TestCommon.h"
#include "func_selector.h"

namespace {

enum class PadType {
  kLeftLuma,
  kRightLuma,
  kLeftChroma,
  kRightChroma,
};

struct PaddingBenchConfig {
  int wd;
  int ht;
  int pad_size;
  PadType pad_type;
  IV_ARCH_T arch;
};

void BM_Padding(benchmark::State& state, PaddingBenchConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const int pad_size = config.pad_size;
  const PadType pad_type = config.pad_type;
  const IV_ARCH_T arch = config.arch;

  const ihevc_func_selector_t* selector = get_func_ptr(arch);
  if (!selector) {
    state.SkipWithError("Architecture not supported on this platform");
    return;
  }

  const bool is_chroma =
      (pad_type == PadType::kLeftChroma || pad_type == PadType::kRightChroma);
  const int block_wd = is_chroma ? (2 * wd) : wd;
  const int stride = block_wd + 2 * pad_size + 32;
  const int total_ht = ht + 2 * pad_size + 32;
  const int src_offset = pad_size * stride + pad_size + 16;

  std::vector<UWORD8> buf(stride * total_ht, 0xAA);

  std::mt19937 rng(42);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  for (int r = 0; r < ht; ++r) {
    for (int c = 0; c < block_wd; ++c) {
      buf[src_offset + r * stride + c] = dist(rng);
    }
  }

  using PadFn = void (*)(UWORD8*, WORD32, WORD32, WORD32);
  PadFn fn = nullptr;
  PadFn ref_fn = nullptr;
  int offset = src_offset;

  const ihevc_func_selector_t* ref_sel = get_ref_func_ptr();

  switch (pad_type) {
    case PadType::kLeftLuma:
      fn = selector->ihevc_pad_left_luma_fptr;
      ref_fn = ref_sel->ihevc_pad_left_luma_fptr;
      offset = src_offset;
      break;
    case PadType::kRightLuma:
      fn = selector->ihevc_pad_right_luma_fptr;
      ref_fn = ref_sel->ihevc_pad_right_luma_fptr;
      offset = src_offset + wd;
      break;
    case PadType::kLeftChroma:
      fn = selector->ihevc_pad_left_chroma_fptr;
      ref_fn = ref_sel->ihevc_pad_left_chroma_fptr;
      offset = src_offset;
      break;
    case PadType::kRightChroma:
      fn = selector->ihevc_pad_right_chroma_fptr;
      ref_fn = ref_sel->ihevc_pad_right_chroma_fptr;
      offset = src_offset + 2 * wd;
      break;
  }

  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  // Pre-measurement verification against reference
  if (arch != ARCH_NA && ref_fn) {
    auto ref_buf = buf;
    auto tst_buf = buf;
    ref_fn(ref_buf.data() + offset, stride, ht, pad_size);
    fn(tst_buf.data() + offset, stride, ht, pad_size);
    if (ref_buf != tst_buf) {
      state.SkipWithError("Output mismatch between SIMD and C reference");
      return;
    }
  }

  for (auto _ : state) {
    fn(buf.data() + offset, stride, ht, pad_size);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * ht * pad_size);
  state.SetBytesProcessed(state.iterations() * ht * pad_size * sizeof(UWORD8));
}

}  // namespace

void RegisterAllBenchmarks() {
  struct PadOpInfo {
    PadType type;
    const char* name;
    bool is_chroma;
  };

  const PadOpInfo ops[] = {
      {PadType::kLeftLuma, "left_luma", false},
      {PadType::kRightLuma, "right_luma", false},
      {PadType::kLeftChroma, "left_chroma", true},
      {PadType::kRightChroma, "right_chroma", true},
  };

  const int pad_sizes[] = {80, 16};

  for (const auto& op : ops) {
    const auto& sizes =
        op.is_chroma ? GetBenchmarkChromaPUSizes() : GetBenchmarkLumaPUSizes();
    for (const auto& size : sizes) {
      const int wd = size.first;
      const int ht = size.second;
      for (int pad_size : pad_sizes) {
        for (auto arch : GetBenchmarkArchitectures()) {
          std::string arch_name = GetArchName(arch);
          std::string name = "BM_Padding/" + std::string(op.name) + "/" +
                             std::to_string(wd) + "x" + std::to_string(ht) +
                             "/pad_" + std::to_string(pad_size) + "/" +
                             arch_name;

          PaddingBenchConfig cfg{wd, ht, pad_size, op.type, arch};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_Padding(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
