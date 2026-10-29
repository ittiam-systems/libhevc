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

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevc_inter_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"
#include "func_selector.h"

enum class InterPredOp {
  kCopy,
  kHorz,
  kVert,
  kCopyW16out,
  kHorzW16out,
  kVertW16out,
  kVertW16inp,
  kVertW16inpW16out,
};

struct InterPredOpInfo {
  InterPredOp op;
  const char* name;
  bool is_w16inp;
  bool is_w16out;
};

const std::vector<InterPredOpInfo>& GetInterPredOps();

inline bool IsW16Inp(InterPredOp op) {
  return op == InterPredOp::kVertW16inp || op == InterPredOp::kVertW16inpW16out;
}

inline bool IsW16Out(InterPredOp op) {
  return op == InterPredOp::kCopyW16out || op == InterPredOp::kHorzW16out ||
         op == InterPredOp::kVertW16out || op == InterPredOp::kVertW16inpW16out;
}

// Configuration for inter prediction benchmarks
struct InterPredConfig {
  int wd;
  int ht;
  InterPredOp op;
  IV_ARCH_T arch = ARCH_NA;
  int bit_depth = 8;
  int coeff_idx = 1;
};

// Buffers structure for inter prediction benchmarks
template <typename SrcT, typename DstT>
struct InterPredBuffers {
  int src_strd;
  int dst_strd;
  int src_offset;
  std::vector<SrcT> src;
  std::vector<DstT> dst;
};

// Buffers structure for inter prediction unit tests
template <typename SrcT, typename DstT>
struct InterPredTestBuffers {
  std::vector<SrcT> src;
  std::vector<DstT> dst_ref;
  std::vector<DstT> dst_tst;
  int src_offset = 0;

  SrcT* src_ptr() { return src.data() + src_offset; }
};

// Creates and initializes inter prediction buffers for unit tests
template <typename SrcT, typename DstT>
InterPredTestBuffers<SrcT, DstT> CreateInterPredTestBuffers(int ht, int src_strd, int dst_strd,
                                                            SrcT src_min, SrcT src_max, int top,
                                                            int bot, int pad_dst = 0) {
// Largest SIMD reads outside the block: 3 before (luma filter), and 12 after (16-byte load on a
// 4-sample row)
const int pad_before = 3;
const int pad_after = 12;

InterPredTestBuffers<SrcT, DstT> b;
b.src_offset = top * src_strd + pad_before;
b.src.resize(src_strd * (top + ht + bot) + pad_before + pad_after);

  FillRandom(b.src, src_min, src_max, 42);

  // Random dst, so any write outside the block changes a value
  b.dst_ref.resize(dst_strd * ht + pad_dst);
  FillRandom(b.dst_ref, std::numeric_limits<DstT>::min(), std::numeric_limits<DstT>::max(), 1337);
  b.dst_tst = b.dst_ref;

  return b;
}

// Creates and initializes inter prediction buffers for Luma (components=1) or
// Chroma (components=2)
template <typename SrcT, typename DstT>
InterPredBuffers<SrcT, DstT> CreateInterPredBuffers(int wd, int ht,
                                                    int components,
                                                    int bit_depth = 8) {
  InterPredBuffers<SrcT, DstT> b;
  const int margin = 16;
  b.src_strd = components * wd + 2 * margin;
  const int total_ht = ht + 2 * margin;
  b.src_offset = margin * b.src_strd + margin;
  b.src.resize(b.src_strd * total_ht);

  b.dst_strd = components * wd + 32;
  b.dst.resize(b.dst_strd * (ht + 16));

  const SrcT max_val = static_cast<SrcT>((1 << bit_depth) - 1);
  FillRandom(b.src, static_cast<SrcT>(0), max_val, 42);
  std::fill(b.dst.begin(), b.dst.end(), static_cast<DstT>(0xCD));

  return b;
}

// Creates and initializes Luma inter prediction buffers
template <typename SrcT, typename DstT>
InterPredBuffers<SrcT, DstT> CreateInterPredLumaBuffers(int wd, int ht,
                                                        int bit_depth = 8) {
  return CreateInterPredBuffers<SrcT, DstT>(wd, ht, 1, bit_depth);
}

// Creates and initializes Chroma inter prediction buffers
template <typename SrcT, typename DstT>
InterPredBuffers<SrcT, DstT> CreateInterPredChromaBuffers(int wd, int ht,
                                                          int bit_depth = 8) {
  return CreateInterPredBuffers<SrcT, DstT>(wd, ht, 2, bit_depth);
}

// Function getters for 8-bit Luma and Chroma
ihevc_inter_pred_ft* GetLumaInterPredFn(const ihevc_func_selector_t* selector,
                                        InterPredOp op);
ihevc_inter_pred_w16out_ft* GetLumaInterPredW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op);
ihevc_inter_pred_w16inp_ft* GetLumaInterPredW16inpFn(
    const ihevc_func_selector_t* selector, InterPredOp op);
ihevc_inter_pred_w16inp_w16out_ft* GetLumaInterPredW16inpW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op);

ihevc_inter_pred_ft* GetChromaInterPredFn(const ihevc_func_selector_t* selector,
                                          InterPredOp op);
ihevc_inter_pred_w16out_ft* GetChromaInterPredW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op);
ihevc_inter_pred_w16inp_ft* GetChromaInterPredW16inpFn(
    const ihevc_func_selector_t* selector, InterPredOp op);
ihevc_inter_pred_w16inp_w16out_ft* GetChromaInterPredW16inpW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op);

// Function getters for HBD Luma and Chroma
ihevc_hbd_inter_pred_ft* GetHbdLumaInterPredFn(const ihevc_func_selector_t* selector,
                                               InterPredOp op);
ihevc_hbd_inter_pred_w16out_ft* GetHbdLumaInterPredW16outFn(const ihevc_func_selector_t* selector,
                                                            InterPredOp op);
ihevc_hbd_inter_pred_w16inp_ft* GetHbdLumaInterPredW16inpFn(const ihevc_func_selector_t* selector,
                                                            InterPredOp op);
ihevc_hbd_inter_pred_w16inp_w16out_ft* GetHbdLumaInterPredW16inpW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op);

ihevc_hbd_inter_pred_ft* GetHbdChromaInterPredFn(const ihevc_func_selector_t* selector,
                                                 InterPredOp op);
ihevc_hbd_inter_pred_w16out_ft* GetHbdChromaInterPredW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op);
ihevc_hbd_inter_pred_w16inp_ft* GetHbdChromaInterPredW16inpFn(
    const ihevc_func_selector_t* selector, InterPredOp op);
ihevc_hbd_inter_pred_w16inp_w16out_ft* GetHbdChromaInterPredW16inpW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op);

ihevc_hbd_inter_pred_ft* GetHbdLumaInterPredFn(IV_ARCH_T arch, InterPredOp op);
ihevc_hbd_inter_pred_w16out_ft* GetHbdLumaInterPredW16outFn(IV_ARCH_T arch,
                                                            InterPredOp op);
ihevc_hbd_inter_pred_w16inp_ft* GetHbdLumaInterPredW16inpFn(IV_ARCH_T arch,
                                                            InterPredOp op);
ihevc_hbd_inter_pred_w16inp_w16out_ft* GetHbdLumaInterPredW16inpW16outFn(
    IV_ARCH_T arch, InterPredOp op);

ihevc_hbd_inter_pred_ft* GetHbdChromaInterPredFn(IV_ARCH_T arch,
                                                 InterPredOp op);
ihevc_hbd_inter_pred_w16out_ft* GetHbdChromaInterPredW16outFn(IV_ARCH_T arch,
                                                              InterPredOp op);
ihevc_hbd_inter_pred_w16inp_ft* GetHbdChromaInterPredW16inpFn(IV_ARCH_T arch,
                                                              InterPredOp op);
ihevc_hbd_inter_pred_w16inp_w16out_ft* GetHbdChromaInterPredW16inpW16outFn(
    IV_ARCH_T arch, InterPredOp op);

// Benchmark sizes
const std::vector<std::pair<int, int>>& GetInterPredLumaBenchmarkSizes();
const std::vector<std::pair<int, int>>& GetInterPredChromaBenchmarkSizes();

#ifdef BENCHMARK_BENCHMARK_H_
template <typename SrcT, typename DstT, int kComponents, typename FnT>
void RunInterPredBenchmarkCase(benchmark::State& state, FnT fn, FnT ref_fn,
                               int wd, int ht, WORD8* pi1_coeff,
                               bool verify_output) {
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }
  auto buf = CreateInterPredBuffers<SrcT, DstT>(wd, ht, kComponents);
  if (verify_output && ref_fn) {
    auto ref_buf = buf;
    auto tst_buf = buf;
    ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.dst.data(),
           ref_buf.src_strd, ref_buf.dst_strd, pi1_coeff, ht, wd);
    fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.dst.data(),
       tst_buf.src_strd, tst_buf.dst_strd, pi1_coeff, ht, wd);
    if (!VerifyOutput2D(ref_buf.dst.data(), tst_buf.dst.data(),
                        kComponents * wd, ht, ref_buf.dst_strd)) {
      state.SkipWithError("Output mismatch between SIMD and C reference");
      return;
    }
  }
  for (auto _ : state) {
    fn(buf.src.data() + buf.src_offset, buf.dst.data(), buf.src_strd,
       buf.dst_strd, pi1_coeff, ht, wd);
    benchmark::DoNotOptimize(buf.dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kComponents * wd * ht);
  state.SetBytesProcessed(state.iterations() * kComponents * wd * ht *
                          sizeof(DstT));
}

template <typename SrcT, typename DstT, int kComponents, typename FnT>
void RunHbdInterPredBenchmarkCase(benchmark::State& state, FnT fn, FnT ref_fn,
                                  int wd, int ht, WORD8* pi1_coeff,
                                  UWORD8 bit_depth, bool verify_output) {
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }
  auto buf = CreateInterPredBuffers<SrcT, DstT>(wd, ht, kComponents, bit_depth);
  if (verify_output && ref_fn) {
    auto ref_buf = buf;
    auto tst_buf = buf;
    ref_fn(ref_buf.src.data() + ref_buf.src_offset, ref_buf.dst.data(),
           ref_buf.src_strd, ref_buf.dst_strd, pi1_coeff, ht, wd, bit_depth);
    fn(tst_buf.src.data() + tst_buf.src_offset, tst_buf.dst.data(),
       tst_buf.src_strd, tst_buf.dst_strd, pi1_coeff, ht, wd, bit_depth);
    if (!VerifyOutput2D(ref_buf.dst.data(), tst_buf.dst.data(),
                        kComponents * wd, ht, ref_buf.dst_strd)) {
      state.SkipWithError("Output mismatch between SIMD and C reference");
      return;
    }
  }
  for (auto _ : state) {
    fn(buf.src.data() + buf.src_offset, buf.dst.data(), buf.src_strd,
       buf.dst_strd, pi1_coeff, ht, wd, bit_depth);
    benchmark::DoNotOptimize(buf.dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kComponents * wd * ht);
  state.SetBytesProcessed(state.iterations() * kComponents * wd * ht *
                          sizeof(DstT));
}
#endif
