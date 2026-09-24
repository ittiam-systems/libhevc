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

// Buffers structure for inter prediction
template <typename SrcT, typename DstT>
struct InterPredBuffers {
  int src_strd;
  int dst_strd;
  int src_offset;
  std::vector<SrcT> src;
  std::vector<DstT> dst;
};

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

#if defined(GTEST_TEST) || defined(GOOGLETEST_INCLUDE_GTEST_GTEST_H_) || \
    defined(GTEST_INCLUDE_GTEST_GTEST_H_)
// Test parameters: width, height, src_stride_mul, dst_stride_mul, coeff_idx,
// arch
using InterPredTestParam =
    std::tuple<std::pair<int, int>, int, int, int, IV_ARCH_T>;

template <int kComponents>
std::string PrintInterPredTestParam(
    const testing::TestParamInfo<InterPredTestParam>& info) {
  int wd, ht, src_strd_mul, dst_strd_mul, coeff_idx;
  IV_ARCH_T arch;
  std::pair<int, int> block_size;
  std::tie(block_size, src_strd_mul, dst_strd_mul, coeff_idx, arch) =
      info.param;
  std::tie(wd, ht) = block_size;
  return std::to_string(wd) + "x" + std::to_string(ht) + "_src_stride_" +
         std::to_string(src_strd_mul * kComponents * wd) + "_dst_stride_" +
         std::to_string(dst_strd_mul * kComponents * wd) + "_coeff_" +
         std::to_string(coeff_idx) + "_" + get_arch_str(arch);
}

template <typename SrcT, typename DstT, int kComponents>
class InterPredTestBase : public ::testing::TestWithParam<InterPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_strd_mul, dst_strd_mul, coeff_idx, arch) =
        GetParam();
    std::tie(wd, ht) = block_size;
    src_strd = kComponents * wd * src_strd_mul;
    dst_strd = kComponents * wd * dst_strd_mul;

    // Increase allocations for x86/x86_64 to avoid out-of-bounds reads/writes
    // in SIMD implementations.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
    int pad_dst = 16;
#else
    int pad_dst = 0;
#endif

    dst_buf_ref.resize(dst_strd * ht + pad_dst);
    dst_buf_tst.resize(dst_strd * ht + pad_dst);

    // Set pv_src to a valid position within src_buf to allow negative indexing
    pv_src = reinterpret_cast<SrcT*>(const_cast<UWORD8*>(getSrc8Buf().data())) +
             (kTapSize / 2) * src_strd;
    pv_dst_ref = dst_buf_ref.data();
    pv_dst_tst = dst_buf_tst.data();

    pi1_coeffs = (kComponents == 1) ? gai1_ihevc_luma_filter[coeff_idx]
                                    : gai1_ihevc_chroma_filter[coeff_idx];
    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  template <typename FuncPtr>
  void RunTest(FuncPtr func_ptr) {
    (ref->*func_ptr)(pv_src, pv_dst_ref, src_strd, dst_strd, pi1_coeffs, ht,
                     wd);
    (tst->*func_ptr)(pv_src, pv_dst_tst, src_strd, dst_strd, pi1_coeffs, ht,
                     wd);
    ASSERT_NO_FATAL_FAILURE(compare_output<DstT>(dst_buf_ref, dst_buf_tst,
                                                 kComponents * wd, ht,
                                                 dst_strd));
  }

  int wd, ht, src_strd_mul, dst_strd_mul, coeff_idx;
  int src_strd, dst_strd;
  std::vector<DstT> dst_buf_ref;
  std::vector<DstT> dst_buf_tst;
  SrcT* pv_src;
  DstT* pv_dst_ref;
  DstT* pv_dst_tst;
  WORD8* pi1_coeffs;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* tst;
  const ihevc_func_selector_t* ref;
};
#endif

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
