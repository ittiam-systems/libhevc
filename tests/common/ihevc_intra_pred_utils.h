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
#include "ihevc_intra_pred.h"
#include "ihevc_chroma_intra_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"
#include "func_selector.h"

// Configuration for intra prediction benchmarks
struct IntraPredConfig {
  int nt;                           // 4, 8, 16, 32
  int mode;                         // 0..34
  int dst_strd_mul = 1;             // Multiplier for dst stride
  int disable_boundary_filter = 0;  // For luma mode 10 and 26
  IV_ARCH_T arch = ARCH_NA;
  int bit_depth = 8;
};

// Buffers structure for Luma and Chroma intra prediction
template <typename T>
struct IntraPredBuffers {
  int nt;
  int src_strd;
  int dst_strd;
  std::vector<T> ref;
  std::vector<T> dst;
};

template <typename T>
using IntraPredLumaBuffers = IntraPredBuffers<T>;

template <typename T>
using IntraPredChromaBuffers = IntraPredBuffers<T>;

// Creates and initializes intra prediction buffers (components = 1 for Luma, 2 for Chroma)
template <typename T>
IntraPredBuffers<T> CreateIntraPredBuffers(int nt, int components,
                                           int dst_strd_mul = 1,
                                           int bit_depth = 8,
                                           uint32_t seed = 42) {
  IntraPredBuffers<T> b;
  b.nt = nt;
  b.src_strd = 1;
  b.dst_strd = components * nt * dst_strd_mul;

  // Reference buffer: components * (4 * nt + 1) plus padding to protect SIMD over-reads
  const int ref_size = components * (4 * nt + 1) + 64;
  const int dst_size = b.dst_strd * nt + 64;

  b.ref.resize(ref_size);
  b.dst.resize(dst_size);

  const T max_val = static_cast<T>((1 << bit_depth) - 1);
  FillRandom(b.ref, static_cast<T>(0), max_val, seed);
  std::fill(b.dst.begin(), b.dst.end(), static_cast<T>(0xCD));

  return b;
}

// Creates and initializes Luma intra prediction buffers
template <typename T>
IntraPredLumaBuffers<T> CreateIntraPredLumaBuffers(int nt, int dst_strd_mul = 1,
                                                   int bit_depth = 8,
                                                   uint32_t seed = 42) {
  return CreateIntraPredBuffers<T>(nt, /*components=*/1, dst_strd_mul,
                                   bit_depth, seed);
}

// Creates and initializes Chroma intra prediction buffers
template <typename T>
IntraPredChromaBuffers<T> CreateIntraPredChromaBuffers(int nt,
                                                       int dst_strd_mul = 1,
                                                       int bit_depth = 8,
                                                       uint32_t seed = 42) {
  return CreateIntraPredBuffers<T>(nt, /*components=*/2, dst_strd_mul,
                                   bit_depth, seed);
}

// Intra stages
enum class IntraPredOp { kRefSubst, kRefFilter, kPred };

enum class IntraRefFill {
  kRandom,
  kMax,              // every sample at the max value, for overflow in the SIMD sums
  kStrongBoth,       // both strong smoothing checks pass: strong smoothing is applied
  kStrongLeftOnly,   // only the left check passes: normal smoothing
  kStrongTopOnly,    // only the top check passes: normal smoothing
  kThresholdBelow,   // both checks one below the threshold: strong smoothing is applied
  kThresholdAt,      // both checks at the threshold: normal smoothing
};

// Configuration for intra prediction test buffers
struct IntraPredTestConfig {
  IntraPredOp op;
  int nt;
  int mode;               // intra mode
  int flag;               // filter: intra_smoothing_flags (bit 3: disabled, bit 0: strong)
                          // pred: disable_boundary_filter, used by luma modes 10 and 26 only
  IntraRefFill fill;
  int chroma_format_idc;
  int nbr_flags;
  int bit_depth;
  bool is_chroma;
};

// Buffers structure for intra prediction tests
template <typename T>
struct IntraPredTestBuffers {
  // SIMD reads before the reference samples (16 kept free by the decoder) and after them
  // (up to 11 on x86)
  static constexpr int kRefOffset = 16;
  static constexpr int kRefPadAfter = 16;
  std::vector<T> ref;
  std::vector<T> pic;
  WORD32 pic_strd;
  int pic_offset;
  int components;

  T* ref_ptr() { return ref.data() + kRefOffset; }
  T* blk_ptr() { return pic.data() + pic_offset; }

  // Neighbours of the block
  T* top() { return blk_ptr() - pic_strd; }
  T* top_left() { return top() - components; }
  T* left() { return blk_ptr() - components; }
};

// Fills the 4 * nt + 1 reference samples of one component
template <typename T>
void FillIntraRefSamples(T* ref, int step, int nt, IntraRefFill fill, int bit_depth,
                         std::mt19937& rng) {
  const int max_val = (1 << bit_depth) - 1;
  const int threshold = 1 << (bit_depth - 5);
  std::uniform_int_distribution<int> dist(0, max_val);
  auto p = [&](int i) { return static_cast<int>(ref[i * step]); };
  auto set = [&](int i, int val) { ref[i * step] = static_cast<T>(val); };
  // Moves a sample by the threshold, so the check through it fails by at least the threshold
  auto away = [&](int val) { return (val >= threshold) ? val - threshold : val + threshold; };

  for (int i = 0; i <= 4 * nt; i++) {
    set(i, (fill == IntraRefFill::kMax) ? max_val : dist(rng));
  }

  switch (fill) {
    case IntraRefFill::kRandom:
    case IntraRefFill::kMax:
      break;
    case IntraRefFill::kStrongBoth:
    case IntraRefFill::kStrongLeftOnly:
    case IntraRefFill::kStrongTopOnly:
      // The middle sample of each side halfway between its ends: the check gives 0 or 1
      set(nt, (p(0) + p(2 * nt)) / 2);
      set(3 * nt, (p(2 * nt) + p(4 * nt)) / 2);
      if (fill == IntraRefFill::kStrongLeftOnly) set(3 * nt, away(p(3 * nt)));
      if (fill == IntraRefFill::kStrongTopOnly) set(nt, away(p(nt)));
      break;
    case IntraRefFill::kThresholdBelow:
    case IntraRefFill::kThresholdAt: {
      // Ends at base + delta and the other checked samples at base: both checks give delta
      const int delta = (fill == IntraRefFill::kThresholdAt) ? threshold : threshold - 1;
      const int base = max_val / 2;
      set(0, base + delta);
      set(nt, base);
      set(2 * nt, base);
      set(3 * nt, base);
      set(4 * nt, base + delta);
      break;
    }
  }
}

// Creates and initializes intra prediction test buffers
template <typename T>
IntraPredTestBuffers<T> CreateIntraPredTestBuffers(const IntraPredTestConfig& cfg,
                                                   uint32_t seed = 42) {
  const int pad = 16;
  const int components = cfg.is_chroma ? 2 : 1;
  const T max_val = static_cast<T>((1 << cfg.bit_depth) - 1);
  IntraPredTestBuffers<T> b;
  b.components = components;

  b.ref.resize(b.kRefOffset + components * (4 * cfg.nt + 1) + b.kRefPadAfter);
  FillRandom(b.ref, static_cast<T>(0), max_val, seed);

  switch (cfg.op) {
    case IntraPredOp::kRefSubst:
      // Row 0 holds the top-left, top and top-right neighbours. The left and bottom-left
      // neighbours are one position before the block in the next 2 * nt rows.
      b.pic_strd = pad + components * 2 * cfg.nt + pad;
      b.pic_offset = b.pic_strd + pad;
      b.pic.resize(b.pic_strd * (2 * cfg.nt + 1));
      break;
    case IntraPredOp::kRefFilter:
      b.pic_strd = 0;
      b.pic_offset = 0;
      break;
    case IntraPredOp::kPred:
      b.pic_strd = components * cfg.nt + pad;
      b.pic_offset = 0;
      b.pic.resize(b.pic_strd * cfg.nt);
      break;
  }
  FillRandom(b.pic, static_cast<T>(0), max_val, seed + 1);

  if (cfg.op != IntraPredOp::kRefSubst) {
    std::mt19937 rng(seed + 2);
    for (int c = 0; c < components; c++) {
      FillIntraRefSamples(b.ref_ptr() + c, components, cfg.nt, cfg.fill, cfg.bit_depth, rng);
    }
  }
  return b;
}

// Function pointer signatures
typedef void (*LumaIntraPredFn)(UWORD8* pu1_ref, WORD32 src_strd,
                                UWORD8* pu1_dst, WORD32 dst_strd, WORD32 nt,
                                WORD32 mode_or_flag);

typedef void (*ChromaIntraPredFn)(UWORD8* pu1_ref, WORD32 src_strd,
                                  UWORD8* pu1_dst, WORD32 dst_strd, WORD32 nt,
                                  WORD32 mode);

typedef void (*HbdLumaIntraPredFn)(UWORD16* pu2_ref, WORD32 src_strd,
                                   UWORD16* pu2_dst, WORD32 dst_strd, WORD32 nt,
                                   WORD32 mode_or_flag, UWORD8 bit_depth);

typedef void (*HbdChromaIntraPredFn)(UWORD16* pu2_ref, WORD32 src_strd,
                                     UWORD16* pu2_dst, WORD32 dst_strd,
                                     WORD32 nt, WORD32 mode);

// Function pointer signatures for reference substitution and filtering
typedef ihevc_intra_pred_luma_ref_substitution_ft* LumaIntraPredRefSubstFn;
typedef ihevc_hbd_intra_pred_luma_ref_substitution_ft* HbdLumaIntraPredRefSubstFn;

typedef ihevc_intra_pred_chroma_ref_substitution_ft* ChromaIntraPredRefSubstFn;
typedef ihevc_hbd_intra_pred_chroma_ref_substitution_ft* HbdChromaIntraPredRefSubstFn;

typedef ihevc_intra_pred_ref_filtering_ft* IntraPredRefFilterFn;
typedef ihevc_hbd_intra_pred_ref_filtering_ft* HbdIntraPredRefFilterFn;

typedef ihevc_intra_pred_chroma_ref_filtering_ft* ChromaIntraPredRefFilterFn;
typedef ihevc_hbd_intra_pred_chroma_ref_filtering_ft* HbdChromaIntraPredRefFilterFn;

// Function getters for 8-bit
LumaIntraPredFn GetLumaIntraPredFn(const ihevc_func_selector_t* selector,
                                   int mode);
ChromaIntraPredFn GetChromaIntraPredFn(const ihevc_func_selector_t* selector,
                                       int mode);
LumaIntraPredRefSubstFn GetLumaIntraPredRefSubstFn(const ihevc_func_selector_t* selector);
ChromaIntraPredRefSubstFn GetChromaIntraPredRefSubstFn(const ihevc_func_selector_t* selector);
IntraPredRefFilterFn GetIntraPredRefFilterFn(const ihevc_func_selector_t* selector);
ChromaIntraPredRefFilterFn GetChromaIntraPredRefFilterFn(const ihevc_func_selector_t* selector);

// Function getters for HBD
HbdLumaIntraPredFn GetHbdLumaIntraPredFn(const ihevc_func_selector_t* selector, int mode);
HbdChromaIntraPredFn GetHbdChromaIntraPredFn(const ihevc_func_selector_t* selector, int mode);
HbdLumaIntraPredRefSubstFn GetHbdLumaIntraPredRefSubstFn(const ihevc_func_selector_t* selector);
HbdChromaIntraPredRefSubstFn GetHbdChromaIntraPredRefSubstFn(
    const ihevc_func_selector_t* selector);
HbdIntraPredRefFilterFn GetHbdIntraPredRefFilterFn(const ihevc_func_selector_t* selector);
HbdChromaIntraPredRefFilterFn GetHbdChromaIntraPredRefFilterFn(
    const ihevc_func_selector_t* selector);

HbdLumaIntraPredFn GetHbdLumaIntraPredFn(IV_ARCH_T arch, int mode);
HbdChromaIntraPredFn GetHbdChromaIntraPredFn(IV_ARCH_T arch, int mode);

struct IntraPredModeInfo {
  int mode;
  const char* name;
};

// Benchmark modes and block sizes
const std::vector<IntraPredModeInfo>& GetIntraPredBenchmarkModes();
const std::vector<int>& GetIntraPredBenchmarkSizes();
