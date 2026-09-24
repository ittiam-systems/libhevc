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

// Function getters for 8-bit
LumaIntraPredFn GetLumaIntraPredFn(const ihevc_func_selector_t* selector,
                                   int mode);
ChromaIntraPredFn GetChromaIntraPredFn(const ihevc_func_selector_t* selector,
                                       int mode);

// Function getters for HBD
HbdLumaIntraPredFn GetHbdLumaIntraPredFn(IV_ARCH_T arch, int mode);
HbdChromaIntraPredFn GetHbdChromaIntraPredFn(IV_ARCH_T arch, int mode);

struct IntraPredModeInfo {
  int mode;
  const char* name;
};

// Benchmark modes and block sizes
const std::vector<IntraPredModeInfo>& GetIntraPredBenchmarkModes();
const std::vector<int>& GetIntraPredBenchmarkSizes();
