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

#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevc_sao.h"
#include "ihevc_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"

// Configuration for SAO Band Offset benchmarks
struct SaoBandConfig {
  int wd;
  int ht;
  int sao_band_pos;
  IV_ARCH_T arch;
  int bit_depth = 8;
};

// Configuration for SAO Edge Offset benchmarks
struct SaoEdgeConfig {
  int edge_class;  // 0, 1, 2, 3
  int wd;
  int ht;
  IV_ARCH_T arch;
  int bit_depth = 8;
};

// Buffers structure for SAO Luma operations
template <typename T>
struct SaoLumaBuffers {
  int stride;
  int total_ht;
  int src_size;
  int src_offset;
  std::vector<T> src;
  std::vector<T> src_left;
  std::vector<T> src_top;
  std::vector<T> src_top_left;
  std::vector<T> src_top_right;
  std::vector<T> src_bot_left;
  std::vector<UWORD8> avail;
  std::vector<WORD8> sao_offset;
};

// Buffers structure for SAO Chroma operations
template <typename T>
struct SaoChromaBuffers {
  int stride;
  int total_ht;
  int src_size;
  int src_offset;
  std::vector<T> src;
  std::vector<T> src_left;
  std::vector<T> src_top;
  std::vector<T> src_top_left;
  std::vector<T> src_top_right;
  std::vector<T> src_bot_left;
  std::vector<UWORD8> avail;
  std::vector<WORD8> sao_offset_u;
  std::vector<WORD8> sao_offset_v;
};

// Creates and initializes Luma SAO buffers
template <typename T>
SaoLumaBuffers<T> CreateSaoLumaBuffers(int wd, int ht, int bit_depth = 8) {
  SaoLumaBuffers<T> b;
  b.stride = wd + 32;
  b.total_ht = ht + 32;
  b.src_size = b.stride * b.total_ht;
  b.src_offset = 16 * b.stride + 16;

  b.src.resize(b.src_size);
  b.src_left.resize(ht + 8 + 1);
  b.src_top.resize(wd + 8);
  b.src_top_left.resize(8);
  b.src_top_right.resize(8);
  b.src_bot_left.resize(8);
  b.avail.resize(8);
  b.sao_offset.resize(8, 0);

  const T max_val = static_cast<T>((1 << bit_depth) - 1);
  FillRandom(b.src, static_cast<T>(0), max_val, 42);
  FillRandom(b.src_left, static_cast<T>(0), max_val, 43);
  FillRandom(b.src_top, static_cast<T>(0), max_val, 44);
  FillRandom(b.src_top_left, static_cast<T>(0), max_val, 45);
  FillRandom(b.src_top_right, static_cast<T>(0), max_val, 46);
  FillRandom(b.src_bot_left, static_cast<T>(0), max_val, 47);

  std::mt19937 rng(48);
  std::uniform_int_distribution<uint8_t> dist_avail(0, 1);
  std::uniform_int_distribution<int8_t> dist_offset(-7, 7);
  for (size_t i = 0; i < 8; ++i) {
    b.avail[i] = dist_avail(rng) ? 255 : 0;
  }
  b.src_left[0] = b.src_top_left[0];

  for (int i = 1; i < 5; ++i) {
    b.sao_offset[i] = dist_offset(rng);
  }
  return b;
}

// Creates and initializes Chroma SAO buffers
template <typename T>
SaoChromaBuffers<T> CreateSaoChromaBuffers(int wd, int ht, int bit_depth = 8) {
  SaoChromaBuffers<T> b;
  b.stride = 2 * wd + 32;
  b.total_ht = ht + 32;
  b.src_size = b.stride * b.total_ht;
  b.src_offset = 16 * b.stride + 16;

  b.src.resize(b.src_size);
  b.src_left.resize(2 * ht + 8 + 2);
  b.src_top.resize(2 * wd + 8);
  b.src_top_left.resize(8);
  b.src_top_right.resize(8);
  b.src_bot_left.resize(8);
  b.avail.resize(8);
  b.sao_offset_u.resize(8, 0);
  b.sao_offset_v.resize(8, 0);

  const T max_val = static_cast<T>((1 << bit_depth) - 1);
  FillRandom(b.src, static_cast<T>(0), max_val, 42);
  FillRandom(b.src_left, static_cast<T>(0), max_val, 43);
  FillRandom(b.src_top, static_cast<T>(0), max_val, 44);
  FillRandom(b.src_top_left, static_cast<T>(0), max_val, 45);
  FillRandom(b.src_top_right, static_cast<T>(0), max_val, 46);
  FillRandom(b.src_bot_left, static_cast<T>(0), max_val, 47);

  std::mt19937 rng(48);
  std::uniform_int_distribution<uint8_t> dist_avail(0, 1);
  std::uniform_int_distribution<int8_t> dist_offset(-7, 7);
  for (size_t i = 0; i < 8; ++i) {
    b.avail[i] = dist_avail(rng) ? 255 : 0;
  }
  b.src_left[0] = b.src_top_left[0];
  b.src_left[1] = b.src_top_left[1];

  for (int i = 1; i < 5; ++i) {
    b.sao_offset_u[i] = dist_offset(rng);
    b.sao_offset_v[i] = dist_offset(rng);
  }
  return b;
}

// 8-bit SAO function getters
ihevc_sao_band_offset_luma_ft* GetSaoBandOffsetLumaFn(
    const ihevc_func_selector_t* selector);

ihevc_sao_band_offset_chroma_ft* GetSaoBandOffsetChromaFn(
    const ihevc_func_selector_t* selector);

ihevc_sao_edge_offset_class0_ft* GetSaoEdgeOffsetLumaFn(
    const ihevc_func_selector_t* selector, int edge_class);

ihevc_sao_edge_offset_class0_chroma_ft* GetSaoEdgeOffsetChromaFn(
    const ihevc_func_selector_t* selector, int edge_class);

// HBD SAO function getters
ihevc_hbd_sao_band_offset_luma_ft* GetHbdSaoBandOffsetLumaFn(IV_ARCH_T arch);

ihevc_hbd_sao_band_offset_chroma_ft* GetHbdSaoBandOffsetChromaFn(
    IV_ARCH_T arch);

ihevc_hbd_sao_edge_offset_class0_ft* GetHbdSaoEdgeOffsetLumaFn(IV_ARCH_T arch,
                                                               int edge_class);

ihevc_hbd_sao_edge_offset_class0_chroma_ft* GetHbdSaoEdgeOffsetChromaFn(
    IV_ARCH_T arch, int edge_class);

// Benchmark representative block sizes
inline const std::vector<std::pair<int, int>>& GetSaoLumaBenchmarkSizes() {
  return GetBenchmarkLumaPUSizes();
}

inline const std::vector<std::pair<int, int>>& GetSaoChromaBenchmarkSizes() {
  return GetBenchmarkChromaPUSizes();
}
