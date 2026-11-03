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

#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevc_inter_pred.h"
#include "ihevc_weighted_pred.h"
#include "ihevc_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "TestCommon.h"

// Luma and chroma bi_default take the same arguments, so they share one pointer type
using WeightedPredUniFn = ihevc_weighted_pred_uni_ft*;
using WeightedPredBiFn = ihevc_weighted_pred_bi_ft*;
using WeightedPredBiDefaultFn = ihevc_weighted_pred_bi_default_ft*;
using ChromaWeightedPredUniFn = ihevc_weighted_pred_chroma_uni_ft*;
using ChromaWeightedPredBiFn = ihevc_weighted_pred_chroma_bi_ft*;
using HbdWeightedPredUniFn = ihevc_hbd_weighted_pred_uni_ft*;
using HbdWeightedPredBiFn = ihevc_hbd_weighted_pred_bi_ft*;
using HbdWeightedPredBiDefaultFn = ihevc_hbd_weighted_pred_bi_default_ft*;
using HbdChromaWeightedPredUniFn = ihevc_hbd_weighted_pred_chroma_uni_ft*;
using HbdChromaWeightedPredBiFn = ihevc_hbd_weighted_pred_chroma_bi_ft*;

enum class WeightedPredOp {
  kUni,        // one list: explicit weight and offset
  kBi,         // two lists: explicit weights and offsets
  kBiDefault,  // two lists: plain average
};

enum class WeightCase {
  kDefault,  // weight 1 << log2_wd and offset 0, the same result as default weighting
  kMin,      // smallest weights and offsets
  kMax,      // largest weights and offsets
  kZero,     // zero weights, so the output is the offset alone
  kMixed,    // list 0 (and Cb) at the max, list 1 (and Cr) at the min
  kRandom,   // random weights and offsets over the legal range
};

// Configuration for weighted prediction buffers
struct WeightedPredConfig {
  int wd;
  int ht;
  int src_strd_mul;
  int dst_strd_mul;
  int bit_depth;
  bool is_chroma;
  WeightedPredOp op;
  WeightCase weight_case = WeightCase::kDefault;
  int log2_wd = 0;
  int lvl_shift1 = 0;
  int lvl_shift2 = 0;
  bool full_range = false;
};

// Buffers and arguments for weighted prediction. Arrays are indexed by plane: 0 for luma or Cb,
// 1 for Cr.
template <typename T>
struct WeightedPredBuffers {
  std::vector<WORD16> src1;
  std::vector<WORD16> src2;
  std::vector<T> dst;
  WORD32 src_strd;
  WORD32 dst_strd;
  WORD32 wgt0[2];
  WORD32 off0[2];
  WORD32 wgt1[2];
  WORD32 off1[2];
  WORD32 shift;
};

// Creates weighted prediction buffers and picks the weights, offsets and shift
template <typename T>
WeightedPredBuffers<T> CreateWeightedPredBuffers(const WeightedPredConfig& cfg);

// Picks the weights, offsets and shift only, so buffers can be reused across weight choices
template <typename T>
void SetWeightedPredWeights(const WeightedPredConfig& cfg, WeightedPredBuffers<T>& b);
