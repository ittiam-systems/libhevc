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

#include "ihevc_weighted_pred_utils.h"

#include <random>
#include <vector>

namespace {

void FillWeightedPredSrc(std::vector<WORD16>& src, const WeightedPredConfig& cfg, int lvl_shift,
                         uint32_t seed) {
  if (cfg.full_range) {
    FillRandom(src, static_cast<WORD16>(-32768), static_cast<WORD16>(32767), seed);
    return;
  }
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pixel(0, (1 << cfg.bit_depth) - 1);
  for (WORD16& val : src) {
    val = static_cast<WORD16>((pixel(rng) << (14 - cfg.bit_depth)) - lvl_shift);
  }
}

}  // namespace

template <typename T>
WeightedPredBuffers<T> CreateWeightedPredBuffers(const WeightedPredConfig& cfg) {
  const int row_samples = (cfg.is_chroma ? 2 : 1) * cfg.wd;
  const T max_val = static_cast<T>((1 << cfg.bit_depth) - 1);

  WeightedPredBuffers<T> b;
  b.src_strd = row_samples * cfg.src_strd_mul;
  b.dst_strd = row_samples * cfg.dst_strd_mul;

  // Guard row below the block catches writes and covers SIMD over-reads
  b.src1.resize(b.src_strd * (cfg.ht + 1));
  b.src2.resize(b.src_strd * (cfg.ht + 1));
  b.dst.resize(b.dst_strd * (cfg.ht + 1));
  FillWeightedPredSrc(b.src1, cfg, cfg.lvl_shift1, 1337);
  FillWeightedPredSrc(b.src2, cfg, cfg.lvl_shift2, 1338);
  // Random dst, so stray writes are caught
  FillRandom(b.dst, static_cast<T>(0), max_val, 42);

  SetWeightedPredWeights(cfg, b);
  return b;
}

template <typename T>
void SetWeightedPredWeights(const WeightedPredConfig& cfg, WeightedPredBuffers<T>& b) {
  const int wgt_def = 1 << cfg.log2_wd;
  const int wgt_min = wgt_def - 128;
  const int wgt_max = wgt_def + 127;
  const int off_min = -(1 << (cfg.bit_depth - 1));
  const int off_max = (1 << (cfg.bit_depth - 1)) - 1;

  std::mt19937 rng(cfg.log2_wd);
  std::uniform_int_distribution<int> wgt_dist(wgt_min, wgt_max);
  std::uniform_int_distribution<int> off_dist(off_min, off_max);
  for (int plane = 0; plane < 2; ++plane) {
    // Cr swaps the list 0 and list 1 choices of Cb, so the two planes differ
    const int hi = (plane == 0) ? 0 : 1;
    int wgt[2], off[2];
    switch (cfg.weight_case) {
      case WeightCase::kDefault:
        wgt[0] = wgt[1] = wgt_def;
        off[0] = off[1] = 0;
        break;
      case WeightCase::kMin:
        wgt[0] = wgt[1] = wgt_min;
        off[0] = off[1] = off_min;
        break;
      case WeightCase::kMax:
        wgt[0] = wgt[1] = wgt_max;
        off[0] = off[1] = off_max;
        break;
      case WeightCase::kZero:
        wgt[0] = wgt[1] = 0;
        off[0] = off[1] = (plane == 0) ? off_max : off_min;
        break;
      case WeightCase::kMixed:
        wgt[hi] = wgt_max;
        wgt[1 - hi] = wgt_min;
        off[hi] = off_max;
        off[1 - hi] = off_min;
        break;
      case WeightCase::kRandom:
      default:
        for (int list = 0; list < 2; ++list) {
          wgt[list] = wgt_dist(rng);
          off[list] = off_dist(rng);
        }
        break;
    }
    b.wgt0[plane] = wgt[0];
    b.off0[plane] = off[0];
    b.wgt1[plane] = wgt[1];
    b.off1[plane] = off[1];
  }

  b.shift = cfg.log2_wd + 14 - cfg.bit_depth + ((cfg.op == WeightedPredOp::kBi) ? 1 : 0);
}

// Explicit instantiations for 8-bit and HBD
template WeightedPredBuffers<UWORD8> CreateWeightedPredBuffers<UWORD8>(
    const WeightedPredConfig& cfg);
template WeightedPredBuffers<UWORD16> CreateWeightedPredBuffers<UWORD16>(
    const WeightedPredConfig& cfg);
template void SetWeightedPredWeights<UWORD8>(const WeightedPredConfig& cfg,
                                             WeightedPredBuffers<UWORD8>& b);
template void SetWeightedPredWeights<UWORD16>(const WeightedPredConfig& cfg,
                                              WeightedPredBuffers<UWORD16>& b);
