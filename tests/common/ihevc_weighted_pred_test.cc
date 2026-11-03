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

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_macros.h"
#include "ihevc_typedefs.h"
#include "ihevc_weighted_pred_utils.h"
// clang-format on

namespace {

// ---------------------------- Test Param -----------------------------------

const char* GetOpStr(WeightedPredOp op) {
  switch (op) {
    case WeightedPredOp::kUni: return "uni";
    case WeightedPredOp::kBi: return "bi";
    case WeightedPredOp::kBiDefault: return "bi_default";
    default: return "unknown";
  }
}

const char* GetWeightCaseStr(WeightCase weight_case) {
  switch (weight_case) {
    case WeightCase::kDefault: return "default";
    case WeightCase::kMin: return "min";
    case WeightCase::kMax: return "max";
    case WeightCase::kZero: return "zero";
    case WeightCase::kMixed: return "mixed";
    case WeightCase::kRandom: return "random";
    default: return "unknown";
  }
}

// Param: op, block size (wd, ht), src stride multiplier, dst stride multiplier, bit_depth, arch
using WeightedPredTestParam =
    std::tuple<WeightedPredOp, std::pair<int, int>, int, int, int, IV_ARCH_T>;

std::string PrintWeightedPredTestParam(const testing::TestParamInfo<WeightedPredTestParam>& info) {
  WeightedPredOp op;
  std::pair<int, int> block_size;
  int src_strd_mul, dst_strd_mul, bit_depth;
  IV_ARCH_T arch;
  std::tie(op, block_size, src_strd_mul, dst_strd_mul, bit_depth, arch) = info.param;
  return std::string(GetOpStr(op)) + "_" + std::to_string(block_size.first) + "x" +
         std::to_string(block_size.second) + "_src_strd_mul_" + std::to_string(src_strd_mul) +
         "_dst_strd_mul_" + std::to_string(dst_strd_mul) + "_bd_" + std::to_string(bit_depth) +
         "_" + get_arch_str(arch);
}

// ---------------------------- Base Class -----------------------------------

class WeightedPredTestBase : public ::testing::TestWithParam<WeightedPredTestParam> {
 protected:
  explicit WeightedPredTestBase(bool is_chroma) { cfg.is_chroma = is_chroma; }

  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(cfg.op, block_size, cfg.src_strd_mul, cfg.dst_strd_mul, cfg.bit_depth, arch) =
        GetParam();
    std::tie(cfg.wd, cfg.ht) = block_size;

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // --------------------------- Overloads for Invoke ---------------------------

  void Invoke(WeightedPredUniFn fn, WeightedPredBuffers<UWORD8>& buf) {
    fn(buf.src1.data(), buf.dst.data(), buf.src_strd, buf.dst_strd, buf.wgt0[0], buf.off0[0],
       buf.shift, cfg.lvl_shift1, cfg.ht, cfg.wd);
  }

  void Invoke(WeightedPredBiFn fn, WeightedPredBuffers<UWORD8>& buf) {
    fn(buf.src1.data(), buf.src2.data(), buf.dst.data(), buf.src_strd, buf.src_strd, buf.dst_strd,
       buf.wgt0[0], buf.off0[0], buf.wgt1[0], buf.off1[0], buf.shift, cfg.lvl_shift1,
       cfg.lvl_shift2, cfg.ht, cfg.wd);
  }

  // Luma and chroma bi_default share this pointer type
  void Invoke(WeightedPredBiDefaultFn fn, WeightedPredBuffers<UWORD8>& buf) {
    fn(buf.src1.data(), buf.src2.data(), buf.dst.data(), buf.src_strd, buf.src_strd, buf.dst_strd,
       cfg.lvl_shift1, cfg.lvl_shift2, cfg.ht, cfg.wd);
  }

  void Invoke(ChromaWeightedPredUniFn fn, WeightedPredBuffers<UWORD8>& buf) {
    fn(buf.src1.data(), buf.dst.data(), buf.src_strd, buf.dst_strd, buf.wgt0[0], buf.wgt0[1],
       buf.off0[0], buf.off0[1], buf.shift, cfg.lvl_shift1, cfg.ht, cfg.wd);
  }

  void Invoke(ChromaWeightedPredBiFn fn, WeightedPredBuffers<UWORD8>& buf) {
    fn(buf.src1.data(), buf.src2.data(), buf.dst.data(), buf.src_strd, buf.src_strd, buf.dst_strd,
       buf.wgt0[0], buf.wgt0[1], buf.off0[0], buf.off0[1], buf.wgt1[0], buf.wgt1[1], buf.off1[0],
       buf.off1[1], buf.shift, cfg.lvl_shift1, cfg.lvl_shift2, cfg.ht, cfg.wd);
  }

  void Invoke(HbdWeightedPredUniFn fn, WeightedPredBuffers<UWORD16>& buf) {
    fn(buf.src1.data(), buf.dst.data(), buf.src_strd, buf.dst_strd, buf.wgt0[0], buf.off0[0],
       buf.shift, cfg.lvl_shift1, cfg.ht, cfg.wd, static_cast<UWORD8>(cfg.bit_depth));
  }

  void Invoke(HbdWeightedPredBiFn fn, WeightedPredBuffers<UWORD16>& buf) {
    fn(buf.src1.data(), buf.src2.data(), buf.dst.data(), buf.src_strd, buf.src_strd, buf.dst_strd,
       buf.wgt0[0], buf.off0[0], buf.wgt1[0], buf.off1[0], buf.shift, cfg.lvl_shift1,
       cfg.lvl_shift2, cfg.ht, cfg.wd, static_cast<UWORD8>(cfg.bit_depth));
  }

  // Luma and chroma bi_default share this pointer type
  void Invoke(HbdWeightedPredBiDefaultFn fn, WeightedPredBuffers<UWORD16>& buf) {
    fn(buf.src1.data(), buf.src2.data(), buf.dst.data(), buf.src_strd, buf.src_strd, buf.dst_strd,
       cfg.lvl_shift1, cfg.lvl_shift2, cfg.ht, cfg.wd, static_cast<UWORD8>(cfg.bit_depth));
  }

  void Invoke(HbdChromaWeightedPredUniFn fn, WeightedPredBuffers<UWORD16>& buf) {
    fn(buf.src1.data(), buf.dst.data(), buf.src_strd, buf.dst_strd, buf.wgt0[0], buf.wgt0[1],
       buf.off0[0], buf.off0[1], buf.shift, cfg.lvl_shift1, cfg.ht, cfg.wd,
       static_cast<UWORD8>(cfg.bit_depth));
  }

  void Invoke(HbdChromaWeightedPredBiFn fn, WeightedPredBuffers<UWORD16>& buf) {
    fn(buf.src1.data(), buf.src2.data(), buf.dst.data(), buf.src_strd, buf.src_strd, buf.dst_strd,
       buf.wgt0[0], buf.wgt0[1], buf.off0[0], buf.off0[1], buf.wgt1[0], buf.wgt1[1], buf.off1[0],
       buf.off1[1], buf.shift, cfg.lvl_shift1, cfg.lvl_shift2, cfg.ht, cfg.wd,
       static_cast<UWORD8>(cfg.bit_depth));
  }

  // Unified runner for luma and chroma weighted prediction
  template <typename PixelType, typename FuncPtr>
  void RunImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    const bool bi_default = (cfg.op == WeightedPredOp::kBiDefault);
    const std::vector<WeightCase> weight_cases = {WeightCase::kDefault, WeightCase::kMin,
                                                  WeightCase::kMax, WeightCase::kZero,
                                                  WeightCase::kMixed, WeightCase::kRandom};
    // bi_default has no weights or shift argument, so one weight case and denominator cover it
    const int num_weight_cases = bi_default ? 1 : static_cast<int>(weight_cases.size());
    const int num_log2_wd = bi_default ? 1 : 8;
    // The decoder sets lvl_shift for luma only; bit 0 is lvl_shift1, bit 1 is lvl_shift2
    const int num_lvl_shifts = cfg.is_chroma ? 1 : (cfg.op == WeightedPredOp::kUni) ? 2 : 4;

    for (int full_range = 0; full_range < 2; ++full_range) {
      for (int lvl_shifts = 0; lvl_shifts < num_lvl_shifts; ++lvl_shifts) {
        cfg.full_range = (full_range != 0);
        cfg.lvl_shift1 = (lvl_shifts & 1) ? (1 << 13) : 0;
        cfg.lvl_shift2 = (lvl_shifts & 2) ? (1 << 13) : 0;
        // Inputs depend only on the range and lvl_shift, so build them once for all weights
        const WeightedPredBuffers<PixelType> init_buf = CreateWeightedPredBuffers<PixelType>(cfg);

        for (int wc = 0; wc < num_weight_cases; ++wc) {
          for (int log2_wd = 0; log2_wd < num_log2_wd; ++log2_wd) {
            cfg.weight_case = weight_cases[wc];
            cfg.log2_wd = log2_wd;
            SCOPED_TRACE(std::string("weight ") + GetWeightCaseStr(cfg.weight_case) +
                         ", full_range " + std::to_string(full_range) + ", lvl_shift1 " +
                         std::to_string(cfg.lvl_shift1) + ", lvl_shift2 " +
                         std::to_string(cfg.lvl_shift2) + ", log2_wd " + std::to_string(log2_wd));

            WeightedPredBuffers<PixelType> ref_buf = init_buf;
            SetWeightedPredWeights(cfg, ref_buf);
            WeightedPredBuffers<PixelType> tst_buf = ref_buf;

            Invoke(ref_fn, ref_buf);
            Invoke(tst_fn, tst_buf);

            // Compare all of dst, so writes outside the block are caught
            ASSERT_EQ(ref_buf.dst, tst_buf.dst);
          }
        }
      }
    }
  }

  WeightedPredConfig cfg;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* tst;
  const ihevc_func_selector_t* ref;
};

// ---------------------------- Luma Class -----------------------------------

class WeightedPredLumaTest : public WeightedPredTestBase {
 protected:
  WeightedPredLumaTest() : WeightedPredTestBase(false) {}

  void Run() {
    switch (cfg.op) {
      case WeightedPredOp::kUni:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(ref->ihevc_weighted_pred_uni_fptr, tst->ihevc_weighted_pred_uni_fptr);
        } else {
          RunImpl<UWORD16>(ref->ihevc_hbd_weighted_pred_uni_fptr,
                           tst->ihevc_hbd_weighted_pred_uni_fptr);
        }
        break;
      case WeightedPredOp::kBi:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(ref->ihevc_weighted_pred_bi_fptr, tst->ihevc_weighted_pred_bi_fptr);
        } else {
          RunImpl<UWORD16>(ref->ihevc_hbd_weighted_pred_bi_fptr,
                           tst->ihevc_hbd_weighted_pred_bi_fptr);
        }
        break;
      case WeightedPredOp::kBiDefault:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(ref->ihevc_weighted_pred_bi_default_fptr,
                          tst->ihevc_weighted_pred_bi_default_fptr);
        } else {
          RunImpl<UWORD16>(ref->ihevc_hbd_weighted_pred_bi_default_fptr,
                           tst->ihevc_hbd_weighted_pred_bi_default_fptr);
        }
        break;
    }
  }
};

// ---------------------------- Chroma Class ---------------------------------

class WeightedPredChromaTest : public WeightedPredTestBase {
 protected:
  WeightedPredChromaTest() : WeightedPredTestBase(true) {}

  void Run() {
    switch (cfg.op) {
      case WeightedPredOp::kUni:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(ref->ihevc_weighted_pred_chroma_uni_fptr,
                          tst->ihevc_weighted_pred_chroma_uni_fptr);
        } else {
          RunImpl<UWORD16>(ref->ihevc_hbd_weighted_pred_chroma_uni_fptr,
                           tst->ihevc_hbd_weighted_pred_chroma_uni_fptr);
        }
        break;
      case WeightedPredOp::kBi:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(ref->ihevc_weighted_pred_chroma_bi_fptr,
                          tst->ihevc_weighted_pred_chroma_bi_fptr);
        } else {
          RunImpl<UWORD16>(ref->ihevc_hbd_weighted_pred_chroma_bi_fptr,
                           tst->ihevc_hbd_weighted_pred_chroma_bi_fptr);
        }
        break;
      case WeightedPredOp::kBiDefault:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(ref->ihevc_weighted_pred_chroma_bi_default_fptr,
                          tst->ihevc_weighted_pred_chroma_bi_default_fptr);
        } else {
          RunImpl<UWORD16>(ref->ihevc_hbd_weighted_pred_chroma_bi_default_fptr,
                           tst->ihevc_hbd_weighted_pred_chroma_bi_default_fptr);
        }
        break;
    }
  }
};

// ---------------------------- Test cases -----------------------------------

TEST_P(WeightedPredLumaTest, Run) { Run(); }

TEST_P(WeightedPredChromaTest, Run) { Run(); }

// ---------------------------- Instantiation --------------------------------

void AddBlockSize(std::vector<std::pair<int, int>>& sizes, int wd, int ht) {
  if (std::find(sizes.begin(), sizes.end(), std::make_pair(wd, ht)) == sizes.end()) {
    sizes.emplace_back(wd, ht);
  }
}

// The decoder also runs chroma through luma bi_default, with U and V interleaved in a row
const std::vector<std::pair<int, int>>& GetLumaBiDefaultBlockSizes() {
  static const std::vector<std::pair<int, int>> kSizes = []() {
    std::vector<std::pair<int, int>> sizes = getLumaPUBlockSizes();
    for (const auto& size : getChromaAllFmtBlockSizes()) {
      AddBlockSize(sizes, 2 * size.first, size.second);
    }
    return sizes;
  }();
  return kSizes;
}

std::vector<WeightedPredTestParam> GenerateWeightedPredTestParams(bool is_chroma) {
  const WeightedPredOp ops[] = {WeightedPredOp::kUni, WeightedPredOp::kBi,
                                WeightedPredOp::kBiDefault};
  const int strd_muls[] = {1, 2};
  const int bit_depths[] = {8, 10};

  std::vector<WeightedPredTestParam> params;
  for (WeightedPredOp op : ops) {
    const bool bi_default = (op == WeightedPredOp::kBiDefault);
    const auto& sizes = is_chroma    ? getChromaAllFmtBlockSizes()
                        : bi_default ? GetLumaBiDefaultBlockSizes()
                                     : getLumaPUBlockSizes();
    for (const auto& size : sizes) {
      for (int src_strd_mul : strd_muls) {
        for (int dst_strd_mul : strd_muls) {
          for (int bit_depth : bit_depths) {
            for (auto arch : getTstArch()) {
              params.emplace_back(op, size, src_strd_mul, dst_strd_mul, bit_depth, arch);
            }
          }
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredLumaTest,
                         ::testing::ValuesIn(GenerateWeightedPredTestParams(false)),
                         PrintWeightedPredTestParam);
INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredChromaTest,
                         ::testing::ValuesIn(GenerateWeightedPredTestParams(true)),
                         PrintWeightedPredTestParam);
}  // namespace
