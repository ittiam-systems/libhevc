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

#include <ios>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_intra_pred_utils.h"
#include "ihevc_typedefs.h"
// clang-format on

namespace {

// ---------------------------- Test Param -----------------------------------

const char* GetOpStr(IntraPredOp op) {
  switch (op) {
    case IntraPredOp::kRefSubst: return "subst";
    case IntraPredOp::kRefFilter: return "filter";
    case IntraPredOp::kPred: return "pred";
    default: return "unknown";
  }
}

const char* GetFillStr(IntraRefFill fill) {
  switch (fill) {
    case IntraRefFill::kRandom: return "random";
    case IntraRefFill::kMax: return "max";
    case IntraRefFill::kStrongBoth: return "strong_both";
    case IntraRefFill::kStrongLeftOnly: return "strong_left";
    case IntraRefFill::kStrongTopOnly: return "strong_top";
    case IntraRefFill::kThresholdBelow: return "thresh_below";
    case IntraRefFill::kThresholdAt: return "thresh_at";
    default: return "unknown";
  }
}

const char* GetChromaFormatStr(int chroma_format_idc) {
  switch (chroma_format_idc) {
    case CHROMA_FMT_IDC_YUV420: return "420";
    case CHROMA_FMT_IDC_YUV422: return "422";
    case CHROMA_FMT_IDC_YUV444: return "444";
    default: return "unknown";
  }
}

// Param: op, nt, mode, flag, fill, chroma_format_idc (0 when unused), bit_depth, arch.
using IntraPredTestParam =
    std::tuple<IntraPredOp, int, int, int, IntraRefFill, int, int, IV_ARCH_T>;

std::string PrintIntraPredTestParam(const testing::TestParamInfo<IntraPredTestParam>& info) {
  IntraPredOp op;
  int nt, mode, flag, chroma_format_idc, bit_depth;
  IntraRefFill fill;
  IV_ARCH_T arch;
  std::tie(op, nt, mode, flag, fill, chroma_format_idc, bit_depth, arch) = info.param;
  std::string name = std::string(GetOpStr(op)) + "_nt_" + std::to_string(nt);
  switch (op) {
    case IntraPredOp::kRefSubst:
      if (chroma_format_idc) name += std::string("_") + GetChromaFormatStr(chroma_format_idc);
      break;
    case IntraPredOp::kRefFilter:
      name += "_mode_" + std::to_string(mode) + "_flags_" + std::to_string(flag) + "_" +
              GetFillStr(fill);
      break;
    case IntraPredOp::kPred:
      name += "_mode_" + std::to_string(mode) + (flag ? "_no_bf" : "") + "_" + GetFillStr(fill);
      break;
  }
  return name + "_bd_" + std::to_string(bit_depth) + "_" + get_arch_str(arch);
}

// Generate Nbr flags
std::vector<int> GetNbrFlagsList(const IntraPredTestConfig& cfg) {
  // Flag bits used per side follow the luma size, which is 2 * nt for 4:2:0 and 4:2:2 chroma
  const int luma_nt =
      (cfg.is_chroma && cfg.chroma_format_idc != CHROMA_FMT_IDC_YUV444) ? 2 * cfg.nt : cfg.nt;
  const int bits = (luma_nt <= 8) ? 1 : (luma_nt == 16) ? 2 : 4;

  std::vector<int> side_flags;  // first bit is the part next to the top-left
  for (int f = 0; f < 16; f++) {
    const bool read_bits_only = (f >> bits) == 0;
    const bool first_bits = (f & (f + 1)) == 0;  // 0, 1, 3, 7 or 15
    if (read_bits_only || first_bits) side_flags.push_back(f);
  }
  // Left and bottom-left are stored bit-reversed
  auto reverse = [](int f) {
    return ((f & 1) << 3) | ((f & 2) << 1) | ((f & 4) >> 1) | ((f & 8) >> 3);
  };

  std::vector<int> list;
  for (int top_left = 0; top_left < 2; top_left++) {
    for (int top_right : side_flags) {
      for (int top : side_flags) {
        for (int left : side_flags) {
          for (int bot_left : side_flags) {
            list.push_back((top_left << 16) | (top_right << 12) | (top << 8) |
                           (reverse(left) << 4) | reverse(bot_left));
          }
        }
      }
    }
  }
  return list;
}

// ---------------------------- Base Class -----------------------------------

class IntraPredTestBase : public ::testing::TestWithParam<IntraPredTestParam> {
 protected:
  explicit IntraPredTestBase(bool is_chroma) { cfg.is_chroma = is_chroma; }

  void SetUp() override {
    std::tie(cfg.op, cfg.nt, cfg.mode, cfg.flag, cfg.fill, cfg.chroma_format_idc,
             cfg.bit_depth, arch) = GetParam();
    cfg.nbr_flags = 0;

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // --------------------------- Overloads for Invoke ---------------------------

  // Substitution
  void Invoke(LumaIntraPredRefSubstFn fn, IntraPredTestBuffers<UWORD8>& buf) {
    fn(buf.top_left(), buf.top(), buf.left(), buf.pic_strd, cfg.nt, cfg.nbr_flags, buf.ref_ptr(),
       1);
  }

  void Invoke(HbdLumaIntraPredRefSubstFn fn, IntraPredTestBuffers<UWORD16>& buf) {
    fn(buf.top_left(), buf.top(), buf.left(), buf.pic_strd, cfg.nt, cfg.nbr_flags, buf.ref_ptr(),
       1, static_cast<UWORD8>(cfg.bit_depth));
  }

  void Invoke(ChromaIntraPredRefSubstFn fn, IntraPredTestBuffers<UWORD8>& buf) {
    fn(buf.top_left(), buf.top(), buf.left(), buf.pic_strd, cfg.nt, cfg.nbr_flags, buf.ref_ptr(),
       1, cfg.chroma_format_idc);
  }

  void Invoke(HbdChromaIntraPredRefSubstFn fn, IntraPredTestBuffers<UWORD16>& buf) {
    fn(buf.top_left(), buf.top(), buf.left(), buf.pic_strd, cfg.nt, cfg.nbr_flags, buf.ref_ptr(),
       1, cfg.chroma_format_idc, static_cast<UWORD8>(cfg.bit_depth));
  }

  // Filtering, in place.
  void Invoke(IntraPredRefFilterFn fn, IntraPredTestBuffers<UWORD8>& buf) {
    fn(buf.ref_ptr(), cfg.nt, buf.ref_ptr(), cfg.mode, cfg.flag);
  }

  void Invoke(HbdIntraPredRefFilterFn fn, IntraPredTestBuffers<UWORD16>& buf) {
    fn(buf.ref_ptr(), cfg.nt, buf.ref_ptr(), cfg.mode, cfg.flag,
       static_cast<UWORD8>(cfg.bit_depth));
  }

  void Invoke(HbdChromaIntraPredRefFilterFn fn, IntraPredTestBuffers<UWORD16>& buf) {
    fn(buf.ref_ptr(), cfg.nt, buf.ref_ptr(), cfg.mode, cfg.flag);
  }

  // Prediction: reference samples -> block in pic. Luma and chroma 8-bit take the same arguments.
  void Invoke(LumaIntraPredFn fn, IntraPredTestBuffers<UWORD8>& buf) {
    fn(buf.ref_ptr(), 1, buf.blk_ptr(), buf.pic_strd, cfg.nt, GetPredArg());
  }

  void Invoke(HbdLumaIntraPredFn fn, IntraPredTestBuffers<UWORD16>& buf) {
    fn(buf.ref_ptr(), 1, buf.blk_ptr(), buf.pic_strd, cfg.nt, GetPredArg(),
       static_cast<UWORD8>(cfg.bit_depth));
  }

  void Invoke(HbdChromaIntraPredFn fn, IntraPredTestBuffers<UWORD16>& buf) {
    fn(buf.ref_ptr(), 1, buf.blk_ptr(), buf.pic_strd, cfg.nt, cfg.mode);
  }

  // Luma horizontal and vertical take disable_boundary_filter in place of the mode
  int GetPredArg() const {
    const bool hor_ver = !cfg.is_chroma && (cfg.mode == 10 || cfg.mode == 26);
    return hor_ver ? cfg.flag : cfg.mode;
  }

  // Unified runner for every intra stage
  template <typename PixelType, typename FuncPtr>
  void RunImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    IntraPredTestBuffers<PixelType> ref_buf = CreateIntraPredTestBuffers<PixelType>(cfg);
    IntraPredTestBuffers<PixelType> tst_buf = ref_buf;
    const std::vector<PixelType> init_ref = ref_buf.ref;

    // Substitution runs every neighbour availability, the other stages run once
    const std::vector<int> nbr_flags_list =
        (cfg.op == IntraPredOp::kRefSubst) ? GetNbrFlagsList(cfg) : std::vector<int>{0};
    for (int nbr_flags : nbr_flags_list) {
      cfg.nbr_flags = nbr_flags;
      ref_buf.ref = init_ref;
      tst_buf.ref = init_ref;

      Invoke(ref_fn, ref_buf);
      Invoke(tst_fn, tst_buf);

      // Compare all of ref, so writes outside the reference samples are caught
      ASSERT_EQ(ref_buf.ref, tst_buf.ref) << "nbr_flags 0x" << std::hex << nbr_flags;
    }

    // Compare all of pic: pred writes only the block, substitution only reads
    ASSERT_EQ(ref_buf.pic, tst_buf.pic);
  }

  IntraPredTestConfig cfg;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// ---------------------------- Luma Class -----------------------------------

class IntraPredLumaTest : public IntraPredTestBase {
 protected:
  IntraPredLumaTest() : IntraPredTestBase(false) {}

  void Run() {
    switch (cfg.op) {
      case IntraPredOp::kRefSubst:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(GetLumaIntraPredRefSubstFn(ref), GetLumaIntraPredRefSubstFn(tst));
        } else {
          RunImpl<UWORD16>(GetHbdLumaIntraPredRefSubstFn(ref), GetHbdLumaIntraPredRefSubstFn(tst));
        }
        break;
      case IntraPredOp::kRefFilter:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(GetIntraPredRefFilterFn(ref), GetIntraPredRefFilterFn(tst));
        } else {
          RunImpl<UWORD16>(GetHbdIntraPredRefFilterFn(ref), GetHbdIntraPredRefFilterFn(tst));
        }
        break;
      case IntraPredOp::kPred:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(GetLumaIntraPredFn(ref, cfg.mode), GetLumaIntraPredFn(tst, cfg.mode));
        } else {
          RunImpl<UWORD16>(GetHbdLumaIntraPredFn(ref, cfg.mode),
                           GetHbdLumaIntraPredFn(tst, cfg.mode));
        }
        break;
    }
  }
};

// ---------------------------- Chroma Class ---------------------------------

class IntraPredChromaTest : public IntraPredTestBase {
 protected:
  IntraPredChromaTest() : IntraPredTestBase(true) {}

  void Run() {
    switch (cfg.op) {
      case IntraPredOp::kRefSubst:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(GetChromaIntraPredRefSubstFn(ref), GetChromaIntraPredRefSubstFn(tst));
        } else {
          RunImpl<UWORD16>(GetHbdChromaIntraPredRefSubstFn(ref),
                           GetHbdChromaIntraPredRefSubstFn(tst));
        }
        break;
      case IntraPredOp::kRefFilter:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(GetChromaIntraPredRefFilterFn(ref), GetChromaIntraPredRefFilterFn(tst));
        } else {
          RunImpl<UWORD16>(GetHbdChromaIntraPredRefFilterFn(ref),
                           GetHbdChromaIntraPredRefFilterFn(tst));
        }
        break;
      case IntraPredOp::kPred:
        if (cfg.bit_depth == 8) {
          RunImpl<UWORD8>(GetChromaIntraPredFn(ref, cfg.mode),
                          GetChromaIntraPredFn(tst, cfg.mode));
        } else {
          RunImpl<UWORD16>(GetHbdChromaIntraPredFn(ref, cfg.mode),
                           GetHbdChromaIntraPredFn(tst, cfg.mode));
        }
        break;
    }
  }
};

// ---------------------------- Test cases -----------------------------------

TEST_P(IntraPredLumaTest, Run) { Run(); }

TEST_P(IntraPredChromaTest, Run) { Run(); }

// ---------------------------- Instantiation --------------------------------

// Block sizes per chroma format: chroma blocks go up to 16 in 4:2:0 and 4:2:2, and up to 32 in
// 4:4:4. Luma uses chroma_format_idc 0.
std::vector<std::pair<int, int>> GetFormatSizes(bool is_chroma) {
  if (!is_chroma) return {{0, 4}, {0, 8}, {0, 16}, {0, 32}};
  return {{CHROMA_FMT_IDC_YUV420, 4}, {CHROMA_FMT_IDC_YUV420, 8}, {CHROMA_FMT_IDC_YUV420, 16},
          {CHROMA_FMT_IDC_YUV422, 4}, {CHROMA_FMT_IDC_YUV422, 8}, {CHROMA_FMT_IDC_YUV422, 16},
          {CHROMA_FMT_IDC_YUV444, 4}, {CHROMA_FMT_IDC_YUV444, 8}, {CHROMA_FMT_IDC_YUV444, 16},
          {CHROMA_FMT_IDC_YUV444, 32}};
}

std::vector<IntraPredTestParam> GenerateIntraPredTestParams(bool is_chroma) {
  const int sizes[] = {4, 8, 16, 32};
  const int bit_depths[] = {8, 10};
  // intra_smoothing_flags as the decoder builds them: (disabled << 3) | strong
  const int smoothing_flags[] = {0, 1, 8, 9};
  const IntraRefFill filter_fills[] = {IntraRefFill::kRandom, IntraRefFill::kMax,
                                       IntraRefFill::kStrongBoth, IntraRefFill::kStrongLeftOnly,
                                       IntraRefFill::kStrongTopOnly, IntraRefFill::kThresholdBelow,
                                       IntraRefFill::kThresholdAt};
  const IntraRefFill pred_fills[] = {IntraRefFill::kRandom, IntraRefFill::kMax};
  // Placeholder for the fill slot of substitution params
  const IntraRefFill unused_fill = IntraRefFill::kRandom;

  std::vector<IntraPredTestParam> params;
  for (int bit_depth : bit_depths) {
    for (auto arch : getTstArch()) {
      // Substitution: the format changes how chroma reads the flags
      for (const auto& fs : GetFormatSizes(is_chroma)) {
        params.emplace_back(IntraPredOp::kRefSubst, fs.second, 0, 0, unused_fill, fs.first,
                            bit_depth, arch);
      }

      // Filtering: the decoder filters chroma only in 4:4:4
      const int filter_fmt = is_chroma ? CHROMA_FMT_IDC_YUV444 : 0;
      for (int nt : sizes) {
        for (int mode = 0; mode < 35; ++mode) {
          for (int flags : smoothing_flags) {
            for (IntraRefFill fill : filter_fills) {
              params.emplace_back(IntraPredOp::kRefFilter, nt, mode, flags, fill, filter_fmt,
                                  bit_depth, arch);
            }
          }
        }
      }

      // Prediction: luma horizontal and vertical also run without the boundary filter
      for (int nt : sizes) {
        for (int mode = 0; mode < 35; ++mode) {
          const bool hor_ver = !is_chroma && (mode == 10 || mode == 26);
          for (int flag = 0; flag < (hor_ver ? 2 : 1); ++flag) {
            for (IntraRefFill fill : pred_fills) {
              params.emplace_back(IntraPredOp::kPred, nt, mode, flag, fill, 0, bit_depth, arch);
            }
          }
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(IntraPred, IntraPredLumaTest,
                         ::testing::ValuesIn(GenerateIntraPredTestParams(false)),
                         PrintIntraPredTestParam);
INSTANTIATE_TEST_SUITE_P(IntraPred, IntraPredChromaTest,
                         ::testing::ValuesIn(GenerateIntraPredTestParams(true)),
                         PrintIntraPredTestParam);
}  // namespace
