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

#include <set>
#include <string>
#include <tuple>
#include <vector>

// clang-format off
#include "func_selector.h"
#include "ihevc_typedefs.h"
#include "TestCommon.h"
#include "ihevc_padding_utils.h"
// clang-format on

namespace {

// ---------------------------- Test Param -----------------------------------

// Param: pad_type, wd, ht, pad_size, bit_depth, arch
using PaddingTestParam = std::tuple<PadType, int, int, int, int, IV_ARCH_T>;

std::string PrintPaddingTestParam(const testing::TestParamInfo<PaddingTestParam>& info) {
  PadType pad_type;
  int wd, ht, pad_size, bit_depth;
  IV_ARCH_T arch;
  std::tie(pad_type, wd, ht, pad_size, bit_depth, arch) = info.param;
  return std::string(GetPadTypeStr(pad_type)) + "_" + std::to_string(wd) + "x" +
         std::to_string(ht) + "_pad_" + std::to_string(pad_size) + "_bd_" +
         std::to_string(bit_depth) + "_" + get_arch_str(arch);
}

// ---------------------------- Base Class -----------------------------------

class PaddingTestBase : public ::testing::TestWithParam<PaddingTestParam> {
 protected:
  explicit PaddingTestBase(bool is_chroma) { cfg.is_chroma = is_chroma; }

  void SetUp() override {
    std::tie(cfg.pad_type, cfg.wd, cfg.ht, cfg.pad_size, cfg.bit_depth, arch) = GetParam();

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // --------------------------- Overloads for Invoke ---------------------------

  template <typename PixelType>
  void Invoke(PadFnT<PixelType> fn, PaddingBuffers<PixelType>& buf) {
    fn(buf.pad_ptr(), buf.strd, buf.len, cfg.pad_size);
  }

  // Horz/vert
  template <typename PixelType>
  void Invoke(PadBlockFnT<PixelType> fn, PaddingBuffers<PixelType>& buf) {
    fn(buf.pad_ptr(), buf.strd, cfg.ht, buf.row_wd, cfg.pad_size);
  }

  // Unified runner for luma and chroma Padding
  template <typename PixelType, typename FuncPtr>
  void RunImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    PaddingBuffers<PixelType> ref_buf = CreatePaddingBuffers<PixelType>(cfg);
    PaddingBuffers<PixelType> tst_buf = ref_buf;

    Invoke(ref_fn, ref_buf);
    Invoke(tst_fn, tst_buf);

    // Compare the whole buffer, so writes outside the padding are caught
    ASSERT_EQ(ref_buf.buf, tst_buf.buf);
  }

  PaddingConfig cfg{};
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// ---------------------------- Luma Class -----------------------------------

class PaddingLumaTest : public PaddingTestBase {
 protected:
  PaddingLumaTest() : PaddingTestBase(false) {}

  void Run() {
    if (IsBlockPad(cfg.pad_type)) {
      if (cfg.bit_depth == 8) {
        RunImpl<UWORD8>(GetPadBlockFn(ref, cfg.pad_type), GetPadBlockFn(tst, cfg.pad_type));
      } else {
        RunImpl<UWORD16>(GetHbdPadBlockFn(ref, cfg.pad_type), GetHbdPadBlockFn(tst, cfg.pad_type));
      }
    } else if (cfg.bit_depth == 8) {
      RunImpl<UWORD8>(GetPadFn(ref, cfg.pad_type), GetPadFn(tst, cfg.pad_type));
    } else {
      RunImpl<UWORD16>(GetHbdPadFn(ref, cfg.pad_type), GetHbdPadFn(tst, cfg.pad_type));
    }
  }
};

// ---------------------------- Chroma Class ---------------------------------

class PaddingChromaTest : public PaddingTestBase {
 protected:
  PaddingChromaTest() : PaddingTestBase(true) {}

  void Run() {
    if (IsBlockPad(cfg.pad_type)) {
      if (cfg.bit_depth == 8) {
        RunImpl<UWORD8>(GetChromaPadBlockFn(ref, cfg.pad_type),
                        GetChromaPadBlockFn(tst, cfg.pad_type));
      } else {
        RunImpl<UWORD16>(GetHbdChromaPadBlockFn(ref, cfg.pad_type),
                         GetHbdChromaPadBlockFn(tst, cfg.pad_type));
      }
    } else if (cfg.bit_depth == 8) {
      RunImpl<UWORD8>(GetChromaPadFn(ref, cfg.pad_type), GetChromaPadFn(tst, cfg.pad_type));
    } else {
      RunImpl<UWORD16>(GetHbdChromaPadFn(ref, cfg.pad_type),
                       GetHbdChromaPadFn(tst, cfg.pad_type));
    }
  }
};

// ---------------------------- Test cases -----------------------------------

TEST_P(PaddingLumaTest, Run) { Run(); }

TEST_P(PaddingChromaTest, Run) { Run(); }

// ---------------------------- Instantiation --------------------------------

// Sizes below follow the padding calls in ihevcd_process_slice.c
constexpr int kPadLeftRight = 80;   // PAD_LEFT / PAD_RIGHT
constexpr int kPadTopBottom = 80;   // PAD_TOP / PAD_BOT
constexpr int kPadWd = 160;         // PAD_WD
// pad_top copies only the first block row and pad_bottom only the last, so 2 rows are enough
constexpr int kTopBottomBlkHt = 2;

struct ChromaFormat {
  int h_samp;
  int v_samp;
};

// 4:2:0, 4:2:2 and 4:4:4
const ChromaFormat kChromaFormats[] = {{2, 2}, {2, 1}, {1, 1}};
// U and V are interleaved, so each chroma pixel is 2 samples
constexpr int kChromaPixelStrd = 2;
const int kCtbSizes[] = {16, 32, 64};
// Picture widths are multiples of the min CU size; these hit every SIMD width tail
const int kPicWidths[] = {8, 16, 24, 40, 64, 88, 96, 1920};
// Left/right ht does not change the block width, so a narrow and a wide block are enough
const int kLeftRightWidths[] = {8, 64};

// Each entry is (pad_type, wd, ht, pad_size); a set drops sizes shared by chroma formats
std::vector<PaddingTestParam> ExpandPaddingTestParams(
    const std::set<std::tuple<PadType, int, int, int>>& sizes) {
  const int bit_depths[] = {8, 10};

  std::vector<PaddingTestParam> params;
  for (const auto& size : sizes) {
    PadType pad_type;
    int wd, ht, pad_size;
    std::tie(pad_type, wd, ht, pad_size) = size;
    for (int bit_depth : bit_depths) {
      for (auto arch : getTstArch()) {
        params.emplace_back(pad_type, wd, ht, pad_size, bit_depth, arch);
      }
    }
  }
  return params;
}

std::vector<PaddingTestParam> GenerateLumaPaddingTestParams() {
  // pad_type, wd, ht, pad_size
  std::set<std::tuple<PadType, int, int, int>> sizes;
  for (int ctb_size : kCtbSizes) {
    // Left/right pad a CTB row, plus 8 more rows at the last CTB row
    for (int ht : {ctb_size, ctb_size + 8}) {
      for (int wd : kLeftRightWidths) {
        sizes.emplace(kPadLeft, wd, ht, kPadLeftRight);
        sizes.emplace(kPadRight, wd, ht, kPadLeftRight);
        sizes.emplace(kPadHorz, wd, ht, kPadLeftRight);
      }
    }
  }
  // Fewer rows than any CTB row, covers SIMD row loops that run only once or twice
  for (int ht : {4, 8}) {
    for (int wd : kLeftRightWidths) {
      sizes.emplace(kPadLeft, wd, ht, kPadLeftRight);
      sizes.emplace(kPadRight, wd, ht, kPadLeftRight);
      sizes.emplace(kPadHorz, wd, ht, kPadLeftRight);
    }
  }
  // Top/bottom replicate a whole row, including the left and right padding
  for (int pic_wd : kPicWidths) {
    sizes.emplace(kPadTop, pic_wd + kPadWd, kTopBottomBlkHt, kPadTopBottom);
    sizes.emplace(kPadBottom, pic_wd + kPadWd, kTopBottomBlkHt, kPadTopBottom);
    sizes.emplace(kPadVert, pic_wd + kPadWd, kTopBottomBlkHt, kPadTopBottom);
  }
  return ExpandPaddingTestParams(sizes);
}

std::vector<PaddingTestParam> GenerateChromaPaddingTestParams() {
  // pad_type, wd, ht, pad_size
  std::set<std::tuple<PadType, int, int, int>> sizes;
  for (const ChromaFormat& fmt : kChromaFormats) {
    // In samples of the interleaved row: 80 for 4:2:0 and 4:2:2, 160 for 4:4:4
    const int pad_left_right = kPadLeftRight * kChromaPixelStrd / fmt.h_samp;
    std::vector<int> heights = {4, 8 * fmt.v_samp};
    for (int ctb_size : kCtbSizes) {
      heights.push_back(ctb_size / fmt.v_samp);
      heights.push_back(ctb_size / fmt.v_samp + 8 * fmt.v_samp);  // last CTB row
    }
    for (int ht : heights) {
      for (int wd : kLeftRightWidths) {
        sizes.emplace(kPadLeft, wd, ht, pad_left_right);
        sizes.emplace(kPadRight, wd, ht, pad_left_right);
        sizes.emplace(kPadHorz, wd, ht, pad_left_right);
      }
    }
    for (int pic_wd : kPicWidths) {
      const int wd = (pic_wd + kPadWd) / fmt.h_samp;
      sizes.emplace(kPadTop, wd, kTopBottomBlkHt, kPadTopBottom / fmt.v_samp);
      sizes.emplace(kPadBottom, wd, kTopBottomBlkHt, kPadTopBottom / fmt.v_samp);
      sizes.emplace(kPadVert, wd, kTopBottomBlkHt, kPadTopBottom / fmt.v_samp);
    }
  }
  return ExpandPaddingTestParams(sizes);
}

INSTANTIATE_TEST_SUITE_P(Padding, PaddingLumaTest,
                         ::testing::ValuesIn(GenerateLumaPaddingTestParams()),
                         PrintPaddingTestParam);
INSTANTIATE_TEST_SUITE_P(Padding, PaddingChromaTest,
                         ::testing::ValuesIn(GenerateChromaPaddingTestParams()),
                         PrintPaddingTestParam);
}  // namespace
