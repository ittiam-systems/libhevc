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
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// clang-format off
#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_itrans_utils.h"
#include "ihevc_macros.h"
#include "ihevc_structs.h"
#include "ihevc_typedefs.h"
// clang-format on

namespace {

// ---------------------------- Test Param -----------------------------------

enum ITransReconVariant {
  kFullRange,   // whole WORD16 range
  kSmallRange,  // realistic coefficients
  kSparse,      // realistic, scattered coefficients
};

const char* GetVariantStr(int variant) {
  switch (variant) {
    case kFullRange: return "full";
    case kSmallRange: return "small";
    case kSparse: return "sparse";
    default: return "unknown";
  }
}

// Param: trans_size, ttype (0: normal, 1: ttype1), arch, non_zero_rows, non_zero_cols, bit_depth,
// variant, plane (chroma: 0 for U, 1 for V)
using ITransReconTestParam = std::tuple<int, int, IV_ARCH_T, int, int, int, int, int>;

std::string PrintITransReconTestParam(const testing::TestParamInfo<ITransReconTestParam>& info) {
  int trans_size, ttype, nz_rows, nz_cols, bit_depth, variant, plane;
  IV_ARCH_T arch;
  std::tie(trans_size, ttype, arch, nz_rows, nz_cols, bit_depth, variant, plane) = info.param;
  return "size_" + std::to_string(trans_size) + "_ttype_" + std::to_string(ttype) + "_nzr_" +
         std::to_string(nz_rows) + "_nzc_" + std::to_string(nz_cols) + "_bd_" +
         std::to_string(bit_depth) + "_" + GetVariantStr(variant) + (plane ? "_v" : "") + "_" +
         get_arch_str(arch);
}

// ---------------------------- Base Class -----------------------------------

class ITransReconTestBase : public ::testing::TestWithParam<ITransReconTestParam> {
 protected:
  explicit ITransReconTestBase(bool is_chroma) { cfg.is_chroma = is_chroma; }

  void SetUp() override {
    int variant;
    std::tie(cfg.trans_size, ttype, arch, cfg.nz_rows, cfg.nz_cols, cfg.bit_depth, variant,
             cfg.plane) = GetParam();
    cfg.sparse = (variant == kSparse);
    cfg.full_range = (variant == kFullRange);

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // --------------------------- Overloads for Invoke ---------------------------

  // Luma and chroma functions take the same arguments, so FuncPtr can be either
  template <typename FuncPtr>
  void Invoke(FuncPtr fn, ITransReconBuffers<UWORD8>& buf) {
    fn(buf.src.data(), buf.tmp.data(), buf.pred_ptr(), buf.dst_ptr(), buf.src_strd, buf.pred_strd,
       buf.dst_strd, buf.zero_cols, buf.zero_rows);
  }

  template <typename FuncPtr>
  void Invoke(FuncPtr fn, ITransReconBuffers<UWORD16>& buf) {
    fn(buf.src.data(), buf.tmp.data(), buf.pred_ptr(), buf.dst_ptr(), buf.src_strd, buf.pred_strd,
       buf.dst_strd, buf.zero_cols, buf.zero_rows, static_cast<UWORD8>(cfg.bit_depth));
  }

  // Unified runner for luma and chroma ITrans Recon
  template <typename PixelType, typename FuncPtr>
  void RunImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    ITransReconBuffers<PixelType> ref_buf = CreateITransReconBuffers<PixelType>(cfg);
    ITransReconBuffers<PixelType> tst_buf = ref_buf;

    Invoke(ref_fn, ref_buf);
    Invoke(tst_fn, tst_buf);

    // Compare all of dst, so writes outside the block are caught
    ASSERT_EQ(ref_buf.dst, tst_buf.dst);
  }

  int ttype;  // luma 4x4 only
  ITransReconConfig cfg;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// ---------------------------- Luma Class -----------------------------------

class ITransReconLumaTest : public ITransReconTestBase {
 protected:
  ITransReconLumaTest() : ITransReconTestBase(false) {}

  void Run() {
    if (cfg.bit_depth == 8) {
      RunImpl<UWORD8>(GetITransReconFn(ref, cfg.trans_size, ttype),
                      GetITransReconFn(tst, cfg.trans_size, ttype));
    } else {
      RunImpl<UWORD16>(GetHbdITransReconFn(ref, cfg.trans_size, ttype),
                       GetHbdITransReconFn(tst, cfg.trans_size, ttype));
    }
  }
};

// ---------------------------- Chroma Class ---------------------------------

class ITransReconChromaTest : public ITransReconTestBase {
 protected:
  ITransReconChromaTest() : ITransReconTestBase(true) {}

  void Run() {
    if (cfg.bit_depth == 8) {
      RunImpl<UWORD8>(GetChromaITransReconFn(ref, cfg.trans_size),
                      GetChromaITransReconFn(tst, cfg.trans_size));
    } else {
      RunImpl<UWORD16>(GetHbdChromaITransReconFn(ref, cfg.trans_size),
                       GetHbdChromaITransReconFn(tst, cfg.trans_size));
    }
  }
};

// ---------------------------- Test cases -----------------------------------

TEST_P(ITransReconLumaTest, Run) { Run(); }

TEST_P(ITransReconChromaTest, Run) { Run(); }

// ---------------------------- Instantiation --------------------------------

std::vector<int> GetNzOptions(int size) {
  switch (size) {
    case 4: return {1, 4};
    case 8: return {4, 5, 8};
    case 16: return {4, 5, 8, 9, 16};
    default: return {4, 5, 8, 9, 32};
  }
}

// num_ttypes: 2 to also cover ttype1 (DST) for 4x4 luma, 1 otherwise
// num_planes: 2 to also cover the V plane for chroma, 1 otherwise
std::vector<ITransReconTestParam> GenerateITransReconTestParams(int num_ttypes, int num_planes) {
  const int sizes[] = {4, 8, 16, 32};
  const int bit_depths[] = {8, 10};
  const int variants[] = {kFullRange, kSmallRange, kSparse};

  std::vector<ITransReconTestParam> params;
  for (int size : sizes) {
    for (int ttype = 0; ttype < ((size == 4) ? num_ttypes : 1); ++ttype) {
      for (int nzr : GetNzOptions(size)) {
        for (int nzc : GetNzOptions(size)) {
          // DC-only DCT blocks never reach itrans_recon: the decoder uses itrans_recon_dc
          if (ttype == 0 && nzr == 1 && nzc == 1) continue;
          for (int bit_depth : bit_depths) {
            for (int variant : variants) {
              for (int plane = 0; plane < num_planes; ++plane) {
                for (auto arch : getTstArch()) {
                  params.emplace_back(size, ttype, arch, nzr, nzc, bit_depth, variant, plane);
                }
              }
            }
          }
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(ITransRecon, ITransReconLumaTest,
                         ::testing::ValuesIn(GenerateITransReconTestParams(2, 1)),
                         PrintITransReconTestParam);
INSTANTIATE_TEST_SUITE_P(ITransRecon, ITransReconChromaTest,
                         ::testing::ValuesIn(GenerateITransReconTestParams(1, 2)),
                         PrintITransReconTestParam);
}  // namespace
