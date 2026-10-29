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

#include <string>
#include <tuple>
#include <vector>

// clang-format off
#include "TestCommon.h"
#include "dec_func_selector.h"
#include "ihevcd_itrans_recon_dc_utils.h"
#include "ihevc_typedefs.h"
// clang-format on

namespace {

// ---------------------------- Test Param -----------------------------------

// Param: trans_size, arch, coeff (DC coefficient value), bit_depth,
// plane (chroma: 0 for U, 1 for V)
using ITransReconDCTestParam = std::tuple<int, IV_ARCH_T, int, int, int>;

std::string PrintITransReconDCTestParam(
    const testing::TestParamInfo<ITransReconDCTestParam>& info) {
  int trans_size, coeff, bit_depth, plane;
  IV_ARCH_T arch;
  std::tie(trans_size, arch, coeff, bit_depth, plane) = info.param;
  const std::string coeff_str =
      (coeff < 0) ? "neg" + std::to_string(-coeff) : std::to_string(coeff);
  return "size_" + std::to_string(trans_size) + "_coeff_" + coeff_str + "_bd_" +
         std::to_string(bit_depth) + (plane ? "_v" : "") + "_" + get_arch_str(arch);
}

// ---------------------------- Base Class -----------------------------------

class ITransReconDCTestBase : public ::testing::TestWithParam<ITransReconDCTestParam> {
 protected:
  explicit ITransReconDCTestBase(bool is_chroma) { cfg.is_chroma = is_chroma; }

  void SetUp() override {
    IV_ARCH_T arch;
    std::tie(cfg.trans_size, arch, coeff, cfg.bit_depth, cfg.plane) = GetParam();
    // The functions take log2 of the size
    log2_trans_size = 0;
    while ((1 << log2_trans_size) < cfg.trans_size) log2_trans_size++;

    ref = get_dec_ref_func_ptr();
    tst = get_dec_tst_func_ptr(arch);
    if (tst == nullptr) {
      GTEST_SKIP() << "No function table for this architecture";
    }
  }

  // --------------------------- Overloads for Invoke ---------------------------

  // Luma and chroma functions take the same arguments, so FuncPtr can be either
  template <typename FuncPtr>
  void Invoke(FuncPtr fn, ITransReconDCBuffers<UWORD8>& buf) {
    fn(buf.pred_ptr(), buf.dst_ptr(), buf.pred_strd, buf.dst_strd, log2_trans_size,
       static_cast<WORD16>(coeff));
  }

  template <typename FuncPtr>
  void Invoke(FuncPtr fn, ITransReconDCBuffers<UWORD16>& buf) {
    fn(buf.pred_ptr(), buf.dst_ptr(), buf.pred_strd, buf.dst_strd, log2_trans_size,
       static_cast<WORD16>(coeff), cfg.bit_depth);
  }

  // Unified runner for luma and chroma ITrans Recon DC
  template <typename PixelType, typename FuncPtr>
  void RunImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    ITransReconDCBuffers<PixelType> ref_buf = CreateITransReconDCBuffers<PixelType>(cfg);
    ITransReconDCBuffers<PixelType> tst_buf = ref_buf;

    Invoke(ref_fn, ref_buf);
    Invoke(tst_fn, tst_buf);

    // Compare all of dst (stride padding too) so writes outside the block are caught
    ASSERT_EQ(ref_buf.dst, tst_buf.dst);
  }

  int log2_trans_size;
  int coeff;
  ITransReconDCConfig cfg;
  const ihevcd_func_selector_t* ref;
  const ihevcd_func_selector_t* tst;
};

// ---------------------------- Luma Class -----------------------------------

class ITransReconDCLumaTest : public ITransReconDCTestBase {
 protected:
  ITransReconDCLumaTest() : ITransReconDCTestBase(false) {}

  void Run() {
    if (cfg.bit_depth == 8) {
      RunImpl<UWORD8>(ref->ihevcd_itrans_recon_dc_luma_fptr,
                      tst->ihevcd_itrans_recon_dc_luma_fptr);
    } else {
      RunImpl<UWORD16>(ref->ihevcd_hbd_itrans_recon_dc_luma_fptr,
                       tst->ihevcd_hbd_itrans_recon_dc_luma_fptr);
    }
  }
};

// ---------------------------- Chroma Class ---------------------------------

class ITransReconDCChromaTest : public ITransReconDCTestBase {
 protected:
  ITransReconDCChromaTest() : ITransReconDCTestBase(true) {}

  void Run() {
    if (cfg.bit_depth == 8) {
      RunImpl<UWORD8>(ref->ihevcd_itrans_recon_dc_chroma_fptr,
                      tst->ihevcd_itrans_recon_dc_chroma_fptr);
    } else {
      RunImpl<UWORD16>(ref->ihevcd_hbd_itrans_recon_dc_chroma_fptr,
                       tst->ihevcd_hbd_itrans_recon_dc_chroma_fptr);
    }
  }
};

// ---------------------------- Test cases -----------------------------------

TEST_P(ITransReconDCLumaTest, Run) { Run(); }

TEST_P(ITransReconDCChromaTest, Run) { Run(); }

// ---------------------------- Instantiation --------------------------------

// num_planes: 2 to also cover the V plane for chroma, 1 otherwise
std::vector<ITransReconDCTestParam> GenerateITransReconDCTestParams(int num_planes) {
  const int sizes[] = {4, 8, 16, 32};
  // Boundary and typical DC coefficient values, plus the rounding thresholds where DC becomes
  // +/-1 (8-bit: 63 / -66, 10-bit: 15 / -18) and a mid-range DC with partial clipping
  const int coeffs[] = {0,   62,  63,  -65,   -66,    14,    15,    -17,
                        -18, 500, -1200, 12800, -12800, 32767, -32768};
  const int bit_depths[] = {8, 10};

  std::vector<ITransReconDCTestParam> params;
  for (int size : sizes) {
    for (int coeff : coeffs) {
      for (int bit_depth : bit_depths) {
        for (int plane = 0; plane < num_planes; ++plane) {
          for (auto arch : getTstArch()) {
            params.emplace_back(size, arch, coeff, bit_depth, plane);
          }
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(ITransReconDC, ITransReconDCLumaTest,
                         ::testing::ValuesIn(GenerateITransReconDCTestParams(1)),
                         PrintITransReconDCTestParam);
INSTANTIATE_TEST_SUITE_P(ITransReconDC, ITransReconDCChromaTest,
                         ::testing::ValuesIn(GenerateITransReconDCTestParams(2)),
                         PrintITransReconDCTestParam);
}  // namespace
