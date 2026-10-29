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
#include <type_traits>
#include <utility>
#include <vector>

// clang-format off
#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_inter_pred_utils.h"
#include "ihevc_typedefs.h"
// clang-format on

namespace {

// ---------------------------- Test Param -----------------------------------

const char* GetOpStr(InterPredOp op) {
  for (const auto& op_info : GetInterPredOps()) {
    if (op_info.op == op) return op_info.name;
  }
  return "unknown";
}

// Param: wd, ht, op, arch, src_strd_mul, dst_strd_mul, coeff_idx, bit_depth
using InterPredParam = std::tuple<int, int, InterPredOp, IV_ARCH_T, int, int, int, int>;

std::string PrintInterPredParam(const testing::TestParamInfo<InterPredParam>& info) {
  int wd, ht, src_strd_mul, dst_strd_mul, coeff_idx, bit_depth;
  InterPredOp op;
  IV_ARCH_T arch;
  std::tie(wd, ht, op, arch, src_strd_mul, dst_strd_mul, coeff_idx, bit_depth) = info.param;
  return "size_" + std::to_string(wd) + "x" + std::to_string(ht) + "_" + GetOpStr(op) +
         "_src_mul_" + std::to_string(src_strd_mul) + "_dst_mul_" + std::to_string(dst_strd_mul) +
         "_coeff_" + std::to_string(coeff_idx) + "_bd_" + std::to_string(bit_depth) + "_" +
         get_arch_str(arch);
}

// ---------------------------- Base Class -----------------------------------

class InterPredTestBase : public ::testing::TestWithParam<InterPredParam> {
 protected:
  explicit InterPredTestBase(bool is_chroma) : is_chroma(is_chroma) {}

  void SetUp() override {
    int src_strd_mul, dst_strd_mul, coeff_idx;
    IV_ARCH_T arch;
    std::tie(wd, ht, op, arch, src_strd_mul, dst_strd_mul, coeff_idx, bit_depth) = GetParam();
    // Chroma rows hold U and V interleaved: 2 samples per pixel
    src_strd = (is_chroma ? 2 : 1) * wd * src_strd_mul;
    dst_strd = (is_chroma ? 2 : 1) * wd * dst_strd_mul;
    pi1_coeffs = is_chroma ? gai1_ihevc_chroma_filter[coeff_idx]
                           : gai1_ihevc_luma_filter[coeff_idx];

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // --------------------------- Invoke ---------------------------------------

  // 8-bit and HBD functions take the same arguments, plus bit_depth at the end for HBD
  template <typename FuncPtr, typename SrcType, typename DstType>
  void Invoke(FuncPtr fn, SrcType* src, DstType* dst) {
    if constexpr (std::is_invocable_v<FuncPtr, SrcType*, DstType*, WORD32, WORD32, WORD8*, WORD32,
                                      WORD32, UWORD8>) {
      fn(src, dst, src_strd, dst_strd, pi1_coeffs, ht, wd, static_cast<UWORD8>(bit_depth));
    } else {
      fn(src, dst, src_strd, dst_strd, pi1_coeffs, ht, wd);
    }
  }

  // Unified runner for luma and chroma Inter Prediction
  template <typename SrcType, typename DstType, typename FuncPtr>
  void RunImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // Largest dst padding the SIMD functions need: 12, chroma copy's 16-byte masked store on a
    // 4-sample remainder
    int pad_dst = 12;
#else
    int pad_dst = 0;
#endif

    // src holds pixels, except for w16inp, which reads horz_w16out output: filter sums, negative
    // and above max pixel too. neg/pos: largest sums of the negative and positive taps.
    const int max_val = (1 << bit_depth) - 1;
    int src_min = 0;
    int src_max = max_val;
    if constexpr (std::is_same_v<SrcType, WORD16>) {
      const int neg = is_chroma ? 10 : 24;
      const int pos = is_chroma ? 74 : 88;
      const int shift = bit_depth - 8;
      src_min = -((neg * max_val) >> shift);
      src_max = (pos * max_val) >> shift;
    }

    // Rows the vertical filter reads above/below the block: 3/4 luma, 1/2 chroma
    const bool is_vert = op == InterPredOp::kVert || op == InterPredOp::kVertW16out ||
                         op == InterPredOp::kVertW16inp || op == InterPredOp::kVertW16inpW16out;
    const int ntaps = is_chroma ? NTAPS_CHROMA : NTAPS_LUMA;
    const int top = is_vert ? ntaps / 2 - 1 : 0;
    const int bot = is_vert ? ntaps / 2 : 0;

    auto buf = CreateInterPredTestBuffers<SrcType, DstType>(
        ht, src_strd, dst_strd, static_cast<SrcType>(src_min), static_cast<SrcType>(src_max), top,
        bot, pad_dst);

    Invoke(ref_fn, buf.src_ptr(), buf.dst_ref.data());
    Invoke(tst_fn, buf.src_ptr(), buf.dst_tst.data());

    // Compare all of dst (stride gaps and pad too) so writes outside the block are caught
    ASSERT_EQ(buf.dst_ref, buf.dst_tst);
  }

  const bool is_chroma;
  int wd, ht, bit_depth;
  int src_strd, dst_strd;
  InterPredOp op;
  WORD8* pi1_coeffs;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// ---------------------------- Luma Class -----------------------------------

class LumaInterPredTest : public InterPredTestBase {
 protected:
  LumaInterPredTest() : InterPredTestBase(false) {}

  void Run() {
    const bool w16_inp = IsW16Inp(op);
    const bool w16_out = IsW16Out(op);

    if (bit_depth == 8) {
      if (!w16_inp && !w16_out) {
        RunImpl<UWORD8, UWORD8>(GetLumaInterPredFn(ref, op), GetLumaInterPredFn(tst, op));
      } else if (!w16_inp && w16_out) {
        RunImpl<UWORD8, WORD16>(GetLumaInterPredW16outFn(ref, op),
                                GetLumaInterPredW16outFn(tst, op));
      } else if (w16_inp && !w16_out) {
        RunImpl<WORD16, UWORD8>(GetLumaInterPredW16inpFn(ref, op),
                                GetLumaInterPredW16inpFn(tst, op));
      } else {
        RunImpl<WORD16, WORD16>(GetLumaInterPredW16inpW16outFn(ref, op),
                                GetLumaInterPredW16inpW16outFn(tst, op));
      }
    } else {
      if (!w16_inp && !w16_out) {
        RunImpl<UWORD16, UWORD16>(GetHbdLumaInterPredFn(ref, op), GetHbdLumaInterPredFn(tst, op));
      } else if (!w16_inp && w16_out) {
        RunImpl<UWORD16, WORD16>(GetHbdLumaInterPredW16outFn(ref, op),
                                 GetHbdLumaInterPredW16outFn(tst, op));
      } else if (w16_inp && !w16_out) {
        RunImpl<WORD16, UWORD16>(GetHbdLumaInterPredW16inpFn(ref, op),
                                 GetHbdLumaInterPredW16inpFn(tst, op));
      } else {
        RunImpl<WORD16, WORD16>(GetHbdLumaInterPredW16inpW16outFn(ref, op),
                                GetHbdLumaInterPredW16inpW16outFn(tst, op));
      }
    }
  }
};

// ---------------------------- Chroma Class ---------------------------------

class ChromaInterPredTest : public InterPredTestBase {
 protected:
  ChromaInterPredTest() : InterPredTestBase(true) {}

  void Run() {
    const bool w16_inp = IsW16Inp(op);
    const bool w16_out = IsW16Out(op);

    if (bit_depth == 8) {
      if (!w16_inp && !w16_out) {
        RunImpl<UWORD8, UWORD8>(GetChromaInterPredFn(ref, op), GetChromaInterPredFn(tst, op));
      } else if (!w16_inp && w16_out) {
        RunImpl<UWORD8, WORD16>(GetChromaInterPredW16outFn(ref, op),
                                GetChromaInterPredW16outFn(tst, op));
      } else if (w16_inp && !w16_out) {
        RunImpl<WORD16, UWORD8>(GetChromaInterPredW16inpFn(ref, op),
                                GetChromaInterPredW16inpFn(tst, op));
      } else {
        RunImpl<WORD16, WORD16>(GetChromaInterPredW16inpW16outFn(ref, op),
                                GetChromaInterPredW16inpW16outFn(tst, op));
      }
    } else {
      if (!w16_inp && !w16_out) {
        RunImpl<UWORD16, UWORD16>(GetHbdChromaInterPredFn(ref, op),
                                  GetHbdChromaInterPredFn(tst, op));
      } else if (!w16_inp && w16_out) {
        RunImpl<UWORD16, WORD16>(GetHbdChromaInterPredW16outFn(ref, op),
                                 GetHbdChromaInterPredW16outFn(tst, op));
      } else if (w16_inp && !w16_out) {
        RunImpl<WORD16, UWORD16>(GetHbdChromaInterPredW16inpFn(ref, op),
                                 GetHbdChromaInterPredW16inpFn(tst, op));
      } else {
        RunImpl<WORD16, WORD16>(GetHbdChromaInterPredW16inpW16outFn(ref, op),
                                GetHbdChromaInterPredW16inpW16outFn(tst, op));
      }
    }
  }
};

// ---------------------------- Test cases -----------------------------------

TEST_P(LumaInterPredTest, Run) { Run(); }

TEST_P(ChromaInterPredTest, Run) { Run(); }

// ---------------------------- Instantiation --------------------------------

// Luma has 4 filter phases, chroma 8
std::vector<InterPredParam> GenerateInterPredTestParams(bool is_chroma) {
  const std::vector<std::pair<int, int>> block_sizes =
      is_chroma ? getChromaAllFmtBlockSizes() : getLumaPUBlockSizes();
  const int num_phases = is_chroma ? 8 : 4;
  const int ntaps = is_chroma ? NTAPS_CHROMA : NTAPS_LUMA;
  const int strd_muls[] = {1, 2};
  const int bit_depths[] = {8, 10};

  std::vector<InterPredParam> params;
  for (const auto& op_info : GetInterPredOps()) {
    for (const auto& block_size : block_sizes) {
      std::vector<int> hts = {block_size.second};
      // 2-D filtering runs horz_w16out on ntaps - 1 extra rows, giving odd heights
      if (op_info.op == InterPredOp::kHorzW16out) hts.push_back(block_size.second + ntaps - 1);
      for (int ht : hts) {
        for (int src_strd_mul : strd_muls) {
          for (int dst_strd_mul : strd_muls) {
            // Copy ignores the filter, so one phase is enough; the decoder never filters with
            // phase 0 (integer position), it copies
            const bool is_copy = op_info.op == InterPredOp::kCopy ||
                                 op_info.op == InterPredOp::kCopyW16out;
            const int first_phase = is_copy ? 0 : 1;
            const int last_phase = is_copy ? 1 : num_phases;
            for (int coeff_idx = first_phase; coeff_idx < last_phase; ++coeff_idx) {
              for (int bit_depth : bit_depths) {
                for (auto arch : getTstArch()) {
                  params.emplace_back(block_size.first, ht, op_info.op, arch, src_strd_mul,
                                      dst_strd_mul, coeff_idx, bit_depth);
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

INSTANTIATE_TEST_SUITE_P(InterPred, LumaInterPredTest,
                         ::testing::ValuesIn(GenerateInterPredTestParams(false)),
                         PrintInterPredParam);
INSTANTIATE_TEST_SUITE_P(InterPred, ChromaInterPredTest,
                         ::testing::ValuesIn(GenerateInterPredTestParams(true)),
                         PrintInterPredParam);
}  // namespace
