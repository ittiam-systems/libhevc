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
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_typedefs.h"
#include "TestCommon.h"
#include "ihevc_deblk_utils.h"
// clang-format on

namespace {

template <typename Buffers>
void VerifyBuffers(const Buffers& ref, const Buffers& tst) {
  ASSERT_EQ(ref, tst);
}

std::string format_int(int val) {
  if (val < 0) {
    return "m" + std::to_string(-val);
  }
  return std::to_string(val);
}

// ---------------------------- Luma Test -------------------------------------

// Param: bs, qp_p, qp_q, beta_offset, tc_offset, filter_pair(p, q), bit_depth, arch
using DeblkLumaParam = std::tuple<int, int, int, int, int, std::pair<int, int>, int, IV_ARCH_T>;

std::string PrintDeblkLumaParam(
    const testing::TestParamInfo<DeblkLumaParam>& info) {
  int bs, qp_p, qp_q, beta_offset, tc_offset, bit_depth;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;
  std::tie(bs, qp_p, qp_q, beta_offset, tc_offset, filter_pair, bit_depth, arch) = info.param;
  return "bs_" + format_int(bs) + "_qpP_" + format_int(qp_p) + "_qpQ_" + format_int(qp_q) +
         "_bOff_" + format_int(beta_offset) + "_tcOff_" + format_int(tc_offset) + "_fP_" +
         format_int(filter_pair.first) + "_fQ_" + format_int(filter_pair.second) + "_bd_" +
         format_int(bit_depth) + "_" + get_arch_str(arch);
}

class DeblkLumaTest : public ::testing::TestWithParam<DeblkLumaParam> {
 protected:
  void SetUp() override {
    std::tie(bs, qp_p, qp_q, beta_offset, tc_offset, filter_pair, bit_depth, arch) = GetParam();

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // ------------------------ Overloads for Deblk Luma -------------------------

  void InvokeDeblk(DeblkLumaFn fn, std::vector<UWORD8>& src) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q, beta_offset, tc_offset,
       filter_pair.first, filter_pair.second);
  }

  void InvokeDeblk(HbdDeblkLumaFn fn, std::vector<UWORD16>& src) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, bs, qp_p, qp_q, beta_offset, tc_offset,
       filter_pair.first, filter_pair.second, static_cast<UWORD8>(bit_depth));
  }

  // Unified runner for both 8-bit and HBD
  template <typename PixelType, typename FuncPtr>
  void RunTestImpl(bool is_vert, FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    // Test multiple deterministic seeds to evaluate varied edge gradients and noise.
    // any_seed_changed tracks whether filtering actively modified pixels when expected.
    bool any_seed_changed = false;
    for (uint32_t seed : {42, 1337, 9999}) {
      std::vector<PixelType> src_ref(kDeblkBufSize);
      std::vector<PixelType> src_tst(kDeblkBufSize);

      std::mt19937 rng(seed);
      InitializeDeblkBuffer(src_ref.data(), kDeblkBufSize, is_vert, bit_depth, rng);
      src_tst = src_ref;
      const std::vector<PixelType> initial_src = src_ref;

      InvokeDeblk(ref_fn, src_ref);
      InvokeDeblk(tst_fn, src_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(src_ref, src_tst));

      if (src_ref != initial_src) {
        any_seed_changed = true;
      }
    }

    if (ExpectDeblkLumaFilter(bs, qp_p, qp_q, beta_offset, tc_offset, filter_pair)) {
      EXPECT_TRUE(any_seed_changed);
    } else {
      EXPECT_FALSE(any_seed_changed);
    }

    // Targeted branch coverage and boundary tests.
    if (bs == 2 && beta_offset == 0 && tc_offset == 0 && filter_pair == std::make_pair(1, 1) &&
       (qp_p == 28 || qp_p == 51) && qp_q == qp_p) {
      RunBranchAndBoundaryTests<PixelType>(is_vert, ref_fn, tst_fn);
    }
  }

  template <typename PixelType, typename FuncPtr>
  void RunBranchAndBoundaryTests(bool is_vert, FuncPtr ref_fn, FuncPtr tst_fn) {
    if (!filter_pair.first && !filter_pair.second) {
      return;
    }

    const int scale = 1 << (bit_depth - 8);
    const int max_val = (1 << bit_depth) - 1;

    int qp_luma = (qp_p + qp_q + 1) >> 1;
    int beta_indx = std::clamp(qp_luma + (beta_offset << 1), 0, 51);
    int beta = gai4_ihevc_beta_table[beta_indx] * scale;
    int tc_indx = std::clamp(qp_luma + (2 * (bs >> 1)) + (tc_offset << 1), 0, 53);
    int tc = gai4_ihevc_tc_table[tc_indx] * scale;

    if (beta <= 0 || tc <= 0) {
      return;
    }

    // 1. Exact boundary abort: d == beta. Filter must abort without modifying pixels.
    if (beta >= 2 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      int target_unscaled = beta / scale;
      int is_odd = target_unscaled & 1;
      int half_even = (target_unscaled - is_odd) / 2;
      int dp0_target = (half_even / 2) * 2 + is_odd;
      int dp3_target = target_unscaled - dp0_target;

      int p0_0 = 100 * scale + (is_odd ? 1 * scale : 0);
      int p1_0 = 100 * scale + (dp0_target + is_odd) / 2 * scale;
      int p1_3 = 100 * scale + (dp3_target / 2) * scale;

      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, 100 * scale, 100 * scale, p1_0, p0_0,
                       100 * scale, 100 * scale, 100 * scale, 100 * scale);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, 100 * scale, 100 * scale, 100 * scale,
                       100 * scale, 100 * scale, 100 * scale, 100 * scale, 100 * scale);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, 100 * scale, 100 * scale, 100 * scale,
                       100 * scale, 100 * scale, 100 * scale, 100 * scale, 100 * scale);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, 100 * scale, 100 * scale, p1_3, 100 * scale,
                       100 * scale, 100 * scale, 100 * scale, 100 * scale);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_EQ(buf_ref, initial);
    }

    // 2. Boundary filter: d == beta - 1*scale (just below boundary). Filter must activate.
    if (beta >= 2 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      int target_unscaled = (beta - 1 * scale) / scale;
      int is_odd = target_unscaled & 1;
      int half_even = (target_unscaled - is_odd) / 2;
      int dp0_target = (half_even / 2) * 2 + is_odd;
      int dp3_target = target_unscaled - dp0_target;

      int p0_0 = 100 * scale + (is_odd ? 1 * scale : 0);
      int p1_0 = 100 * scale + (dp0_target + is_odd) / 2 * scale;
      int p1_3 = 100 * scale + (dp3_target / 2) * scale;
      int step = 4 * scale;

      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, 100 * scale, 100 * scale, p1_0, p0_0,
                       100 * scale + step, 100 * scale + step,
                       100 * scale + step, 100 * scale + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, 100 * scale, 100 * scale, 100 * scale,
                       100 * scale, 100 * scale + step, 100 * scale + step,
                       100 * scale + step, 100 * scale + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, 100 * scale, 100 * scale, 100 * scale,
                       100 * scale, 100 * scale + step, 100 * scale + step,
                       100 * scale + step, 100 * scale + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, 100 * scale, 100 * scale, p1_3, 100 * scale,
                       100 * scale + step, 100 * scale + step,
                       100 * scale + step, 100 * scale + step);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }

    // 3. Strong filter upper (+2tc) and lower (-2tc) clamping.
    const int t = 2 * tc + scale;
    if ((beta >> 3) > 0 && 2 * t <= 100 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      const int base = 100 * scale;
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, base, base, base, base,
                       base, base + t, base + 2 * t, base);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, base, base + 2 * t, base + t, base,
                       base, base, base, base);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, base, base - 2 * t, base - t, base,
                       base, base, base, base);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, base, base, base, base,
                       base, base - t, base - 2 * t, base);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }

    // 4. Weak filter skip when |delta| >= 10tc
    if (28 * tc <= max_val - 50 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(50 * scale));
      const int base = 50 * scale;
      const int step = 28 * tc;
      FillDeblkEdge(buf_ref.data(), is_vert, base, base, base, base,
                    base + step, base + step, base + step, base + step);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_EQ(buf_ref, initial);
    }

    // 5. Weak filter delta clamped to +tc and -tc
    if (6 * tc <= max_val - 100 * scale && 100 * scale >= 6 * tc) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      const int base = 100 * scale;
      const int step = 6 * tc;
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, base, base, base, base,
                       base + step, base + step, base + step, base + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, base, base, base, base,
                       base + step, base + step, base + step, base + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, base, base, base, base,
                       base - step, base - step, base - step, base - step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, base, base, base, base,
                       base - step, base - step, base - step, base - step);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }

    // 6A. Asymmetric curvature: dEp = 1, dEq = 0 (p1 filtered, q1 untouched)
    if (beta >= 16 * scale && tc >= 2 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      const int base = 100 * scale;
      const int step = 6 * scale;
      const int curve = 2 * scale;
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, base, base, base, base,
                       base + step, base + step + curve, base + step, base + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, base, base, base, base,
                       base + step, base + step + curve, base + step, base + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, base, base, base, base,
                       base + step, base + step, base + step, base + step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, base, base, base, base,
                       base + step, base + step, base + step, base + step);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }

    // 6B. Asymmetric curvature: dEp = 0, dEq = 1 (q1 filtered, p1 untouched)
    if (beta >= 16 * scale && tc >= 2 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      const int base = 100 * scale;
      const int step = 6 * scale;
      const int curve = 2 * scale;
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, base + step, base + step,
                       base + step + curve, base + step, base, base, base, base);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, base + step, base + step,
                       base + step + curve, base + step, base, base, base, base);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, base + step, base + step,
                       base + step, base + step, base, base, base, base);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, base + step, base + step,
                       base + step, base + step, base, base, base, base);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }

    // 7. Clipping to 0 (underflow) and max_val (overflow)
    if (beta >= 16 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(0));
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, 0, 0, 0, 0,
                       0, 6 * scale, 6 * scale, 6 * scale);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, 0, 0, 0, 0,
                       0, 6 * scale, 6 * scale, 6 * scale);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, max_val, max_val, max_val, max_val,
                       max_val, max_val - 6 * scale, max_val - 6 * scale, max_val - 6 * scale);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, max_val, max_val, max_val, max_val,
                       max_val, max_val - 6 * scale, max_val - 6 * scale, max_val - 6 * scale);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }

    // 8. Mixed lines 0 and 3: d_sam0 = 1, d_sam3 = 0, causing de = 1 (weak filter)
    if (beta >= 8 * scale && ((5 * tc + 1) >> 1) >= 2 * scale) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      const int base = 100 * scale;
      const int small_step = 2 * scale;
      const int large_step = ((5 * tc + 1) >> 1) + 4 * scale;
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, base, base, base, base,
                       base + small_step, base + small_step,
                       base + small_step, base + small_step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, base, base, base, base,
                       base + small_step, base + small_step,
                       base + small_step, base + small_step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, base, base, base, base,
                       base + small_step, base + small_step,
                       base + small_step, base + small_step);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, base, base, base, base,
                       base + large_step, base + large_step,
                       base + large_step, base + large_step);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }

    // 9. Strong filter rejected by flatness alone: zero curvature and a small step, but
    // |p3 - p0| + |q0 - q3| == beta >> 3 (q3 on line 0, p3 on line 3), so weak filter
    if ((beta >> 2) > 0) {
      std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(100 * scale));
      const int base = 100 * scale;
      const int q = base + 2 * scale;
      const int flat = beta >> 3;
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 0, base, base, base, base, q, q, q, q + flat);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 1, base, base, base, base, q, q, q, q);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 2, base, base, base, base, q, q, q, q);
      SetDeblkEdgeLine(buf_ref.data(), is_vert, 3, base - flat, base, base, base, q, q, q, q);
      std::vector<PixelType> buf_tst = buf_ref;
      const std::vector<PixelType> initial = buf_ref;

      InvokeDeblk(ref_fn, buf_ref);
      InvokeDeblk(tst_fn, buf_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
      EXPECT_NE(buf_ref, initial);
    }
  }

  void RunTest(bool is_vert) {
    if (bit_depth == 8) {
      if (qp_p < 0 || qp_q < 0) {
        GTEST_SKIP() << "Negative QP is only valid for High Bit Depth (HBD)";
      }
      RunTestImpl<UWORD8>(is_vert, GetLumaTargetFn(ref, is_vert), GetLumaTargetFn(tst, is_vert));
    } else {
      RunTestImpl<UWORD16>(is_vert, GetHbdLumaTargetFn(ref, is_vert),
                           GetHbdLumaTargetFn(tst, is_vert));
    }
  }

  int bs, qp_p, qp_q, beta_offset, tc_offset, bit_depth;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;

  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

TEST_P(DeblkLumaTest, LumaVert) { RunTest(true); }

TEST_P(DeblkLumaTest, LumaHorz) { RunTest(false); }

// ---------------------------- Chroma Test ------------------------------------

// Sets p1, p0, q0, q1 of one chroma edge position for both U and V (interleaved UV).
// Vertical edge: row idx (0..3). Horizontal edge: UV pair idx (0..3), rows -2..1.
template <typename PixelType>
void SetDeblkChromaEdge(PixelType* buf, bool is_vert, int idx, int p1, int p0, int q0, int q1) {
  if (is_vert) {
    SetDeblkEdgeLine(buf, true, idx, p1, p1, p0, p0, q0, q0, q1, q1);
    return;
  }
  PixelType* pu = buf + kDeblkSrcOffset + 2 * idx;
  for (int c = 0; c < 2; ++c) {
    pu[c - 2 * kDeblkStride] = static_cast<PixelType>(p1);
    pu[c - 1 * kDeblkStride] = static_cast<PixelType>(p0);
    pu[c + 0 * kDeblkStride] = static_cast<PixelType>(q0);
    pu[c + 1 * kDeblkStride] = static_cast<PixelType>(q1);
  }
}

// Param: qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, filter_pair(p, q), chroma_fmt_idc,
// bit_depth, arch
using DeblkChromaParam = std::tuple<int, int, int, int, int, std::pair<int, int>, int, int,
                                    IV_ARCH_T>;

std::string PrintDeblkChromaParam(const testing::TestParamInfo<DeblkChromaParam>& info) {
  int qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, chroma_fmt_idc, bit_depth;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;
  std::tie(qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, filter_pair, chroma_fmt_idc,
           bit_depth, arch) = info.param;
  return "qpP_" + format_int(qp_p) + "_qpQ_" + format_int(qp_q) + "_qpU_" +
         format_int(qp_offset_u) + "_qpV_" + format_int(qp_offset_v) + "_tcOff_" +
         format_int(tc_offset) + "_fP_" + format_int(filter_pair.first) + "_fQ_" +
         format_int(filter_pair.second) + "_fmt_" + format_int(chroma_fmt_idc) +
         "_bd_" + format_int(bit_depth) + "_" + get_arch_str(arch);
}

class DeblkChromaTest : public ::testing::TestWithParam<DeblkChromaParam> {
 protected:
  void SetUp() override {
    std::tie(qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, filter_pair, chroma_fmt_idc,
             bit_depth, arch) = GetParam();

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // ----------------------- Overloads for Deblk Chroma ------------------------

  void InvokeDeblk(DeblkChromaFn fn, std::vector<UWORD8>& src) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset,
       filter_pair.first, filter_pair.second, static_cast<WORD8>(chroma_fmt_idc));
  }

  void InvokeDeblk(HbdDeblkChromaFn fn, std::vector<UWORD16>& src) {
    fn(src.data() + kDeblkSrcOffset, kDeblkStride, qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset,
       filter_pair.first, filter_pair.second, static_cast<UWORD8>(bit_depth),
       static_cast<WORD8>(chroma_fmt_idc));
  }

  // Unified runner for both 8-bit and HBD
  template <typename PixelType, typename FuncPtr>
  void RunTestImpl(bool is_vert, FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    // Test multiple deterministic seeds to evaluate varied edge gradients and noise.
    // any_seed_changed tracks whether filtering actively modified pixels when expected.
    bool any_seed_changed = false;
    for (uint32_t seed : {42, 1337, 9999}) {
      std::vector<PixelType> src_ref(kDeblkBufSize);
      std::vector<PixelType> src_tst(kDeblkBufSize);

      std::mt19937 rng(seed);
      InitializeDeblkBuffer(src_ref.data(), kDeblkBufSize, is_vert, bit_depth, rng);
      src_tst = src_ref;
      const std::vector<PixelType> initial_src = src_ref;

      InvokeDeblk(ref_fn, src_ref);
      InvokeDeblk(tst_fn, src_tst);

      ASSERT_NO_FATAL_FAILURE(VerifyBuffers(src_ref, src_tst));

      if (src_ref != initial_src) {
        any_seed_changed = true;
      }
    }

    if (ExpectDeblkChromaFilter(qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, filter_pair,
                                chroma_fmt_idc)) {
      EXPECT_TRUE(any_seed_changed);
    } else {
      EXPECT_FALSE(any_seed_changed);
    }

    // Targeted clipping test, once per format / bit depth / arch / orientation
    if (qp_p == 28 && qp_q == 28 && qp_offset_u == 0 && qp_offset_v == 0 && tc_offset == 0 &&
        filter_pair == std::make_pair(1, 1)) {
      RunClipTest<PixelType>(is_vert, ref_fn, tst_fn);
    }
  }

  // p0 + delta and q0 - delta clipped to 0 and max_val, for U and V. With extreme values delta
  // saturates at +-tc, so p0 / q0 at an end of the range is pushed past it on each line.
  template <typename PixelType, typename FuncPtr>
  void RunClipTest(bool is_vert, FuncPtr ref_fn, FuncPtr tst_fn) {
    const int max_val = (1 << bit_depth) - 1;
    // {p1, p0, q0, q1}: delta = clip((4 * (q0 - p0) + p1 - q1 + 4) >> 3, -tc, tc)
    const int lines[4][4] = {
        {max_val, max_val, max_val, 0},  // delta = +tc: p0 above max_val
        {max_val, 0, 0, 0},              // delta = +tc: q0 below 0
        {0, 0, 0, max_val},              // delta = -tc: p0 below 0
        {0, max_val, max_val, max_val},  // delta = -tc: q0 above max_val
    };
    std::vector<PixelType> buf_ref(kDeblkBufSize, static_cast<PixelType>(max_val / 2));
    for (int i = 0; i < 4; ++i) {
      SetDeblkChromaEdge(buf_ref.data(), is_vert, i, lines[i][0], lines[i][1], lines[i][2],
                         lines[i][3]);
    }
    std::vector<PixelType> buf_tst = buf_ref;
    const std::vector<PixelType> initial = buf_ref;

    InvokeDeblk(ref_fn, buf_ref);
    InvokeDeblk(tst_fn, buf_tst);

    ASSERT_NO_FATAL_FAILURE(VerifyBuffers(buf_ref, buf_tst));
    EXPECT_NE(buf_ref, initial);
  }

  void RunTest(bool is_vert) {
    if (bit_depth == 8) {
      if (qp_p < 0 || qp_q < 0) {
        GTEST_SKIP() << "Negative QP is only valid for High Bit Depth (HBD)";
      }
      RunTestImpl<UWORD8>(is_vert, GetChromaTargetFn(ref, is_vert, chroma_fmt_idc),
                          GetChromaTargetFn(tst, is_vert, chroma_fmt_idc));
    } else {
      RunTestImpl<UWORD16>(is_vert, GetHbdChromaTargetFn(ref, is_vert, chroma_fmt_idc),
                           GetHbdChromaTargetFn(tst, is_vert, chroma_fmt_idc));
    }
  }

  int qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, chroma_fmt_idc, bit_depth;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;

  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

TEST_P(DeblkChromaTest, ChromaVert) { RunTest(true); }

TEST_P(DeblkChromaTest, ChromaHorz) { RunTest(false); }

// ---------------------------- Instantiation --------------------------------

auto kDeblkLumaParams = ::testing::Combine(
    ::testing::Values(1, 2, 3),            // bs
    ::testing::Values(-12, 0, 28, 51),     // qp_p
    ::testing::Values(-12, 0, 28, 51),     // qp_q
    ::testing::Values(-6, -3, 0, 3, 6),    // beta_offset
    ::testing::Values(-6, -3, 0, 3, 6),    // tc_offset
    ::testing::Values(std::make_pair(0, 1), std::make_pair(1, 0), std::make_pair(1, 1)),
                                           // filter_p, filter_q
    ::testing::Values(8, 10),              // bit_depth
    ::testing::ValuesIn(getTstArch()));

auto kDeblkChromaParams = ::testing::Combine(
    ::testing::Values(-12, 0, 28, 51),    // qp_p
    ::testing::Values(-12, 0, 28, 51),    // qp_q
    ::testing::Values(-12, -6, 0, 6, 12), // qp_offset_u
    ::testing::Values(-12, -6, 0, 6, 12), // qp_offset_v
    ::testing::Values(-6, -3, 0, 3, 6),   // tc_offset
    ::testing::Values(std::make_pair(0, 1), std::make_pair(1, 0), std::make_pair(1, 1)),
                                          // filter_p, filter_q
    ::testing::Values(1, 2, 3),           // chroma_fmt_idc (YUV420, YUV422, YUV444)
    ::testing::Values(8, 10),             // bit_depth
    ::testing::ValuesIn(getTstArch()));

INSTANTIATE_TEST_SUITE_P(Deblk, DeblkLumaTest, kDeblkLumaParams, PrintDeblkLumaParam);
INSTANTIATE_TEST_SUITE_P(Deblk, DeblkChromaTest, kDeblkChromaParams, PrintDeblkChromaParam);

}  // namespace
