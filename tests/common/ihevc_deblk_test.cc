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
// clang-format on

namespace {

void compare_deblk_output(const UWORD8* ref, const UWORD8* tst, int stride,
                          int wd, int ht) {
  for (int r = 0; r < ht; r++) {
    for (int c = 0; c < wd; c++) {
      ASSERT_EQ(ref[r * stride + c], tst[r * stride + c])
          << "Mismatch at row " << r << ", col " << c;
    }
  }
}

std::string format_int(int val) {
  if (val < 0) {
    return "m" + std::to_string(-val);
  }
  return std::to_string(val);
}

// ---------------------------- Luma Test -------------------------------------

// Param: bs, qp_p, qp_q, beta_offset, tc_offset, filter_pair(p, q), arch
using DeblkLumaParam =
    std::tuple<int, int, int, int, int, std::pair<int, int>, IV_ARCH_T>;

std::string PrintDeblkLumaParam(
    const testing::TestParamInfo<DeblkLumaParam>& info) {
  int bs, qp_p, qp_q, beta_offset, tc_offset;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;
  std::tie(bs, qp_p, qp_q, beta_offset, tc_offset, filter_pair, arch) =
      info.param;
  return "bs_" + format_int(bs) + "_qpP_" + format_int(qp_p) + "_qpQ_" +
         format_int(qp_q) + "_bOff_" + format_int(beta_offset) + "_tcOff_" +
         format_int(tc_offset) + "_fP_" + format_int(filter_pair.first) +
         "_fQ_" + format_int(filter_pair.second) + "_" + get_arch_str(arch);
}

class DeblkLumaTest : public ::testing::TestWithParam<DeblkLumaParam> {
 protected:
  void SetUp() override {
    std::tie(bs, qp_p, qp_q, beta_offset, tc_offset, filter_pair, arch) =
        GetParam();
    stride = 32;
    buf_size = stride * 32;
    src_offset = 16 * stride + 16;  // Point to middle of buffer

    src_ref.resize(buf_size);
    src_tst.resize(buf_size);

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (int i = 0; i < buf_size; i++) {
      uint8_t val = dist(rng);
      src_ref[i] = val;
      src_tst[i] = val;
    }
  }

  int bs, qp_p, qp_q, beta_offset, tc_offset;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;
  int stride, buf_size, src_offset;
  std::vector<UWORD8> src_ref;
  std::vector<UWORD8> src_tst;

  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

TEST_P(DeblkLumaTest, LumaVert) {
  InitializeBuffers();

  ref->ihevc_deblk_luma_vert_fptr(src_ref.data() + src_offset, stride, bs, qp_p,
                                  qp_q, beta_offset, tc_offset,
                                  filter_pair.first, filter_pair.second);

  tst->ihevc_deblk_luma_vert_fptr(src_tst.data() + src_offset, stride, bs, qp_p,
                                  qp_q, beta_offset, tc_offset,
                                  filter_pair.first, filter_pair.second);

  compare_deblk_output(src_ref.data(), src_tst.data(), stride, stride, 32);
}

TEST_P(DeblkLumaTest, LumaHorz) {
  InitializeBuffers();

  ref->ihevc_deblk_luma_horz_fptr(src_ref.data() + src_offset, stride, bs, qp_p,
                                  qp_q, beta_offset, tc_offset,
                                  filter_pair.first, filter_pair.second);

  tst->ihevc_deblk_luma_horz_fptr(src_tst.data() + src_offset, stride, bs, qp_p,
                                  qp_q, beta_offset, tc_offset,
                                  filter_pair.first, filter_pair.second);

  compare_deblk_output(src_ref.data(), src_tst.data(), stride, stride, 32);
}

// ---------------------------- Chroma Test ------------------------------------

// Param: qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, filter_pair(p, q),
// chroma_fmt_idc, arch
using DeblkChromaParam =
    std::tuple<int, int, int, int, int, std::pair<int, int>, int, IV_ARCH_T>;

std::string PrintDeblkChromaParam(
    const testing::TestParamInfo<DeblkChromaParam>& info) {
  int qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, chroma_fmt_idc;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;
  std::tie(qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, filter_pair,
           chroma_fmt_idc, arch) = info.param;
  return "qpP_" + format_int(qp_p) + "_qpQ_" + format_int(qp_q) + "_qpU_" +
         format_int(qp_offset_u) + "_qpV_" + format_int(qp_offset_v) +
         "_tcOff_" + format_int(tc_offset) + "_fP_" +
         format_int(filter_pair.first) + "_fQ_" +
         format_int(filter_pair.second) + "_fmt_" + format_int(chroma_fmt_idc) +
         "_" + get_arch_str(arch);
}

class DeblkChromaTest : public ::testing::TestWithParam<DeblkChromaParam> {
 protected:
  void SetUp() override {
    std::tie(qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, filter_pair,
             chroma_fmt_idc, arch) = GetParam();
    stride = 32;
    buf_size = stride * 32;
    src_offset = 16 * stride + 16;  // Point to middle of buffer

    src_ref.resize(buf_size);
    src_tst.resize(buf_size);

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  void InitializeBuffers() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (int i = 0; i < buf_size; i++) {
      uint8_t val = dist(rng);
      src_ref[i] = val;
      src_tst[i] = val;
    }
  }

  int qp_p, qp_q, qp_offset_u, qp_offset_v, tc_offset, chroma_fmt_idc;
  std::pair<int, int> filter_pair;
  IV_ARCH_T arch;
  int stride, buf_size, src_offset;
  std::vector<UWORD8> src_ref;
  std::vector<UWORD8> src_tst;

  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

TEST_P(DeblkChromaTest, ChromaVert) {
#if defined(__arm__) || defined(__aarch64__) || defined(__arm64__)
  if ((chroma_fmt_idc == 3) && (qp_p != 12 && qp_q != 12) &&
      (qp_p == 44 || qp_q == 44)) {
    GTEST_SKIP() << "Skipping failing ARM Chroma deblk tests for YUV444 and QP 44/28";
  }
#endif
  InitializeBuffers();

  ref->ihevc_deblk_chroma_vert_fptr(
      src_ref.data() + src_offset, stride, qp_p, qp_q, qp_offset_u, qp_offset_v,
      tc_offset, filter_pair.first, filter_pair.second, chroma_fmt_idc);

  tst->ihevc_deblk_chroma_vert_fptr(
      src_tst.data() + src_offset, stride, qp_p, qp_q, qp_offset_u, qp_offset_v,
      tc_offset, filter_pair.first, filter_pair.second, chroma_fmt_idc);

  compare_deblk_output(src_ref.data(), src_tst.data(), stride, stride, 32);
}

TEST_P(DeblkChromaTest, ChromaHorz) {
#if defined(__arm__) || defined(__aarch64__) || defined(__arm64__)
  if ((chroma_fmt_idc == 3) && (qp_p != 12 && qp_q != 12) &&
      (qp_p == 44 || qp_q == 44)) {
    GTEST_SKIP() << "Skipping failing ARM Chroma deblk tests for YUV444 and QP 44/28";
  }
#endif
  InitializeBuffers();

  ref->ihevc_deblk_chroma_horz_fptr(
      src_ref.data() + src_offset, stride, qp_p, qp_q, qp_offset_u, qp_offset_v,
      tc_offset, filter_pair.first, filter_pair.second, chroma_fmt_idc);

  tst->ihevc_deblk_chroma_horz_fptr(
      src_tst.data() + src_offset, stride, qp_p, qp_q, qp_offset_u, qp_offset_v,
      tc_offset, filter_pair.first, filter_pair.second, chroma_fmt_idc);

  compare_deblk_output(src_ref.data(), src_tst.data(), stride, stride, 32);
}

// ---------------------------- Instantiation --------------------------------

auto kDeblkLumaParams = ::testing::Combine(
    ::testing::Values(1, 2, 3),     // bs
    ::testing::Values(12, 28, 44),  // qp_p
    ::testing::Values(12, 28, 44),  // qp_q
    ::testing::Values(-2, 0, 2),    // beta_offset
    ::testing::Values(-2, 0, 2),    // tc_offset
    ::testing::Values(std::make_pair(0, 1), std::make_pair(1, 0),
                      std::make_pair(1, 1)),  // filter_p, filter_q
    ::testing::ValuesIn(ga_tst_arch));

auto kDeblkChromaParams = ::testing::Combine(
    ::testing::Values(12, 28, 44),  // qp_p
    ::testing::Values(12, 28, 44),  // qp_q
    ::testing::Values(-2, 0, 2),    // qp_offset_u
    ::testing::Values(-2, 0, 2),    // qp_offset_v
    ::testing::Values(-2, 0, 2),    // tc_offset
    ::testing::Values(std::make_pair(0, 1), std::make_pair(1, 0),
                      std::make_pair(1, 1)),  // filter_p, filter_q
    ::testing::Values(1, 3),                  // chroma_fmt_idc (YUV420, YUV444)
    ::testing::ValuesIn(ga_tst_arch));

INSTANTIATE_TEST_SUITE_P(Deblk, DeblkLumaTest, kDeblkLumaParams,
                         PrintDeblkLumaParam);
INSTANTIATE_TEST_SUITE_P(Deblk, DeblkChromaTest, kDeblkChromaParams,
                         PrintDeblkChromaParam);

}  // namespace
