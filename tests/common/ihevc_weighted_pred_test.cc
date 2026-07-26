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
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_inter_pred.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_typedefs.h"
#include "ihevcd_function_selector.h"
#include "tests_common.h"
// clang-format on

namespace {

// ---------------------------- Param Types ----------------------------------

using WeightedPredTestParam =
    std::tuple<std::pair<int, int>, int, int, IVD_ARCH_T>;

std::string PrintWeightedPredTestParam(
    const testing::TestParamInfo<WeightedPredTestParam>& info) {
  int wd, ht, src_strd_mul, dst_strd_mul;
  IVD_ARCH_T arch;
  std::pair<int, int> block_size;
  std::tie(block_size, src_strd_mul, dst_strd_mul, arch) = info.param;
  std::tie(wd, ht) = block_size;
  std::stringstream ss;
  ss << wd << "x" << ht << "_src_stride_mul_" << src_strd_mul
     << "_dst_stride_mul_" << dst_strd_mul << "_" << get_arch_str(arch);
  return ss.str();
}

// ---------------------------- Test Classes ---------------------------------

class WeightedPredUniLumaTest
    : public ::testing::TestWithParam<WeightedPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_stride_mul, dst_stride_mul, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    src_strd = wd * src_stride_mul;
    dst_strd = wd * dst_stride_mul;

    src_buf.resize(src_strd * ht + 16);
    dst_buf_ref.resize(dst_strd * ht + 16);
    dst_buf_tst.resize(dst_strd * ht + 16);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  int wd, ht, src_stride_mul, dst_stride_mul;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  IVD_ARCH_T arch;
  const func_selector_t* tst;
  const func_selector_t* ref;
};

TEST_P(WeightedPredUniLumaTest, Run) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int16_t> src_dist(-8192, 8191);
  for (auto& v : src_buf) v = src_dist(rng);

  std::uniform_int_distribution<int> wgt_dist(-128, 127);
  std::uniform_int_distribution<int> off_dist(-128, 127);
  std::uniform_int_distribution<int> shift_dist(7, 14);
  std::uniform_int_distribution<int> lvl_shift_dist(0, 1);

  int wgt0 = wgt_dist(rng);
  int off0 = off_dist(rng);
  int shift = shift_dist(rng);
  int lvl_shift = lvl_shift_dist(rng) ? 8192 : 0;

  std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xAA);
  std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xAA);

  ref->ihevc_weighted_pred_uni_fptr(src_buf.data(), dst_buf_ref.data(),
                                    src_strd, dst_strd, wgt0, off0, shift,
                                    lvl_shift, ht, wd);

  tst->ihevc_weighted_pred_uni_fptr(src_buf.data(), dst_buf_tst.data(),
                                    src_strd, dst_strd, wgt0, off0, shift,
                                    lvl_shift, ht, wd);

  ASSERT_NO_FATAL_FAILURE(
      compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, wd, ht, dst_strd));
}

// ---------------------------------------------------------------------------

class WeightedPredUniChromaTest
    : public ::testing::TestWithParam<WeightedPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_stride_mul, dst_stride_mul, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    src_strd = 2 * wd * src_stride_mul;
    dst_strd = 2 * wd * dst_stride_mul;

    src_buf.resize(src_strd * ht + 16);
    dst_buf_ref.resize(dst_strd * ht + 16);
    dst_buf_tst.resize(dst_strd * ht + 16);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  int wd, ht, src_stride_mul, dst_stride_mul;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  IVD_ARCH_T arch;
  const func_selector_t* tst;
  const func_selector_t* ref;
};

TEST_P(WeightedPredUniChromaTest, Run) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int16_t> src_dist(-8192, 8191);
  for (auto& v : src_buf) v = src_dist(rng);

  std::uniform_int_distribution<int> wgt_dist(-128, 127);
  std::uniform_int_distribution<int> off_dist(-128, 127);
  std::uniform_int_distribution<int> shift_dist(7, 14);

  int wgt0_cb = wgt_dist(rng);
  int wgt0_cr = wgt_dist(rng);
  int off0_cb = off_dist(rng);
  int off0_cr = off_dist(rng);
  int shift = shift_dist(rng);
  int lvl_shift = 0;

  std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xAA);
  std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xAA);

  ref->ihevc_weighted_pred_chroma_uni_fptr(
      src_buf.data(), dst_buf_ref.data(), src_strd, dst_strd, wgt0_cb, wgt0_cr,
      off0_cb, off0_cr, shift, lvl_shift, ht, wd);

  tst->ihevc_weighted_pred_chroma_uni_fptr(
      src_buf.data(), dst_buf_tst.data(), src_strd, dst_strd, wgt0_cb, wgt0_cr,
      off0_cb, off0_cr, shift, lvl_shift, ht, wd);

  ASSERT_NO_FATAL_FAILURE(
      compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, 2 * wd, ht, dst_strd));
}

// ---------------------------------------------------------------------------

class WeightedPredBiLumaTest
    : public ::testing::TestWithParam<WeightedPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_stride_mul, dst_stride_mul, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    src_strd = wd * src_stride_mul;
    dst_strd = wd * dst_stride_mul;

    src_buf1.resize(src_strd * ht + 16);
    src_buf2.resize(src_strd * ht + 16);
    dst_buf_ref.resize(dst_strd * ht + 16);
    dst_buf_tst.resize(dst_strd * ht + 16);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  int wd, ht, src_stride_mul, dst_stride_mul;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf1;
  std::vector<WORD16> src_buf2;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  IVD_ARCH_T arch;
  const func_selector_t* tst;
  const func_selector_t* ref;
};

TEST_P(WeightedPredBiLumaTest, Run) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int16_t> src_dist(-8192, 8191);
  for (auto& v : src_buf1) v = src_dist(rng);
  for (auto& v : src_buf2) v = src_dist(rng);

  std::uniform_int_distribution<int> wgt_dist(-128, 127);
  std::uniform_int_distribution<int> off_dist(-128, 127);
  std::uniform_int_distribution<int> shift_dist(7, 14);
  std::uniform_int_distribution<int> lvl_shift_dist(0, 1);

  int wgt0 = wgt_dist(rng);
  int wgt1 = wgt_dist(rng);
  int off0 = off_dist(rng);
  int off1 = off_dist(rng);
  int shift = shift_dist(rng);
  int lvl_shift1 = lvl_shift_dist(rng) ? 8192 : 0;
  int lvl_shift2 = lvl_shift_dist(rng) ? 8192 : 0;

  std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xAA);
  std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xAA);

  ref->ihevc_weighted_pred_bi_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_ref.data(), src_strd, src_strd,
      dst_strd, wgt0, off0, wgt1, off1, shift, lvl_shift1, lvl_shift2, ht, wd);

  tst->ihevc_weighted_pred_bi_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_tst.data(), src_strd, src_strd,
      dst_strd, wgt0, off0, wgt1, off1, shift, lvl_shift1, lvl_shift2, ht, wd);

  ASSERT_NO_FATAL_FAILURE(
      compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, wd, ht, dst_strd));
}

// ---------------------------------------------------------------------------

class WeightedPredBiChromaTest
    : public ::testing::TestWithParam<WeightedPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_stride_mul, dst_stride_mul, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    src_strd = 2 * wd * src_stride_mul;
    dst_strd = 2 * wd * dst_stride_mul;

    src_buf1.resize(src_strd * ht + 16);
    src_buf2.resize(src_strd * ht + 16);
    dst_buf_ref.resize(dst_strd * ht + 16);
    dst_buf_tst.resize(dst_strd * ht + 16);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  int wd, ht, src_stride_mul, dst_stride_mul;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf1;
  std::vector<WORD16> src_buf2;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  IVD_ARCH_T arch;
  const func_selector_t* tst;
  const func_selector_t* ref;
};

TEST_P(WeightedPredBiChromaTest, Run) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int16_t> src_dist(-8192, 8191);
  for (auto& v : src_buf1) v = src_dist(rng);
  for (auto& v : src_buf2) v = src_dist(rng);

  std::uniform_int_distribution<int> wgt_dist(-128, 127);
  std::uniform_int_distribution<int> off_dist(-128, 127);
  std::uniform_int_distribution<int> shift_dist(7, 14);

  int wgt0_cb = wgt_dist(rng);
  int wgt0_cr = wgt_dist(rng);
  int off0_cb = off_dist(rng);
  int off0_cr = off_dist(rng);
  int wgt1_cb = wgt_dist(rng);
  int wgt1_cr = wgt_dist(rng);
  int off1_cb = off_dist(rng);
  int off1_cr = off_dist(rng);
  int shift = shift_dist(rng);
  int lvl_shift1 = 0;
  int lvl_shift2 = 0;

  std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xAA);
  std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xAA);

  ref->ihevc_weighted_pred_chroma_bi_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_ref.data(), src_strd, src_strd,
      dst_strd, wgt0_cb, wgt0_cr, off0_cb, off0_cr, wgt1_cb, wgt1_cr, off1_cb,
      off1_cr, shift, lvl_shift1, lvl_shift2, ht, wd);

  tst->ihevc_weighted_pred_chroma_bi_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_tst.data(), src_strd, src_strd,
      dst_strd, wgt0_cb, wgt0_cr, off0_cb, off0_cr, wgt1_cb, wgt1_cr, off1_cb,
      off1_cr, shift, lvl_shift1, lvl_shift2, ht, wd);

  ASSERT_NO_FATAL_FAILURE(
      compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, 2 * wd, ht, dst_strd));
}

// ---------------------------------------------------------------------------

class WeightedPredBiDefaultLumaTest
    : public ::testing::TestWithParam<WeightedPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_stride_mul, dst_stride_mul, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    src_strd = wd * src_stride_mul;
    dst_strd = wd * dst_stride_mul;

    src_buf1.resize(src_strd * ht + 16);
    src_buf2.resize(src_strd * ht + 16);
    dst_buf_ref.resize(dst_strd * ht + 16);
    dst_buf_tst.resize(dst_strd * ht + 16);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  int wd, ht, src_stride_mul, dst_stride_mul;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf1;
  std::vector<WORD16> src_buf2;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  IVD_ARCH_T arch;
  const func_selector_t* tst;
  const func_selector_t* ref;
};

TEST_P(WeightedPredBiDefaultLumaTest, Run) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int16_t> src_dist(-8192, 8191);
  for (auto& v : src_buf1) v = src_dist(rng);
  for (auto& v : src_buf2) v = src_dist(rng);

  std::uniform_int_distribution<int> lvl_shift_dist(0, 1);
  int lvl_shift1 = lvl_shift_dist(rng) ? 8192 : 0;
  int lvl_shift2 = lvl_shift_dist(rng) ? 8192 : 0;

  std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xAA);
  std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xAA);

  ref->ihevc_weighted_pred_bi_default_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_ref.data(), src_strd, src_strd,
      dst_strd, lvl_shift1, lvl_shift2, ht, wd);

  tst->ihevc_weighted_pred_bi_default_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_tst.data(), src_strd, src_strd,
      dst_strd, lvl_shift1, lvl_shift2, ht, wd);

  ASSERT_NO_FATAL_FAILURE(
      compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, wd, ht, dst_strd));
}

// ---------------------------------------------------------------------------

class WeightedPredBiDefaultChromaTest
    : public ::testing::TestWithParam<WeightedPredTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, src_stride_mul, dst_stride_mul, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    src_strd = 2 * wd * src_stride_mul;
    dst_strd = 2 * wd * dst_stride_mul;

    src_buf1.resize(src_strd * ht + 16);
    src_buf2.resize(src_strd * ht + 16);
    dst_buf_ref.resize(dst_strd * ht + 16);
    dst_buf_tst.resize(dst_strd * ht + 16);

    tst = get_tst_func_ptr(arch);
    ref = get_ref_func_ptr();
  }

  int wd, ht, src_stride_mul, dst_stride_mul;
  int src_strd, dst_strd;
  std::vector<WORD16> src_buf1;
  std::vector<WORD16> src_buf2;
  std::vector<UWORD8> dst_buf_ref;
  std::vector<UWORD8> dst_buf_tst;
  IVD_ARCH_T arch;
  const func_selector_t* tst;
  const func_selector_t* ref;
};

TEST_P(WeightedPredBiDefaultChromaTest, Run) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int16_t> src_dist(-8192, 8191);
  for (auto& v : src_buf1) v = src_dist(rng);
  for (auto& v : src_buf2) v = src_dist(rng);

  int lvl_shift1 = 0;
  int lvl_shift2 = 0;

  std::fill(dst_buf_ref.begin(), dst_buf_ref.end(), 0xAA);
  std::fill(dst_buf_tst.begin(), dst_buf_tst.end(), 0xAA);

  ref->ihevc_weighted_pred_chroma_bi_default_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_ref.data(), src_strd, src_strd,
      dst_strd, lvl_shift1, lvl_shift2, ht, wd);

  tst->ihevc_weighted_pred_chroma_bi_default_fptr(
      src_buf1.data(), src_buf2.data(), dst_buf_tst.data(), src_strd, src_strd,
      dst_strd, lvl_shift1, lvl_shift2, ht, wd);

  ASSERT_NO_FATAL_FAILURE(
      compare_output<UWORD8>(dst_buf_ref, dst_buf_tst, 2 * wd, ht, dst_strd));
}

// ---------------------------- Instantiation --------------------------------

auto kLumaTestParams = ::testing::Combine(
    ::testing::ValuesIn(kPUBlockSizes), ::testing::Values(1, 2),
    ::testing::Values(1, 2), ::testing::ValuesIn(ga_tst_arch));

const std::vector<std::pair<int, int>> kChromaPUBlockSizes = []() {
  std::vector<std::pair<int, int>> ret;
  for (const auto& size : kPUBlockSizes) {
    ret.push_back({size.first / 2, size.second / 2});
  }
  return ret;
}();

auto kChromaTestParams = ::testing::Combine(
    ::testing::ValuesIn(kChromaPUBlockSizes), ::testing::Values(1, 2),
    ::testing::Values(1, 2), ::testing::ValuesIn(ga_tst_arch));

INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredUniLumaTest, kLumaTestParams,
                         PrintWeightedPredTestParam);

INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredUniChromaTest,
                         kChromaTestParams, PrintWeightedPredTestParam);

INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredBiLumaTest, kLumaTestParams,
                         PrintWeightedPredTestParam);

INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredBiChromaTest,
                         kChromaTestParams, PrintWeightedPredTestParam);

INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredBiDefaultLumaTest,
                         kLumaTestParams, PrintWeightedPredTestParam);

INSTANTIATE_TEST_SUITE_P(WeightedPred, WeightedPredBiDefaultChromaTest,
                         kChromaTestParams, PrintWeightedPredTestParam);

}  // namespace
