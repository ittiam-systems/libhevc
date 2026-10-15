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
#include <set>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "func_selector.h"
#include "TestCommon.h"
#include "ihevc_sao_utils.h"
// clang-format on

namespace {

template <typename Buffers>
void VerifySaoBuffers(const Buffers& ref, const Buffers& tst) {
  ASSERT_EQ(ref.src, tst.src);
  ASSERT_EQ(ref.src_left, tst.src_left);
  ASSERT_EQ(ref.src_top, tst.src_top);
  ASSERT_EQ(ref.src_top_left, tst.src_top_left);
}

// Bit i of mask sets avail[i]: left, right, top, bottom, top-left, top-right,
// bottom-left, bottom-right.
void SetAvail(std::vector<UWORD8>& avail, int mask) {
  for (int i = 0; i < 8; ++i) avail[i] = ((mask >> i) & 1) ? 255 : 0;
}

// ---------------------------- Test Param -----------------------------------

// Param: block size, band position (band tests only), bit_depth, arch
using SaoTestParam = std::tuple<std::pair<int, int>, int, int, IV_ARCH_T>;

std::string PrintSaoTestParam(
    const testing::TestParamInfo<SaoTestParam>& info) {
  int wd, ht, val, bit_depth;
  IV_ARCH_T arch;
  std::pair<int, int> block_size;
  std::tie(block_size, val, bit_depth, arch) = info.param;
  std::tie(wd, ht) = block_size;
  return std::to_string(wd) + "x" + std::to_string(ht) + "_val_" + std::to_string(val) + "_bd_" +
         std::to_string(bit_depth) + "_" + get_arch_str(arch);
}

// ---------------------------- Luma Base Class ------------------------------

class SaoLumaTest : public ::testing::TestWithParam<SaoTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, offset_val, bit_depth, arch) = GetParam();
    std::tie(wd, ht) = block_size;

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // ------------------------ Overloads for Band Offset ------------------------

  void InvokeBandOffset(ihevc_sao_band_offset_luma_ft* fn, SaoLumaBuffers<UWORD8>& buf) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(), buf.src_top.data(),
       buf.src_top_left.data(), offset_val, buf.sao_offset.data(), wd, ht);
  }

  void InvokeBandOffset(ihevc_hbd_sao_band_offset_luma_ft* fn, SaoLumaBuffers<UWORD16>& buf) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(), buf.src_top.data(),
       buf.src_top_left.data(), offset_val, buf.sao_offset.data(), wd, ht, bit_depth);
  }

  // Unified runner for Luma Band Offset
  template <typename PixelType, typename FuncPtr>
  void RunBandOffsetImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    SaoLumaBuffers<PixelType> ref_buf = CreateSaoLumaBuffers<PixelType>(wd, ht, bit_depth);
    SaoLumaBuffers<PixelType> tst_buf = ref_buf;

    InvokeBandOffset(ref_fn, ref_buf);
    InvokeBandOffset(tst_fn, tst_buf);

    ASSERT_NO_FATAL_FAILURE(VerifySaoBuffers(ref_buf, tst_buf));
  }

  // ------------------------ Overloads for Edge Offset --------------------------

  void InvokeEdgeOffset(ihevc_sao_edge_offset_class0_ft* fn, SaoLumaBuffers<UWORD8>& buf,
                        int left_offset) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data() + left_offset,
       buf.src_top.data(), buf.src_top_left.data(), buf.src_top_right.data(),
       buf.src_bot_left.data(), buf.avail.data(), buf.sao_offset.data(), wd, ht);
  }

  void InvokeEdgeOffset(ihevc_hbd_sao_edge_offset_class0_ft* fn, SaoLumaBuffers<UWORD16>& buf,
                        int left_offset) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data() + left_offset,
       buf.src_top.data(), buf.src_top_left.data(), buf.src_top_right.data(),
       buf.src_bot_left.data(), buf.avail.data(), buf.sao_offset.data(), wd, ht, bit_depth);
  }

  // Unified runner for Luma Edge Offset
  template <typename PixelType, typename FuncPtr>
  void RunEdgeOffsetImpl(FuncPtr ref_fn, FuncPtr tst_fn, int edge_class) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    int left_offset = (edge_class == 2) ? 1 : 0;
    SaoLumaBuffers<PixelType> buf = CreateSaoLumaBuffers<PixelType>(wd, ht, bit_depth);

    for (int mask = 0; mask < 256; ++mask) {
      SCOPED_TRACE("avail mask " + std::to_string(mask));
      SetAvail(buf.avail, mask);
      SaoLumaBuffers<PixelType> ref_buf = buf;
      SaoLumaBuffers<PixelType> tst_buf = buf;

      InvokeEdgeOffset(ref_fn, ref_buf, left_offset);
      InvokeEdgeOffset(tst_fn, tst_buf, left_offset);

      ASSERT_NO_FATAL_FAILURE(VerifySaoBuffers(ref_buf, tst_buf));
    }
  }

  void RunBandOffset() {
    if (bit_depth == 8) {
      RunBandOffsetImpl<UWORD8>(GetSaoBandOffsetLumaFn(ref), GetSaoBandOffsetLumaFn(tst));
    } else {
      RunBandOffsetImpl<UWORD16>(GetHbdSaoBandOffsetLumaFn(ref), GetHbdSaoBandOffsetLumaFn(tst));
    }
  }

  void RunEdgeOffset(int edge_class) {
    if (bit_depth == 8) {
      RunEdgeOffsetImpl<UWORD8>(GetSaoEdgeOffsetLumaFn(ref, edge_class),
                                GetSaoEdgeOffsetLumaFn(tst, edge_class), edge_class);
    } else {
      RunEdgeOffsetImpl<UWORD16>(GetHbdSaoEdgeOffsetLumaFn(ref, edge_class),
                                 GetHbdSaoEdgeOffsetLumaFn(tst, edge_class), edge_class);
    }
  }

  int wd, ht, offset_val, bit_depth;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// --------------------------- Chroma Base Class -----------------------------

class SaoChromaTest : public ::testing::TestWithParam<SaoTestParam> {
 protected:
  void SetUp() override {
    std::pair<int, int> block_size;
    std::tie(block_size, offset_val, bit_depth, arch) = GetParam();
    std::tie(wd, ht) =
        block_size;  // wd is chroma component size, 2 * wd is bytes/samples

    ref = get_ref_func_ptr();
    tst = get_tst_func_ptr(arch);
  }

  // ------------------------ Overloads for Band Offset --------------------------

  void InvokeBandOffset(ihevc_sao_band_offset_chroma_ft* fn, SaoChromaBuffers<UWORD8>& buf,
                        int offset_val_u, int offset_val_v) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(), buf.src_top.data(),
       buf.src_top_left.data(), offset_val_u, offset_val_v, buf.sao_offset_u.data(),
       buf.sao_offset_v.data(), 2 * wd, ht);
  }

  void InvokeBandOffset(ihevc_hbd_sao_band_offset_chroma_ft* fn, SaoChromaBuffers<UWORD16>& buf,
                        int offset_val_u, int offset_val_v) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data(), buf.src_top.data(),
       buf.src_top_left.data(), offset_val_u, offset_val_v, buf.sao_offset_u.data(),
       buf.sao_offset_v.data(), 2 * wd, ht, bit_depth);
  }

  // Unified runner for Chroma Band Offset
  template <typename PixelType, typename FuncPtr>
  void RunBandOffsetImpl(FuncPtr ref_fn, FuncPtr tst_fn) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    int offset_val_u = offset_val;
    int offset_val_v = (offset_val + 4) % 32;

    SaoChromaBuffers<PixelType> ref_buf = CreateSaoChromaBuffers<PixelType>(wd, ht, bit_depth);
    SaoChromaBuffers<PixelType> tst_buf = ref_buf;

    InvokeBandOffset(ref_fn, ref_buf, offset_val_u, offset_val_v);
    InvokeBandOffset(tst_fn, tst_buf, offset_val_u, offset_val_v);

    ASSERT_NO_FATAL_FAILURE(VerifySaoBuffers(ref_buf, tst_buf));
  }

  // ------------------------ Overloads for Edge Offset -----------------------------

  void InvokeEdgeOffset(ihevc_sao_edge_offset_class0_chroma_ft* fn,
                        SaoChromaBuffers<UWORD8>& buf, int left_offset) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data() + left_offset,
       buf.src_top.data(), buf.src_top_left.data(), buf.src_top_right.data(),
       buf.src_bot_left.data(), buf.avail.data(), buf.sao_offset_u.data(), buf.sao_offset_v.data(),
       2 * wd, ht);
  }

  void InvokeEdgeOffset(ihevc_hbd_sao_edge_offset_class0_chroma_ft* fn,
                        SaoChromaBuffers<UWORD16>& buf, int left_offset) {
    fn(buf.src.data() + buf.src_offset, buf.stride, buf.src_left.data() + left_offset,
       buf.src_top.data(), buf.src_top_left.data(), buf.src_top_right.data(),
       buf.src_bot_left.data(), buf.avail.data(), buf.sao_offset_u.data(), buf.sao_offset_v.data(),
       2 * wd, ht, bit_depth);
  }

  // Unified runner for Chroma Edge Offset
  template <typename PixelType, typename FuncPtr>
  void RunEdgeOffsetImpl(FuncPtr ref_fn, FuncPtr tst_fn, int edge_class) {
    if (tst_fn == nullptr || tst_fn == ref_fn) {
      GTEST_SKIP() << "Architecture does not implement this function";
    }

    int left_offset = (edge_class == 2) ? 2 : 0;
    SaoChromaBuffers<PixelType> buf = CreateSaoChromaBuffers<PixelType>(wd, ht, bit_depth);

    for (int mask = 0; mask < 256; ++mask) {
      SCOPED_TRACE("avail mask " + std::to_string(mask));
      SetAvail(buf.avail, mask);
      SaoChromaBuffers<PixelType> ref_buf = buf;
      SaoChromaBuffers<PixelType> tst_buf = buf;

      InvokeEdgeOffset(ref_fn, ref_buf, left_offset);
      InvokeEdgeOffset(tst_fn, tst_buf, left_offset);

      ASSERT_NO_FATAL_FAILURE(VerifySaoBuffers(ref_buf, tst_buf));
    }
  }

  void RunBandOffset() {
    if (bit_depth == 8) {
      RunBandOffsetImpl<UWORD8>(GetSaoBandOffsetChromaFn(ref), GetSaoBandOffsetChromaFn(tst));
    } else {
      RunBandOffsetImpl<UWORD16>(GetHbdSaoBandOffsetChromaFn(ref), GetHbdSaoBandOffsetChromaFn(tst));
    }
  }

  void RunEdgeOffset(int edge_class) {
    if (bit_depth == 8) {
      RunEdgeOffsetImpl<UWORD8>(GetSaoEdgeOffsetChromaFn(ref, edge_class),
                                GetSaoEdgeOffsetChromaFn(tst, edge_class), edge_class);
    } else {
      RunEdgeOffsetImpl<UWORD16>(GetHbdSaoEdgeOffsetChromaFn(ref, edge_class),
                                 GetHbdSaoEdgeOffsetChromaFn(tst, edge_class), edge_class);
    }
  }

  int wd, ht, offset_val, bit_depth;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref;
  const ihevc_func_selector_t* tst;
};

// ---------------------------- Test cases -----------------------------------

class SaoBandOffsetLumaTest : public SaoLumaTest {};
TEST_P(SaoBandOffsetLumaTest, Run) { RunBandOffset(); }

class SaoBandOffsetChromaTest : public SaoChromaTest {};
TEST_P(SaoBandOffsetChromaTest, Run) { RunBandOffset(); }

class SaoEdgeOffsetClass0Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass0Test, Run) { RunEdgeOffset(0); }

class SaoEdgeOffsetClass0ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass0ChromaTest, Run) { RunEdgeOffset(0); }

class SaoEdgeOffsetClass1Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass1Test, Run) { RunEdgeOffset(1); }

class SaoEdgeOffsetClass1ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass1ChromaTest, Run) { RunEdgeOffset(1); }

class SaoEdgeOffsetClass2Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass2Test, Run) { RunEdgeOffset(2); }

class SaoEdgeOffsetClass2ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass2ChromaTest, Run) { RunEdgeOffset(2); }

class SaoEdgeOffsetClass3Test : public SaoLumaTest {};
TEST_P(SaoEdgeOffsetClass3Test, Run) { RunEdgeOffset(3); }

class SaoEdgeOffsetClass3ChromaTest : public SaoChromaTest {};
TEST_P(SaoEdgeOffsetClass3ChromaTest, Run) { RunEdgeOffset(3); }

// ---------------------------- Instantiation --------------------------------

std::vector<std::pair<int, int>> GetSaoLumaBlockSizes() {
  std::vector<std::pair<int, int>> sizes;
  for (const auto& size : getLumaPUBlockSizes()) {
    if (size.first >= 8 && size.first % 8 == 0 && size.second >= 4 && size.second % 4 == 0) {
      sizes.push_back(size);
    }
  }
  return sizes;
}

// Chroma component sizes for 4:2:0, 4:2:2 and 4:4:4.
std::set<std::pair<int, int>> GetSaoChromaBlockSizes() {
  std::set<std::pair<int, int>> sizes;
  for (const auto& size : getLumaPUBlockSizes()) {
    const std::pair<int, int> comps[] = {
        {size.first / 2, size.second / 2},  // 4:2:0
        {size.first / 2, size.second},      // 4:2:2
         size};                             // 4:4:4
    for (const auto& comp : comps) {
      if (comp.first >= 4 && comp.first % 4 == 0 && comp.second >= 4 && comp.second % 4 == 0) {
          sizes.insert(comp);
      }
    }
  }
  return sizes;
}

// Luma Params: block sizes, band position, bit_depth, arch
auto kSaoLumaParams = ::testing::Combine(
    ::testing::ValuesIn(GetSaoLumaBlockSizes()),
    ::testing::Values(0, 7, 15, 23, 28, 31),
    ::testing::Values(8, 10),
    ::testing::ValuesIn(getTstArch()));

// Chroma Params: block sizes, band position, bit_depth, arch
auto kSaoChromaParams = ::testing::Combine(
    ::testing::ValuesIn(GetSaoChromaBlockSizes()),
    ::testing::Values(0, 7, 15, 23, 28, 31),
    ::testing::Values(8, 10),
    ::testing::ValuesIn(getTstArch()));

// Edge tests don't use the band position; run each size once.
auto kSaoLumaEdgeParams = ::testing::Combine(
    ::testing::ValuesIn(GetSaoLumaBlockSizes()),
    ::testing::Values(0),
    ::testing::Values(8, 10),
    ::testing::ValuesIn(getTstArch()));

auto kSaoChromaEdgeParams = ::testing::Combine(
    ::testing::ValuesIn(GetSaoChromaBlockSizes()),
    ::testing::Values(0),
    ::testing::Values(8, 10),
    ::testing::ValuesIn(getTstArch()));

INSTANTIATE_TEST_SUITE_P(Sao, SaoBandOffsetLumaTest, kSaoLumaParams, PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoBandOffsetChromaTest, kSaoChromaParams, PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass0Test, kSaoLumaEdgeParams, PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass0ChromaTest,kSaoChromaEdgeParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass1Test, kSaoLumaEdgeParams, PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass1ChromaTest, kSaoChromaEdgeParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass2Test, kSaoLumaEdgeParams, PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass2ChromaTest, kSaoChromaEdgeParams,
                         PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass3Test, kSaoLumaEdgeParams, PrintSaoTestParam);
INSTANTIATE_TEST_SUITE_P(Sao, SaoEdgeOffsetClass3ChromaTest, kSaoChromaEdgeParams,
                         PrintSaoTestParam);
}  // namespace
