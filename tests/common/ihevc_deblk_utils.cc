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

#include "ihevc_deblk_utils.h"
#include "func_selector.h"
#include <algorithm>
#include <random>

DeblkLumaFn GetLumaTargetFn(const ihevc_func_selector_t* selector, bool is_vert) {
  if (!selector) {
    return nullptr;
  }
  return is_vert ? selector->ihevc_deblk_luma_vert_fptr : selector->ihevc_deblk_luma_horz_fptr;
}

DeblkChromaFn GetChromaTargetFn(const ihevc_func_selector_t* selector, bool is_vert,
                                int chroma_fmt_idc) {
  if (!selector) {
    return nullptr;
  }
  if (chroma_fmt_idc == 2) {
    return is_vert ? selector->ihevc_deblk_422chroma_vert_fptr
                   : selector->ihevc_deblk_422chroma_horz_fptr;
  }
  return is_vert ? selector->ihevc_deblk_chroma_vert_fptr : selector->ihevc_deblk_chroma_horz_fptr;
}

HbdDeblkLumaFn GetHbdLumaTargetFn(const ihevc_func_selector_t* selector, bool is_vert) {
  if (!selector) {
    return nullptr;
  }
  return is_vert ? selector->ihevc_hbd_deblk_luma_vert_fptr
                 : selector->ihevc_hbd_deblk_luma_horz_fptr;
}

HbdDeblkChromaFn GetHbdChromaTargetFn(const ihevc_func_selector_t* selector, bool is_vert,
                                      int chroma_fmt_idc) {
  if (!selector) {
    return nullptr;
  }
  if (chroma_fmt_idc == 2) {
    return is_vert ? selector->ihevc_hbd_deblk_422chroma_vert_fptr
                   : selector->ihevc_hbd_deblk_422chroma_horz_fptr;
  }
  return is_vert ? selector->ihevc_hbd_deblk_chroma_vert_fptr
                 : selector->ihevc_hbd_deblk_chroma_horz_fptr;
}

namespace {
// Dual-edge buffer for benchmarks (quadrant Q at bottom-right) supporting both edge orientations
template <typename PixelType>
void InitializeDeblkDualEdgeBufferImpl(PixelType* buf, size_t size, int bit_depth) {
  std::mt19937 rng(42);
  const int stride = kDeblkStride;
  const int height = static_cast<int>(size / stride);
  const int max_val = (1 << bit_depth) - 1;
  const int scale = 1 << (bit_depth - 8);

  // General random base_p and boundary step
  std::uniform_int_distribution<int> p_dist(0, max_val);
  int base_p = p_dist(rng);

  std::uniform_int_distribution<int> step_dist(-24 * scale, 24 * scale);
  int step = step_dist(rng);
  if (step == 0) {
    step = (rng() % 2 == 0) ? (2 * scale) : (-2 * scale);
  }
  int base_q = std::clamp(base_p + step, 0, max_val);

  std::uniform_int_distribution<int> noise_dist(-1, 1);

  constexpr int kEdgeCol = 16;
  constexpr int kEdgeRow = 16;

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < stride; ++c) {
      bool is_q = (r >= kEdgeRow && c >= kEdgeCol);
      int base = is_q ? base_q : base_p;
      int val = std::clamp(base + noise_dist(rng), 0, max_val);
      buf[r * stride + c] = static_cast<PixelType>(val);
    }
  }
}

}  // namespace

template <typename PixelType>
void InitializeDeblkBuffer(PixelType* buf, size_t size, bool is_vert, int bit_depth,
                           std::mt19937& rng) {
  const int stride = kDeblkStride;
  const int height = static_cast<int>(size / stride);
  const int max_val = (1 << bit_depth) - 1;
  const int scale = 1 << (bit_depth - 8);

  std::uniform_int_distribution<int> p_dist(0, max_val);
  int base_p = p_dist(rng);

  // Random step across the edge boundary to hit strong/weak filtering paths
  std::uniform_int_distribution<int> step_dist(-24 * scale, 24 * scale);
  int step = step_dist(rng);
  if (step == 0) {
    step = (rng() % 2 == 0) ? (2 * scale) : (-2 * scale);
  }
  int base_q = std::clamp(base_p + step, 0, max_val);

  std::uniform_int_distribution<int> noise_dist(-1, 1);

  constexpr int kEdgeCol = 16;
  constexpr int kEdgeRow = 16;

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < stride; ++c) {
      bool is_p = is_vert ? (c < kEdgeCol) : (r < kEdgeRow);
      int base = is_p ? base_p : base_q;
      int val = std::clamp(base + noise_dist(rng), 0, max_val);
      buf[r * stride + c] = static_cast<PixelType>(val);
    }
  }
}

template <typename PixelType>
void SetDeblkEdgeLine(PixelType* buf, bool is_vert, int line_idx, int p3, int p2, int p1, int p0,
                      int q0, int q1, int q2, int q3) {
  if (is_vert) {
    PixelType* pu = buf + kDeblkSrcOffset + line_idx * kDeblkStride;
    pu[-4] = static_cast<PixelType>(p3);
    pu[-3] = static_cast<PixelType>(p2);
    pu[-2] = static_cast<PixelType>(p1);
    pu[-1] = static_cast<PixelType>(p0);
    pu[0] = static_cast<PixelType>(q0);
    pu[1] = static_cast<PixelType>(q1);
    pu[2] = static_cast<PixelType>(q2);
    pu[3] = static_cast<PixelType>(q3);
  } else {
    PixelType* pu = buf + kDeblkSrcOffset + line_idx;
    pu[-4 * kDeblkStride] = static_cast<PixelType>(p3);
    pu[-3 * kDeblkStride] = static_cast<PixelType>(p2);
    pu[-2 * kDeblkStride] = static_cast<PixelType>(p1);
    pu[-1 * kDeblkStride] = static_cast<PixelType>(p0);
    pu[0 * kDeblkStride] = static_cast<PixelType>(q0);
    pu[1 * kDeblkStride] = static_cast<PixelType>(q1);
    pu[2 * kDeblkStride] = static_cast<PixelType>(q2);
    pu[3 * kDeblkStride] = static_cast<PixelType>(q3);
  }
}

template <typename PixelType>
void FillDeblkEdge(PixelType* buf, bool is_vert, int p3, int p2, int p1, int p0, int q0, int q1,
                   int q2, int q3) {
  for (int line = 0; line < 4; ++line) {
    SetDeblkEdgeLine(buf, is_vert, line, p3, p2, p1, p0, q0, q1, q2, q3);
  }
}

// Explicit template instantiations
template void InitializeDeblkBuffer(UWORD8* buf, size_t size, bool is_vert, int bit_depth,
                                    std::mt19937& rng);
template void InitializeDeblkBuffer(UWORD16* buf, size_t size, bool is_vert, int bit_depth,
                                    std::mt19937& rng);
template void SetDeblkEdgeLine(UWORD8* buf, bool is_vert, int line_idx, int p3, int p2, int p1,
                               int p0, int q0, int q1, int q2, int q3);
template void SetDeblkEdgeLine(UWORD16* buf, bool is_vert, int line_idx, int p3, int p2, int p1,
                               int p0, int q0, int q1, int q2, int q3);
template void FillDeblkEdge(UWORD8* buf, bool is_vert, int p3, int p2, int p1, int p0, int q0,
                            int q1, int q2, int q3);
template void FillDeblkEdge(UWORD16* buf, bool is_vert, int p3, int p2, int p1, int p0, int q0,
                            int q1, int q2, int q3);

void InitializeDeblkBuffer(UWORD8* buf, size_t size) {
  InitializeDeblkDualEdgeBufferImpl(buf, size, 8);
}

void InitializeDeblkBuffer(UWORD16* buf, size_t size, int bit_depth) {
  InitializeDeblkDualEdgeBufferImpl(buf, size, bit_depth);
}

HbdDeblkLumaFn GetHbdLumaTargetFn(IV_ARCH_T arch, bool is_vert) {
  return GetHbdLumaTargetFn(get_func_ptr(arch), is_vert);
}

HbdDeblkChromaFn GetHbdChromaTargetFn(IV_ARCH_T arch, bool is_vert, int chroma_fmt_idc) {
  return GetHbdChromaTargetFn(get_func_ptr(arch), is_vert, chroma_fmt_idc);
}

bool ExpectDeblkLumaFilter(int bs, int qp_p, int qp_q, int beta_offset, int tc_offset,
                           const std::pair<int, int>& filter_pair) {
  if (bs <= 0 || (!filter_pair.first && !filter_pair.second)) {
    return false;
  }
  int qp_luma = (qp_p + qp_q + 1) >> 1;
  int beta_indx = std::clamp(qp_luma + (beta_offset << 1), 0, 51);
  int tc_indx = std::clamp(qp_luma + (2 * (bs >> 1)) + (tc_offset << 1), 0, 53);
  return (gai4_ihevc_beta_table[beta_indx] > 0) && (gai4_ihevc_tc_table[tc_indx] > 0);
}

bool ExpectDeblkChromaFilter(int qp_p, int qp_q, int qp_offset_u, int qp_offset_v, int tc_offset,
                             const std::pair<int, int>& filter_pair, int chroma_fmt_idc) {
  if (!filter_pair.first && !filter_pair.second) {
    return false;
  }
  int qp_u = qp_offset_u + ((qp_p + qp_q + 1) >> 1);
  int qp_v = qp_offset_v + ((qp_p + qp_q + 1) >> 1);
  bool is_422_or_444 = (chroma_fmt_idc == CHROMA_FMT_IDC_YUV444 ||
                        chroma_fmt_idc == CHROMA_FMT_IDC_YUV422);
  int qp_c_u = is_422_or_444 ? std::min(qp_u, 51)
                             : (qp_u < 0 ? qp_u : (qp_u > 57 ? qp_u - 6
                                                             : gai4_ihevc_qp_table[qp_u]));
  int qp_c_v = is_422_or_444 ? std::min(qp_v, 51)
                             : (qp_v < 0 ? qp_v : (qp_v > 57 ? qp_v - 6
                                                             : gai4_ihevc_qp_table[qp_v]));
  int tc_u_indx = std::clamp(qp_c_u + 2 + (tc_offset << 1), 0, 53);
  int tc_v_indx = std::clamp(qp_c_v + 2 + (tc_offset << 1), 0, 53);
  return (gai4_ihevc_tc_table[tc_u_indx] > 0) || (gai4_ihevc_tc_table[tc_v_indx] > 0);
}
