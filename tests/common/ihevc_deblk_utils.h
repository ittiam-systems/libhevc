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

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevc_deblk.h"
#include "ihevc_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"

// Buffer constants for deblocking tests and benchmarks
constexpr int kDeblkStride = 32;
constexpr int kDeblkBufHeight = 32;
constexpr int kDeblkBufSize = kDeblkStride * kDeblkBufHeight;
constexpr int kDeblkSrcOffset = 16 * kDeblkStride + 16;

// Function pointer types for 8-bit deblocking
using DeblkLumaFn = void (*)(UWORD8* pu1_src, WORD32 src_strd, WORD32 bs,
                             WORD32 quant_param_p, WORD32 quant_param_q,
                             WORD32 beta_offset_div2, WORD32 tc_offset_div2,
                             WORD32 filter_flag_p, WORD32 filter_flag_q);

using DeblkChromaFn = void (*)(UWORD8* pu1_src, WORD32 src_strd,
                               WORD32 quant_param_p, WORD32 quant_param_q,
                               WORD32 qp_offset_u, WORD32 qp_offset_v,
                               WORD32 tc_offset_div2, WORD32 filter_flag_p,
                               WORD32 filter_flag_q, WORD8 chroma_fmt_idc);

// Function pointer types for High Bit Depth (HBD) deblocking
using HbdDeblkLumaFn = void (*)(UWORD16* pu2_src, WORD32 src_strd, WORD32 bs,
                                WORD32 quant_param_p, WORD32 quant_param_q,
                                WORD32 beta_offset_div2, WORD32 tc_offset_div2,
                                WORD32 filter_flag_p, WORD32 filter_flag_q,
                                UWORD8 bit_depth);

using HbdDeblkChromaFn = void (*)(UWORD16* pu2_src, WORD32 src_strd,
                                  WORD32 quant_param_p, WORD32 quant_param_q,
                                  WORD32 qp_offset_u, WORD32 qp_offset_v,
                                  WORD32 tc_offset_div2, WORD32 filter_flag_p,
                                  WORD32 filter_flag_q, UWORD8 bit_depth,
                                  WORD8 chroma_fmt_idc);

// Configuration for Luma deblocking benchmark
struct DeblkLumaConfig {
  bool is_vert;
  int bs;
  int qp_p;
  int qp_q;
  int beta_offset;
  int tc_offset;
  int filter_p;
  int filter_q;
  IV_ARCH_T arch;
  int bit_depth = 8;
};

// Configuration for Chroma deblocking benchmark
struct DeblkChromaConfig {
  bool is_vert;
  int qp_p;
  int qp_q;
  int qp_offset_u;
  int qp_offset_v;
  int tc_offset;
  int filter_p;
  int filter_q;
  int chroma_fmt_idc;
  IV_ARCH_T arch;
  int bit_depth = 8;
};

// Returns target 8-bit Luma deblocking function pointer
DeblkLumaFn GetLumaTargetFn(const ihevc_func_selector_t* selector,
                            bool is_vert);

// Returns target 8-bit Chroma deblocking function pointer
DeblkChromaFn GetChromaTargetFn(const ihevc_func_selector_t* selector,
                                bool is_vert);

// Returns target HBD Luma deblocking function pointer
HbdDeblkLumaFn GetHbdLumaTargetFn(IV_ARCH_T arch, bool is_vert);

// Returns target HBD Chroma deblocking function pointer
HbdDeblkChromaFn GetHbdChromaTargetFn(IV_ARCH_T arch, bool is_vert);

// Initializes 8-bit deblocking buffer with deterministic random data
void InitializeDeblkBuffer(UWORD8* buf, size_t size);

// Initializes 16-bit HBD deblocking buffer with deterministic random data
void InitializeDeblkBuffer(UWORD16* buf, size_t size, int bit_depth = 8);
