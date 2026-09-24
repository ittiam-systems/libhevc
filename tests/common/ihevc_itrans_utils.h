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

#include <string>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevc_itrans.h"
#include "ihevc_itrans_res.h"
#include "ihevc_itrans_recon.h"
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_recon.h"
#include "ihevc_chroma_recon.h"
#include "ihevc_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"

using ITransFn = void (*)(WORD16* pi2_src, WORD16* pi2_dst, WORD32 src_strd,
                          WORD32 dst_strd, WORD32 shift, WORD32 zero_cols);

using ITransResFn = void (*)(WORD16* pi2_src, WORD16* pi2_tmp, WORD16* pi2_dst,
                             WORD32 src_strd, WORD32 dst_strd, WORD32 zero_cols,
                             WORD32 zero_rows, UWORD8 bit_depth);

using ITransReconFn = void (*)(WORD16* pi2_src, WORD16* pi2_tmp,
                               UWORD8* pu1_pred, UWORD8* pu1_dst,
                               WORD32 src_strd, WORD32 pred_strd,
                               WORD32 dst_strd, WORD32 zero_cols,
                               WORD32 zero_rows);

using ChromaITransReconFn = void (*)(WORD16* pi2_src, WORD16* pi2_tmp,
                                     UWORD8* pu1_pred, UWORD8* pu1_dst,
                                     WORD32 src_strd, WORD32 pred_strd,
                                     WORD32 dst_strd, WORD32 zero_cols,
                                     WORD32 zero_rows);

using HbdITransReconFn = void (*)(WORD16* pi2_src, WORD16* pi2_tmp,
                                  UWORD16* pu2_pred, UWORD16* pu2_dst,
                                  WORD32 src_strd, WORD32 pred_strd,
                                  WORD32 dst_strd, WORD32 zero_cols,
                                  WORD32 zero_rows, UWORD8 bit_depth);

using ReconFn = void (*)(WORD16* pi2_src, UWORD8* pu1_pred, UWORD8* pu1_dst,
                         WORD32 src_strd, WORD32 pred_strd, WORD32 dst_strd,
                         WORD32 zero_cols);

using ChromaReconFn = void (*)(WORD16* pi2_src, UWORD8* pu1_pred,
                               UWORD8* pu1_dst, WORD32 src_strd,
                               WORD32 pred_strd, WORD32 dst_strd,
                               WORD32 zero_cols);

ITransFn GetITransFn(const ihevc_func_selector_t* selector, int trans_size,
                     int ttype);
ITransResFn GetITransResFn(const ihevc_func_selector_t* selector,
                           int trans_size, int ttype);
ITransReconFn GetITransReconFn(const ihevc_func_selector_t* selector,
                               int trans_size, int ttype);
ChromaITransReconFn GetChromaITransReconFn(
    const ihevc_func_selector_t* selector, int trans_size);
HbdITransReconFn GetHbdITransReconFn(IV_ARCH_T arch, int trans_size, int ttype);
ReconFn GetReconFn(const ihevc_func_selector_t* selector, int trans_size,
                   int ttype);
ChromaReconFn GetChromaReconFn(const ihevc_func_selector_t* selector,
                               int trans_size);

// Non-zero transform coefficient region dimensions
struct NonZeroRegion {
  int cols;
  int rows;
};

// Benchmark test case definition for transform size and type
struct ITransTestCase {
  int size;
  int ttype;
  std::vector<NonZeroRegion> nz_regions;
};

// Benchmark configuration for inverse transform tests
struct ITransBenchConfig {
  int trans_size;
  int ttype;          // 0: DCT, 1: DST (4x4 only)
  int non_zero_cols;  // non-zero width
  int non_zero_rows;  // non-zero height
  IV_ARCH_T arch;
  int bit_depth = 8;
};

// Generates inputs using ihevc_resi_trans_* by computing the difference between
// two [0, 255] inputs and performing forward transform, then zeroing out
// outside the nz_cols x nz_rows non-zero region.
void GenerateITransInput(int trans_size, int ttype, int nz_cols, int nz_rows,
                         WORD16* pi2_coeffs, UWORD8* pu1_pred_recon,
                         WORD32* zero_cols, WORD32* zero_rows);

// Generates inputs for High Bit Depth (HBD) inverse transform benchmarks with
// prediction values in [0, (1 << bit_depth) - 1].
void GenerateITransInput(int trans_size, int ttype, int nz_cols, int nz_rows,
                         WORD16* pi2_coeffs, UWORD16* pu2_pred_recon,
                         WORD32* zero_cols, WORD32* zero_rows,
                         int bit_depth = 10);

// Returns common test case configurations for inverse transform benchmarks.
const std::vector<ITransTestCase>& GetITransTestCases();
