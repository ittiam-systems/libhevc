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

using ITransFn = ihevc_itrans_4x4_ft*;
using ITransResFn = ihevc_itrans_res_4x4_ft*;
using ITransReconFn = ihevc_itrans_recon_4x4_ft*;
using ChromaITransReconFn = ihevc_chroma_itrans_recon_4x4_ft*;
using HbdITransReconFn = ihevc_hbd_itrans_recon_4x4_ft*;
using HbdChromaITransReconFn = ihevc_hbd_chroma_itrans_recon_4x4_ft*;
using ReconFn = ihevc_recon_4x4_ft*;
using ChromaReconFn = ihevc_chroma_recon_4x4_ft*;

ITransFn GetITransFn(const ihevc_func_selector_t* selector, int trans_size,
                     int ttype);
ITransResFn GetITransResFn(const ihevc_func_selector_t* selector,
                           int trans_size, int ttype);
ITransReconFn GetITransReconFn(const ihevc_func_selector_t* selector,
                               int trans_size, int ttype);
ITransReconFn GetITransReconFn(IV_ARCH_T arch, int trans_size, int ttype);
ChromaITransReconFn GetChromaITransReconFn(
    const ihevc_func_selector_t* selector, int trans_size);
ChromaITransReconFn GetChromaITransReconFn(IV_ARCH_T arch, int trans_size);
HbdITransReconFn GetHbdITransReconFn(const ihevc_func_selector_t* selector, int trans_size,
                                     int ttype);
HbdITransReconFn GetHbdITransReconFn(IV_ARCH_T arch, int trans_size, int ttype);
HbdChromaITransReconFn GetHbdChromaITransReconFn(const ihevc_func_selector_t* selector,
                                                 int trans_size);
HbdChromaITransReconFn GetHbdChromaITransReconFn(IV_ARCH_T arch, int trans_size);
ReconFn GetReconFn(const ihevc_func_selector_t* selector, int trans_size,
                   int ttype);
ChromaReconFn GetChromaReconFn(const ihevc_func_selector_t* selector,
                               int trans_size);

// Configuration for ITrans Recon buffers
struct ITransReconConfig {
  int trans_size;
  int nz_rows;     // coefficients lie in the top-left nz_rows x nz_cols region
  int nz_cols;
  int bit_depth;
  bool is_chroma;  // chroma: U and V interleaved, so a row holds 2 samples per pixel
  int plane;       // chroma only: 0 for U, 1 for V
  bool sparse;     // scattered coefficients, giving zero masks with holes
  bool full_range; // coefficients over the whole WORD16 range (saturation paths)
};

// Buffers structure for ITrans Recon operations.
template <typename T>
struct ITransReconBuffers {
  std::vector<WORD16> src;
  std::vector<WORD16> tmp;
  std::vector<T> dst;  // holds pred on input, recon on output
  WORD32 src_strd;
  WORD32 pred_strd;
  WORD32 dst_strd;
  WORD32 zero_cols;
  WORD32 zero_rows;
  int offset;

  T* pred_ptr() { return dst.data() + offset; }
  T* dst_ptr() { return dst.data() + offset; }
};

// Creates ITrans Recon buffers.
template <typename T>
ITransReconBuffers<T> CreateITransReconBuffers(const ITransReconConfig& cfg);

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
