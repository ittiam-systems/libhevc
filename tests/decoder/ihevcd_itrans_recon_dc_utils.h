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

#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
extern "C" {
#include "ihevcd_itrans_recon_dc.h"
#include "ihevcd_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "TestCommon.h"
#include "dec_func_selector.h"

// Configuration for ITrans Recon DC buffers
struct ITransReconDCConfig {
  int trans_size;
  int bit_depth;
  bool is_chroma;  // chroma: U and V interleaved, so a row holds 2 samples per pixel
  int plane;       // chroma only: 0 for U, 1 for V
};

template <typename T>
struct ITransReconDCBuffers {
  std::vector<T> dst;  // holds pred on input, recon on output
  WORD32 pred_strd;
  WORD32 dst_strd;
  int offset;

  T* pred_ptr() { return dst.data() + offset; }
  T* dst_ptr() { return dst.data() + offset; }
};

template <typename T>
ITransReconDCBuffers<T> CreateITransReconDCBuffers(const ITransReconDCConfig& cfg) {
  const int pad = 16;
  const T max_val = static_cast<T>((1 << cfg.bit_depth) - 1);

  ITransReconDCBuffers<T> b;
  b.pred_strd = (cfg.is_chroma ? 2 : 1) * cfg.trans_size + pad;
  b.dst_strd = b.pred_strd;
  b.offset = cfg.plane;

  // Guard row below the block catches writes past the last row
  b.dst.resize(b.dst_strd * (cfg.trans_size + 1));

  FillRandom(b.dst, static_cast<T>(0), max_val, 42);
  return b;
}
