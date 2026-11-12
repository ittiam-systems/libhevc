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
#include "ihevc_padding.h"
#include "ihevc_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "TestCommon.h"

enum PadType {
  kPadLeft,
  kPadRight,
  kPadTop,
  kPadBottom,
  kPadHorz,
  kPadVert,
};

const char* GetPadTypeStr(PadType pad_type);

// True for horz/vert, which pad both sides of the block and take ht and wd
inline bool IsBlockPad(PadType pad_type) { return pad_type == kPadHorz || pad_type == kPadVert; }

// Left/right/top/bottom share one signature: the third argument is ht for left/right and the row
// width for top/bottom. T is UWORD8 for 8-bit and UWORD16 for HBD
template <typename T>
using PadFnT = void (*)(T*, WORD32, WORD32, WORD32);
using PadFn = PadFnT<UWORD8>;
using HbdPadFn = PadFnT<UWORD16>;

// Horz/vert share one signature: (src, src_strd, ht, wd, pad_size)
template <typename T>
using PadBlockFnT = void (*)(T*, WORD32, WORD32, WORD32, WORD32);
using PadBlockFn = PadBlockFnT<UWORD8>;
using HbdPadBlockFn = PadBlockFnT<UWORD16>;

// Top/bottom/vert pad luma and chroma rows alike, so luma and chroma return the same function
PadFn GetPadFn(const ihevc_func_selector_t* selector, PadType pad_type);
PadFn GetChromaPadFn(const ihevc_func_selector_t* selector, PadType pad_type);
HbdPadFn GetHbdPadFn(const ihevc_func_selector_t* selector, PadType pad_type);
HbdPadFn GetHbdChromaPadFn(const ihevc_func_selector_t* selector, PadType pad_type);
PadBlockFn GetPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type);
PadBlockFn GetChromaPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type);
HbdPadBlockFn GetHbdPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type);
HbdPadBlockFn GetHbdChromaPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type);

// Configuration for Padding buffers
struct PaddingConfig {
  PadType pad_type;
  int wd;
  int ht;
  int pad_size;
  int bit_depth;
  bool is_chroma;
};

// Buffers structure for Padding operations.
template <typename T>
struct PaddingBuffers {
  std::vector<T> buf;
  WORD32 strd;
  WORD32 len;
  WORD32 row_wd;
  int offset;
  T* pad_ptr() { return buf.data() + offset; }
};

// Creates Padding buffers.
template <typename T>
PaddingBuffers<T> CreatePaddingBuffers(const PaddingConfig& cfg);
