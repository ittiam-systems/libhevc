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

#include "ihevc_padding_utils.h"

#include <vector>

namespace {

// Samples (and rows) of guard band around the buffer, so writes past the padding are caught
constexpr int kGuard = 16;

}  // namespace

const char* GetPadTypeStr(PadType pad_type) {
  switch (pad_type) {
    case kPadLeft: return "left";
    case kPadRight: return "right";
    case kPadTop: return "top";
    case kPadBottom: return "bottom";
    case kPadHorz: return "horz";
    case kPadVert: return "vert";
    default: return "unknown";
  }
}

PadFn GetPadFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadLeft: return selector->ihevc_pad_left_luma_fptr;
    case kPadRight: return selector->ihevc_pad_right_luma_fptr;
    case kPadTop: return selector->ihevc_pad_top_fptr;
    case kPadBottom: return selector->ihevc_pad_bottom_fptr;
    default: return nullptr;
  }
}

PadFn GetChromaPadFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadLeft: return selector->ihevc_pad_left_chroma_fptr;
    case kPadRight: return selector->ihevc_pad_right_chroma_fptr;
    case kPadTop: return selector->ihevc_pad_top_fptr;
    case kPadBottom: return selector->ihevc_pad_bottom_fptr;
    default: return nullptr;
  }
}

HbdPadFn GetHbdPadFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadLeft: return selector->ihevc_hbd_pad_left_luma_fptr;
    case kPadRight: return selector->ihevc_hbd_pad_right_luma_fptr;
    case kPadTop: return selector->ihevc_hbd_pad_top_fptr;
    case kPadBottom: return selector->ihevc_hbd_pad_bottom_fptr;
    default: return nullptr;
  }
}

HbdPadFn GetHbdChromaPadFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadLeft: return selector->ihevc_hbd_pad_left_chroma_fptr;
    case kPadRight: return selector->ihevc_hbd_pad_right_chroma_fptr;
    case kPadTop: return selector->ihevc_hbd_pad_top_fptr;
    case kPadBottom: return selector->ihevc_hbd_pad_bottom_fptr;
    default: return nullptr;
  }
}

PadBlockFn GetPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadHorz: return selector->ihevc_pad_horz_luma_fptr;
    case kPadVert: return selector->ihevc_pad_vert_fptr;
    default: return nullptr;
  }
}

PadBlockFn GetChromaPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadHorz: return selector->ihevc_pad_horz_chroma_fptr;
    case kPadVert: return selector->ihevc_pad_vert_fptr;
    default: return nullptr;
  }
}

HbdPadBlockFn GetHbdPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadHorz: return selector->ihevc_hbd_pad_horz_luma_fptr;
    case kPadVert: return selector->ihevc_hbd_pad_vert_fptr;
    default: return nullptr;
  }
}

HbdPadBlockFn GetHbdChromaPadBlockFn(const ihevc_func_selector_t* selector, PadType pad_type) {
  if (!selector) return nullptr;
  switch (pad_type) {
    case kPadHorz: return selector->ihevc_hbd_pad_horz_chroma_fptr;
    case kPadVert: return selector->ihevc_hbd_pad_vert_fptr;
    default: return nullptr;
  }
}

template <typename T>
PaddingBuffers<T> CreatePaddingBuffers(const PaddingConfig& cfg) {
  const bool is_horz =
      (cfg.pad_type == kPadLeft) || (cfg.pad_type == kPadRight) || (cfg.pad_type == kPadHorz);
  const int row_samples = (cfg.is_chroma ? 2 : 1) * cfg.wd;
  // Room for the padding on both sides of the padded direction only
  const int pad_cols = is_horz ? cfg.pad_size : 0;
  const int pad_rows = is_horz ? 0 : cfg.pad_size;
  const T max_val = static_cast<T>((1 << cfg.bit_depth) - 1);

  PaddingBuffers<T> b;
  b.strd = row_samples + 2 * (pad_cols + kGuard);
  b.len = is_horz ? cfg.ht : row_samples;
  b.row_wd = row_samples;
  b.buf.resize(b.strd * (cfg.ht + 2 * (pad_rows + kGuard)));

  const int blk_offset = (pad_rows + kGuard) * b.strd + pad_cols + kGuard;
  switch (cfg.pad_type) {
    case kPadRight: b.offset = blk_offset + row_samples; break;
    case kPadBottom: b.offset = blk_offset + cfg.ht * b.strd; break;
    default: b.offset = blk_offset; break;
  }

  // Random block, padding and guard band, so stray writes and wrong source samples are caught
  FillRandom(b.buf, static_cast<T>(0), max_val, 42);
  return b;
}

// Explicit instantiations for 8-bit and HBD
template PaddingBuffers<UWORD8> CreatePaddingBuffers<UWORD8>(const PaddingConfig& cfg);
template PaddingBuffers<UWORD16> CreatePaddingBuffers<UWORD16>(const PaddingConfig& cfg);
