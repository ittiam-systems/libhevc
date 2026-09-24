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

#include <random>

DeblkLumaFn GetLumaTargetFn(const ihevc_func_selector_t* selector,
                            bool is_vert) {
  if (!selector) {
    return nullptr;
  }
  return is_vert ? selector->ihevc_deblk_luma_vert_fptr
                 : selector->ihevc_deblk_luma_horz_fptr;
}

DeblkChromaFn GetChromaTargetFn(const ihevc_func_selector_t* selector,
                                bool is_vert) {
  if (!selector) {
    return nullptr;
  }
  return is_vert ? selector->ihevc_deblk_chroma_vert_fptr
                 : selector->ihevc_deblk_chroma_horz_fptr;
}

HbdDeblkLumaFn GetHbdLumaTargetFn(IV_ARCH_T arch, bool is_vert) {
  if (arch == ARCH_NA) {
    return is_vert ? ihevc_hbd_deblk_luma_vert : ihevc_hbd_deblk_luma_horz;
  }
  // Currently HBD deblocking only has C reference implementations.
  // SIMD implementations can be plugged in here when available.
  return nullptr;
}

HbdDeblkChromaFn GetHbdChromaTargetFn(IV_ARCH_T arch, bool is_vert) {
  if (arch == ARCH_NA) {
    return is_vert ? ihevc_hbd_deblk_chroma_vert : ihevc_hbd_deblk_chroma_horz;
  }
  // Currently HBD deblocking only has C reference implementations.
  // SIMD implementations can be plugged in here when available.
  return nullptr;
}

void InitializeDeblkBuffer(UWORD8* buf, size_t size) {
  FillRandom(buf, size, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 42);
}

void InitializeDeblkBuffer(UWORD16* buf, size_t size, int bit_depth) {
  FillRandom(buf, size, static_cast<UWORD16>(0),
             static_cast<UWORD16>((1 << bit_depth) - 1), 42);
}
