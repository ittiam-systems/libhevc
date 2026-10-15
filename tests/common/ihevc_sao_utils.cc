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

#include "ihevc_sao_utils.h"

#include "func_selector.h"

ihevc_sao_band_offset_luma_ft* GetSaoBandOffsetLumaFn(const ihevc_func_selector_t* selector) {
  return selector ? selector->ihevc_sao_band_offset_luma_fptr : nullptr;
}

ihevc_sao_band_offset_chroma_ft* GetSaoBandOffsetChromaFn(const ihevc_func_selector_t* selector) {
  return selector ? selector->ihevc_sao_band_offset_chroma_fptr : nullptr;
}

ihevc_sao_edge_offset_class0_ft* GetSaoEdgeOffsetLumaFn(
    const ihevc_func_selector_t* selector, int edge_class) {
  if (!selector) return nullptr;
  switch (edge_class) {
    case 0:
      return selector->ihevc_sao_edge_offset_class0_fptr;
    case 1:
      return selector->ihevc_sao_edge_offset_class1_fptr;
    case 2:
      return selector->ihevc_sao_edge_offset_class2_fptr;
    case 3:
      return selector->ihevc_sao_edge_offset_class3_fptr;
    default:
      return nullptr;
  }
}

ihevc_sao_edge_offset_class0_chroma_ft* GetSaoEdgeOffsetChromaFn(
    const ihevc_func_selector_t* selector, int edge_class) {
  if (!selector) return nullptr;
  switch (edge_class) {
    case 0:
      return selector->ihevc_sao_edge_offset_class0_chroma_fptr;
    case 1:
      return selector->ihevc_sao_edge_offset_class1_chroma_fptr;
    case 2:
      return selector->ihevc_sao_edge_offset_class2_chroma_fptr;
    case 3:
      return selector->ihevc_sao_edge_offset_class3_chroma_fptr;
    default:
      return nullptr;
  }
}

ihevc_hbd_sao_band_offset_luma_ft* GetHbdSaoBandOffsetLumaFn(
    const ihevc_func_selector_t* selector) {
  return selector ? selector->ihevc_hbd_sao_band_offset_luma_fptr : nullptr;
}

ihevc_hbd_sao_band_offset_chroma_ft* GetHbdSaoBandOffsetChromaFn(
    const ihevc_func_selector_t* selector) {
  return selector ? selector->ihevc_hbd_sao_band_offset_chroma_fptr : nullptr;
}

ihevc_hbd_sao_edge_offset_class0_ft* GetHbdSaoEdgeOffsetLumaFn(
    const ihevc_func_selector_t* selector, int edge_class) {
  if (!selector) return nullptr;
  switch (edge_class) {
    case 0:
      return selector->ihevc_hbd_sao_edge_offset_class0_fptr;
    case 1:
      return selector->ihevc_hbd_sao_edge_offset_class1_fptr;
    case 2:
      return selector->ihevc_hbd_sao_edge_offset_class2_fptr;
    case 3:
      return selector->ihevc_hbd_sao_edge_offset_class3_fptr;
    default:
      return nullptr;
  }
}

ihevc_hbd_sao_edge_offset_class0_chroma_ft* GetHbdSaoEdgeOffsetChromaFn(
    const ihevc_func_selector_t* selector, int edge_class) {
  if (!selector) return nullptr;
  switch (edge_class) {
    case 0:
      return selector->ihevc_hbd_sao_edge_offset_class0_chroma_fptr;
    case 1:
      return selector->ihevc_hbd_sao_edge_offset_class1_chroma_fptr;
    case 2:
      return selector->ihevc_hbd_sao_edge_offset_class2_chroma_fptr;
    case 3:
      return selector->ihevc_hbd_sao_edge_offset_class3_chroma_fptr;
    default:
      return nullptr;
  }
}

ihevc_hbd_sao_band_offset_luma_ft* GetHbdSaoBandOffsetLumaFn(IV_ARCH_T arch) {
  return GetHbdSaoBandOffsetLumaFn(get_func_ptr(arch));
}

ihevc_hbd_sao_band_offset_chroma_ft* GetHbdSaoBandOffsetChromaFn(IV_ARCH_T arch) {
  return GetHbdSaoBandOffsetChromaFn(get_func_ptr(arch));
}

ihevc_hbd_sao_edge_offset_class0_ft* GetHbdSaoEdgeOffsetLumaFn(IV_ARCH_T arch, int edge_class) {
  return GetHbdSaoEdgeOffsetLumaFn(get_func_ptr(arch), edge_class);
}

ihevc_hbd_sao_edge_offset_class0_chroma_ft* GetHbdSaoEdgeOffsetChromaFn(IV_ARCH_T arch,
                                                                        int edge_class) {
  return GetHbdSaoEdgeOffsetChromaFn(get_func_ptr(arch), edge_class);
}
