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

ihevc_sao_band_offset_luma_ft* GetSaoBandOffsetLumaFn(
    const ihevc_func_selector_t* selector) {
  return selector ? selector->ihevc_sao_band_offset_luma_fptr : nullptr;
}

ihevc_sao_band_offset_chroma_ft* GetSaoBandOffsetChromaFn(
    const ihevc_func_selector_t* selector) {
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

ihevc_hbd_sao_band_offset_luma_ft* GetHbdSaoBandOffsetLumaFn(IV_ARCH_T arch) {
  if (arch == ARCH_NA) {
    return ihevc_hbd_sao_band_offset_luma;
  }
  // Future SIMD implementations can be added here
  return nullptr;
}

ihevc_hbd_sao_band_offset_chroma_ft* GetHbdSaoBandOffsetChromaFn(
    IV_ARCH_T arch) {
  if (arch == ARCH_NA) {
    return ihevc_hbd_sao_band_offset_chroma;
  }
  // Future SIMD implementations can be added here
  return nullptr;
}

ihevc_hbd_sao_edge_offset_class0_ft* GetHbdSaoEdgeOffsetLumaFn(IV_ARCH_T arch,
                                                               int edge_class) {
  if (arch == ARCH_NA) {
    switch (edge_class) {
      case 0:
        return ihevc_hbd_sao_edge_offset_class0;
      case 1:
        return ihevc_hbd_sao_edge_offset_class1;
      case 2:
        return ihevc_hbd_sao_edge_offset_class2;
      case 3:
        return ihevc_hbd_sao_edge_offset_class3;
      default:
        return nullptr;
    }
  }
  // Future SIMD implementations can be added here
  return nullptr;
}

ihevc_hbd_sao_edge_offset_class0_chroma_ft* GetHbdSaoEdgeOffsetChromaFn(
    IV_ARCH_T arch, int edge_class) {
  if (arch == ARCH_NA) {
    switch (edge_class) {
      case 0:
        return ihevc_hbd_sao_edge_offset_class0_chroma;
      case 1:
        return ihevc_hbd_sao_edge_offset_class1_chroma;
      case 2:
        return ihevc_hbd_sao_edge_offset_class2_chroma;
      case 3:
        return ihevc_hbd_sao_edge_offset_class3_chroma;
      default:
        return nullptr;
    }
  }
  // Future SIMD implementations can be added here
  return nullptr;
}
