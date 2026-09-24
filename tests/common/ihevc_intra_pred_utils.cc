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

#include "ihevc_intra_pred_utils.h"

LumaIntraPredFn GetLumaIntraPredFn(const ihevc_func_selector_t* selector,
                                   int mode) {
  if (!selector) return nullptr;
  if (mode == 0) return selector->ihevc_intra_pred_luma_planar_fptr;
  if (mode == 1) return selector->ihevc_intra_pred_luma_dc_fptr;
  if (mode == 2) return selector->ihevc_intra_pred_luma_mode2_fptr;
  if (mode >= 3 && mode <= 9)
    return selector->ihevc_intra_pred_luma_mode_3_to_9_fptr;
  if (mode == 10)
    return (LumaIntraPredFn)selector->ihevc_intra_pred_luma_horz_fptr;
  if (mode >= 11 && mode <= 17)
    return selector->ihevc_intra_pred_luma_mode_11_to_17_fptr;
  if (mode == 18 || mode == 34)
    return selector->ihevc_intra_pred_luma_mode_18_34_fptr;
  if (mode >= 19 && mode <= 25)
    return selector->ihevc_intra_pred_luma_mode_19_to_25_fptr;
  if (mode == 26)
    return (LumaIntraPredFn)selector->ihevc_intra_pred_luma_ver_fptr;
  if (mode >= 27 && mode <= 33)
    return selector->ihevc_intra_pred_luma_mode_27_to_33_fptr;
  return nullptr;
}

ChromaIntraPredFn GetChromaIntraPredFn(const ihevc_func_selector_t* selector,
                                       int mode) {
  if (!selector) return nullptr;
  if (mode == 0) return selector->ihevc_intra_pred_chroma_planar_fptr;
  if (mode == 1) return selector->ihevc_intra_pred_chroma_dc_fptr;
  if (mode == 2) return selector->ihevc_intra_pred_chroma_mode2_fptr;
  if (mode >= 3 && mode <= 9)
    return selector->ihevc_intra_pred_chroma_mode_3_to_9_fptr;
  if (mode == 10) return selector->ihevc_intra_pred_chroma_horz_fptr;
  if (mode >= 11 && mode <= 17)
    return selector->ihevc_intra_pred_chroma_mode_11_to_17_fptr;
  if (mode == 18 || mode == 34)
    return selector->ihevc_intra_pred_chroma_mode_18_34_fptr;
  if (mode >= 19 && mode <= 25)
    return selector->ihevc_intra_pred_chroma_mode_19_to_25_fptr;
  if (mode == 26) return selector->ihevc_intra_pred_chroma_ver_fptr;
  if (mode >= 27 && mode <= 33)
    return selector->ihevc_intra_pred_chroma_mode_27_to_33_fptr;
  return nullptr;
}

HbdLumaIntraPredFn GetHbdLumaIntraPredFn(IV_ARCH_T arch, int mode) {
  if (arch == ARCH_NA) {
    if (mode == 0) return ihevc_hbd_intra_pred_luma_planar;
    if (mode == 1) return ihevc_hbd_intra_pred_luma_dc;
    if (mode == 2) return ihevc_hbd_intra_pred_luma_mode2;
    if (mode >= 3 && mode <= 9) return ihevc_hbd_intra_pred_luma_mode_3_to_9;
    if (mode == 10) return (HbdLumaIntraPredFn)ihevc_hbd_intra_pred_luma_horz;
    if (mode >= 11 && mode <= 17)
      return ihevc_hbd_intra_pred_luma_mode_11_to_17;
    if (mode == 18 || mode == 34) return ihevc_hbd_intra_pred_luma_mode_18_34;
    if (mode >= 19 && mode <= 25)
      return ihevc_hbd_intra_pred_luma_mode_19_to_25;
    if (mode == 26) return (HbdLumaIntraPredFn)ihevc_hbd_intra_pred_luma_ver;
    if (mode >= 27 && mode <= 33)
      return ihevc_hbd_intra_pred_luma_mode_27_to_33;
  }
  // Future SIMD implementations can be added here
  return nullptr;
}

HbdChromaIntraPredFn GetHbdChromaIntraPredFn(IV_ARCH_T arch, int mode) {
  if (arch == ARCH_NA) {
    if (mode == 0) return ihevc_hbd_intra_pred_chroma_planar;
    if (mode == 1) return ihevc_hbd_intra_pred_chroma_dc;
    if (mode == 2) return ihevc_hbd_intra_pred_chroma_mode2;
    if (mode >= 3 && mode <= 9) return ihevc_hbd_intra_pred_chroma_mode_3_to_9;
    if (mode == 10) return ihevc_hbd_intra_pred_chroma_horz;
    if (mode >= 11 && mode <= 17)
      return ihevc_hbd_intra_pred_chroma_mode_11_to_17;
    if (mode == 18 || mode == 34) return ihevc_hbd_intra_pred_chroma_mode_18_34;
    if (mode >= 19 && mode <= 25)
      return ihevc_hbd_intra_pred_chroma_mode_19_to_25;
    if (mode == 26) return ihevc_hbd_intra_pred_chroma_ver;
    if (mode >= 27 && mode <= 33)
      return ihevc_hbd_intra_pred_chroma_mode_27_to_33;
  }
  // Future SIMD implementations can be added here
  return nullptr;
}

const std::vector<IntraPredModeInfo>& GetIntraPredBenchmarkModes() {
  static const std::vector<IntraPredModeInfo> kModes = {
      {0, "planar"},      {1, "dc"},
      {2, "mode_2"},      {4, "mode_3_to_9"},
      {10, "horz"},       {14, "mode_11_to_17"},
      {18, "mode_18_34"}, {22, "mode_19_to_25"},
      {26, "ver"},        {30, "mode_27_to_33"},
  };
  return kModes;
}

const std::vector<int>& GetIntraPredBenchmarkSizes() {
  static const std::vector<int> kSizes = {4, 8, 16, 32};
  return kSizes;
}
