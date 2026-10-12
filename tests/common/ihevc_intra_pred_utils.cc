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
#include "func_selector.h"

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

LumaIntraPredRefSubstFn GetLumaIntraPredRefSubstFn(const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_intra_pred_luma_ref_substitution_fptr;
}

ChromaIntraPredRefSubstFn GetChromaIntraPredRefSubstFn(const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_intra_pred_chroma_ref_substitution_fptr;
}

IntraPredRefFilterFn GetIntraPredRefFilterFn(const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_intra_pred_ref_filtering_fptr;
}

ChromaIntraPredRefFilterFn GetChromaIntraPredRefFilterFn(const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_intra_pred_chroma_ref_filtering_fptr;
}

HbdLumaIntraPredFn GetHbdLumaIntraPredFn(const ihevc_func_selector_t* selector, int mode) {
  if (!selector) return nullptr;
  if (mode == 0) return selector->ihevc_hbd_intra_pred_luma_planar_fptr;
  if (mode == 1) return selector->ihevc_hbd_intra_pred_luma_dc_fptr;
  if (mode == 2) return selector->ihevc_hbd_intra_pred_luma_mode2_fptr;
  if (mode >= 3 && mode <= 9) return selector->ihevc_hbd_intra_pred_luma_mode_3_to_9_fptr;
  if (mode == 10) return (HbdLumaIntraPredFn)selector->ihevc_hbd_intra_pred_luma_horz_fptr;
  if (mode >= 11 && mode <= 17) return selector->ihevc_hbd_intra_pred_luma_mode_11_to_17_fptr;
  if (mode == 18 || mode == 34) return selector->ihevc_hbd_intra_pred_luma_mode_18_34_fptr;
  if (mode >= 19 && mode <= 25) return selector->ihevc_hbd_intra_pred_luma_mode_19_to_25_fptr;
  if (mode == 26) return (HbdLumaIntraPredFn)selector->ihevc_hbd_intra_pred_luma_ver_fptr;
  if (mode >= 27 && mode <= 33) return selector->ihevc_hbd_intra_pred_luma_mode_27_to_33_fptr;
  return nullptr;
}

HbdChromaIntraPredFn GetHbdChromaIntraPredFn(const ihevc_func_selector_t* selector, int mode) {
  if (!selector) return nullptr;
  if (mode == 0) return selector->ihevc_hbd_intra_pred_chroma_planar_fptr;
  if (mode == 1) return selector->ihevc_hbd_intra_pred_chroma_dc_fptr;
  if (mode == 2) return selector->ihevc_hbd_intra_pred_chroma_mode2_fptr;
  if (mode >= 3 && mode <= 9) return selector->ihevc_hbd_intra_pred_chroma_mode_3_to_9_fptr;
  if (mode == 10) return selector->ihevc_hbd_intra_pred_chroma_horz_fptr;
  if (mode >= 11 && mode <= 17) return selector->ihevc_hbd_intra_pred_chroma_mode_11_to_17_fptr;
  if (mode == 18 || mode == 34) return selector->ihevc_hbd_intra_pred_chroma_mode_18_34_fptr;
  if (mode >= 19 && mode <= 25) return selector->ihevc_hbd_intra_pred_chroma_mode_19_to_25_fptr;
  if (mode == 26) return selector->ihevc_hbd_intra_pred_chroma_ver_fptr;
  if (mode >= 27 && mode <= 33) return selector->ihevc_hbd_intra_pred_chroma_mode_27_to_33_fptr;
  return nullptr;
}

HbdLumaIntraPredRefSubstFn GetHbdLumaIntraPredRefSubstFn(const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_hbd_intra_pred_luma_ref_substitution_fptr;
}

HbdChromaIntraPredRefSubstFn GetHbdChromaIntraPredRefSubstFn(
    const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_hbd_intra_pred_chroma_ref_substitution_fptr;
}

HbdIntraPredRefFilterFn GetHbdIntraPredRefFilterFn(const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_hbd_intra_pred_ref_filtering_fptr;
}

HbdChromaIntraPredRefFilterFn GetHbdChromaIntraPredRefFilterFn(
    const ihevc_func_selector_t* selector) {
  if (!selector) return nullptr;
  return selector->ihevc_hbd_intra_pred_chroma_ref_filtering_fptr;
}

HbdLumaIntraPredFn GetHbdLumaIntraPredFn(IV_ARCH_T arch, int mode) {
  return GetHbdLumaIntraPredFn(get_func_ptr(arch), mode);
}

HbdChromaIntraPredFn GetHbdChromaIntraPredFn(IV_ARCH_T arch, int mode) {
  return GetHbdChromaIntraPredFn(get_func_ptr(arch), mode);
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
