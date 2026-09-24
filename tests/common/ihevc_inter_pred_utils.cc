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

#include "ihevc_inter_pred_utils.h"

const std::vector<InterPredOpInfo>& GetInterPredOps() {
  static const std::vector<InterPredOpInfo> kOps = {
      {InterPredOp::kCopy, "copy", false, false},
      {InterPredOp::kHorz, "horz", false, false},
      {InterPredOp::kVert, "vert", false, false},
      {InterPredOp::kCopyW16out, "copy_w16out", false, true},
      {InterPredOp::kHorzW16out, "horz_w16out", false, true},
      {InterPredOp::kVertW16out, "vert_w16out", false, true},
      {InterPredOp::kVertW16inp, "vert_w16inp", true, false},
      {InterPredOp::kVertW16inpW16out, "vert_w16inp_w16out", true, true},
  };
  return kOps;
}

ihevc_inter_pred_ft* GetLumaInterPredFn(const ihevc_func_selector_t* selector,
                                        InterPredOp op) {
  if (!selector) return nullptr;
  switch (op) {
    case InterPredOp::kCopy:
      return selector->ihevc_inter_pred_luma_copy_fptr;
    case InterPredOp::kHorz:
      return selector->ihevc_inter_pred_luma_horz_fptr;
    case InterPredOp::kVert:
      return selector->ihevc_inter_pred_luma_vert_fptr;
    default:
      return nullptr;
  }
}

ihevc_inter_pred_w16out_ft* GetLumaInterPredW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op) {
  if (!selector) return nullptr;
  switch (op) {
    case InterPredOp::kCopyW16out:
      return selector->ihevc_inter_pred_luma_copy_w16out_fptr;
    case InterPredOp::kHorzW16out:
      return selector->ihevc_inter_pred_luma_horz_w16out_fptr;
    case InterPredOp::kVertW16out:
      return selector->ihevc_inter_pred_luma_vert_w16out_fptr;
    default:
      return nullptr;
  }
}

ihevc_inter_pred_w16inp_ft* GetLumaInterPredW16inpFn(
    const ihevc_func_selector_t* selector, InterPredOp op) {
  if (!selector) return nullptr;
  if (op == InterPredOp::kVertW16inp) {
    return selector->ihevc_inter_pred_luma_vert_w16inp_fptr;
  }
  return nullptr;
}

ihevc_inter_pred_w16inp_w16out_ft* GetLumaInterPredW16inpW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op) {
  if (!selector) return nullptr;
  if (op == InterPredOp::kVertW16inpW16out) {
    return selector->ihevc_inter_pred_luma_vert_w16inp_w16out_fptr;
  }
  return nullptr;
}

ihevc_inter_pred_ft* GetChromaInterPredFn(const ihevc_func_selector_t* selector,
                                          InterPredOp op) {
  if (!selector) return nullptr;
  switch (op) {
    case InterPredOp::kCopy:
      return selector->ihevc_inter_pred_chroma_copy_fptr;
    case InterPredOp::kHorz:
      return selector->ihevc_inter_pred_chroma_horz_fptr;
    case InterPredOp::kVert:
      return selector->ihevc_inter_pred_chroma_vert_fptr;
    default:
      return nullptr;
  }
}

ihevc_inter_pred_w16out_ft* GetChromaInterPredW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op) {
  if (!selector) return nullptr;
  switch (op) {
    case InterPredOp::kCopyW16out:
      return selector->ihevc_inter_pred_chroma_copy_w16out_fptr;
    case InterPredOp::kHorzW16out:
      return selector->ihevc_inter_pred_chroma_horz_w16out_fptr;
    case InterPredOp::kVertW16out:
      return selector->ihevc_inter_pred_chroma_vert_w16out_fptr;
    default:
      return nullptr;
  }
}

ihevc_inter_pred_w16inp_ft* GetChromaInterPredW16inpFn(
    const ihevc_func_selector_t* selector, InterPredOp op) {
  if (!selector) return nullptr;
  if (op == InterPredOp::kVertW16inp) {
    return selector->ihevc_inter_pred_chroma_vert_w16inp_fptr;
  }
  return nullptr;
}

ihevc_inter_pred_w16inp_w16out_ft* GetChromaInterPredW16inpW16outFn(
    const ihevc_func_selector_t* selector, InterPredOp op) {
  if (!selector) return nullptr;
  if (op == InterPredOp::kVertW16inpW16out) {
    return selector->ihevc_inter_pred_chroma_vert_w16inp_w16out_fptr;
  }
  return nullptr;
}

ihevc_hbd_inter_pred_ft* GetHbdLumaInterPredFn(IV_ARCH_T arch, InterPredOp op) {
  if (arch == ARCH_NA) {
    switch (op) {
      case InterPredOp::kCopy:
        return ihevc_hbd_inter_pred_luma_copy;
      case InterPredOp::kHorz:
        return ihevc_hbd_inter_pred_luma_horz;
      case InterPredOp::kVert:
        return ihevc_hbd_inter_pred_luma_vert;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

ihevc_hbd_inter_pred_w16out_ft* GetHbdLumaInterPredW16outFn(IV_ARCH_T arch,
                                                            InterPredOp op) {
  if (arch == ARCH_NA) {
    switch (op) {
      case InterPredOp::kCopyW16out:
        return ihevc_hbd_inter_pred_luma_copy_w16out;
      case InterPredOp::kHorzW16out:
        return ihevc_hbd_inter_pred_luma_horz_w16out;
      case InterPredOp::kVertW16out:
        return ihevc_hbd_inter_pred_luma_vert_w16out;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

ihevc_hbd_inter_pred_w16inp_ft* GetHbdLumaInterPredW16inpFn(IV_ARCH_T arch,
                                                            InterPredOp op) {
  if (arch == ARCH_NA) {
    if (op == InterPredOp::kVertW16inp) {
      return ihevc_hbd_inter_pred_luma_vert_w16inp;
    }
  }
  return nullptr;
}

ihevc_hbd_inter_pred_w16inp_w16out_ft* GetHbdLumaInterPredW16inpW16outFn(
    IV_ARCH_T arch, InterPredOp op) {
  if (arch == ARCH_NA) {
    if (op == InterPredOp::kVertW16inpW16out) {
      return ihevc_hbd_inter_pred_luma_vert_w16inp_w16out;
    }
  }
  return nullptr;
}

ihevc_hbd_inter_pred_ft* GetHbdChromaInterPredFn(IV_ARCH_T arch,
                                                 InterPredOp op) {
  if (arch == ARCH_NA) {
    switch (op) {
      case InterPredOp::kCopy:
        return ihevc_hbd_inter_pred_chroma_copy;
      case InterPredOp::kHorz:
        return ihevc_hbd_inter_pred_chroma_horz;
      case InterPredOp::kVert:
        return ihevc_hbd_inter_pred_chroma_vert;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

ihevc_hbd_inter_pred_w16out_ft* GetHbdChromaInterPredW16outFn(IV_ARCH_T arch,
                                                              InterPredOp op) {
  if (arch == ARCH_NA) {
    switch (op) {
      case InterPredOp::kCopyW16out:
        return ihevc_hbd_inter_pred_chroma_copy_w16out;
      case InterPredOp::kHorzW16out:
        return ihevc_hbd_inter_pred_chroma_horz_w16out;
      case InterPredOp::kVertW16out:
        return ihevc_hbd_inter_pred_chroma_vert_w16out;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

ihevc_hbd_inter_pred_w16inp_ft* GetHbdChromaInterPredW16inpFn(IV_ARCH_T arch,
                                                              InterPredOp op) {
  if (arch == ARCH_NA) {
    if (op == InterPredOp::kVertW16inp) {
      return ihevc_hbd_inter_pred_chroma_vert_w16inp;
    }
  }
  return nullptr;
}

ihevc_hbd_inter_pred_w16inp_w16out_ft* GetHbdChromaInterPredW16inpW16outFn(
    IV_ARCH_T arch, InterPredOp op) {
  if (arch == ARCH_NA) {
    if (op == InterPredOp::kVertW16inpW16out) {
      return ihevc_hbd_inter_pred_chroma_vert_w16inp_w16out;
    }
  }
  return nullptr;
}

const std::vector<std::pair<int, int>>& GetInterPredLumaBenchmarkSizes() {
  static const std::vector<std::pair<int, int>> kSizes = {
      {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}, {64, 64},
  };
  return kSizes;
}

const std::vector<std::pair<int, int>>& GetInterPredChromaBenchmarkSizes() {
  static const std::vector<std::pair<int, int>> kSizes = {
      {4, 4}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {32, 32},
  };
  return kSizes;
}
