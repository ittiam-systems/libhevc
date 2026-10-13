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

#include "ihevc_itrans_utils.h"

#include <random>
#include <string>
#include <vector>

#include "func_selector.h"

extern "C" {
#include "ihevc_defs.h"
#include "ihevc_resi_trans.h"
#include "ihevc_typedefs.h"
}

namespace {

void GenerateITransCoeffs(int trans_size, int ttype, int nz_cols, int nz_rows,
                          WORD16* pi2_coeffs, WORD32* zero_cols,
                          WORD32* zero_rows) {
  std::vector<UWORD8> pu1_src(trans_size * trans_size);
  std::vector<UWORD8> pu1_pred_fwd(trans_size * trans_size);
  std::vector<WORD32> pi4_temp(trans_size * trans_size);

  FillRandom(pu1_src, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 1337);
  FillRandom(pu1_pred_fwd, static_cast<UWORD8>(0), static_cast<UWORD8>(255),
             1338);

  // Forward residue transform: resi = pu1_src - pu1_pred_fwd in [-255, 255],
  // transformed into 16-bit coefficients.
  ihevc_resi_trans_4x4_ft* resi_trans_fn = nullptr;
  if (trans_size == 4) {
    resi_trans_fn =
        (ttype == 1) ? ihevc_resi_trans_4x4_ttype1 : ihevc_resi_trans_4x4;
  } else if (trans_size == 8) {
    resi_trans_fn = ihevc_resi_trans_8x8;
  } else if (trans_size == 16) {
    resi_trans_fn = ihevc_resi_trans_16x16;
  } else if (trans_size == 32) {
    resi_trans_fn = ihevc_resi_trans_32x32;
  }

  if (resi_trans_fn) {
    resi_trans_fn(pu1_src.data(), pu1_pred_fwd.data(), pi4_temp.data(),
                  pi2_coeffs, trans_size, trans_size, trans_size, NULL_PLANE);
  }

  // Zero out coefficients outside the top-left nz_cols x nz_rows region
  for (int r = 0; r < trans_size; ++r) {
    for (int c = 0; c < trans_size; ++c) {
      if (c >= nz_cols || r >= nz_rows) {
        pi2_coeffs[r * trans_size + c] = 0;
      }
    }
  }

  // Compute row/col zero bitmasks
  *zero_cols = ComputeZeroMask(trans_size, nz_cols);
  *zero_rows = ComputeZeroMask(trans_size, nz_rows);
}

}  // namespace

ITransFn GetITransFn(const ihevc_func_selector_t* selector, int trans_size,
                     int ttype) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return (ttype == 1) ? selector->ihevc_itrans_4x4_ttype1_fptr
                        : selector->ihevc_itrans_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_itrans_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_itrans_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_itrans_32x32_fptr;
  }
  return nullptr;
}

ITransResFn GetITransResFn(const ihevc_func_selector_t* selector,
                           int trans_size, int ttype) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return (ttype == 1) ? selector->ihevc_itrans_res_4x4_ttype1_fptr
                        : selector->ihevc_itrans_res_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_itrans_res_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_itrans_res_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_itrans_res_32x32_fptr;
  }
  return nullptr;
}

ITransReconFn GetITransReconFn(const ihevc_func_selector_t* selector,
                               int trans_size, int ttype) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return (ttype == 1) ? selector->ihevc_itrans_recon_4x4_ttype1_fptr
                        : selector->ihevc_itrans_recon_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_itrans_recon_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_itrans_recon_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_itrans_recon_32x32_fptr;
  }
  return nullptr;
}

ITransReconFn GetITransReconFn(IV_ARCH_T arch, int trans_size, int ttype) {
  return GetITransReconFn(get_func_ptr(arch), trans_size, ttype);
}

ChromaITransReconFn GetChromaITransReconFn(
    const ihevc_func_selector_t* selector, int trans_size) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return selector->ihevc_chroma_itrans_recon_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_chroma_itrans_recon_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_chroma_itrans_recon_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_chroma_itrans_recon_32x32_fptr;
  }
  return nullptr;
}

ChromaITransReconFn GetChromaITransReconFn(IV_ARCH_T arch, int trans_size) {
  return GetChromaITransReconFn(get_func_ptr(arch), trans_size);
}

HbdITransReconFn GetHbdITransReconFn(const ihevc_func_selector_t* selector, int trans_size,
                                     int ttype) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return (ttype == 1) ? selector->ihevc_hbd_itrans_recon_4x4_ttype1_fptr
                        : selector->ihevc_hbd_itrans_recon_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_hbd_itrans_recon_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_hbd_itrans_recon_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_hbd_itrans_recon_32x32_fptr;
  }
  return nullptr;
}

HbdITransReconFn GetHbdITransReconFn(IV_ARCH_T arch, int trans_size, int ttype) {
  return GetHbdITransReconFn(get_func_ptr(arch), trans_size, ttype);
}

HbdChromaITransReconFn GetHbdChromaITransReconFn(const ihevc_func_selector_t* selector,
                                                 int trans_size) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return selector->ihevc_hbd_chroma_itrans_recon_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_hbd_chroma_itrans_recon_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_hbd_chroma_itrans_recon_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_hbd_chroma_itrans_recon_32x32_fptr;
  }
  return nullptr;
}

HbdChromaITransReconFn GetHbdChromaITransReconFn(IV_ARCH_T arch, int trans_size) {
  return GetHbdChromaITransReconFn(get_func_ptr(arch), trans_size);
}

ReconFn GetReconFn(const ihevc_func_selector_t* selector, int trans_size,
                   int ttype) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return (ttype == 1) ? selector->ihevc_recon_4x4_ttype1_fptr
                        : selector->ihevc_recon_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_recon_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_recon_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_recon_32x32_fptr;
  }
  return nullptr;
}

ChromaReconFn GetChromaReconFn(const ihevc_func_selector_t* selector,
                               int trans_size) {
  if (!selector) return nullptr;
  if (trans_size == 4) {
    return selector->ihevc_chroma_recon_4x4_fptr;
  } else if (trans_size == 8) {
    return selector->ihevc_chroma_recon_8x8_fptr;
  } else if (trans_size == 16) {
    return selector->ihevc_chroma_recon_16x16_fptr;
  } else if (trans_size == 32) {
    return selector->ihevc_chroma_recon_32x32_fptr;
  }
  return nullptr;
}

void GenerateITransInput(int trans_size, int ttype, int nz_cols, int nz_rows,
                         WORD16* pi2_coeffs, UWORD8* pu1_pred_recon,
                         WORD32* zero_cols, WORD32* zero_rows) {
  GenerateITransCoeffs(trans_size, ttype, nz_cols, nz_rows, pi2_coeffs,
                       zero_cols, zero_rows);

  if (pu1_pred_recon) {
    FillRandom(pu1_pred_recon, static_cast<size_t>(trans_size * trans_size),
               static_cast<UWORD8>(0), static_cast<UWORD8>(255), 1337);
  }
}

void GenerateITransInput(int trans_size, int ttype, int nz_cols, int nz_rows,
                         WORD16* pi2_coeffs, UWORD16* pu2_pred_recon,
                         WORD32* zero_cols, WORD32* zero_rows, int bit_depth) {
  GenerateITransCoeffs(trans_size, ttype, nz_cols, nz_rows, pi2_coeffs,
                       zero_cols, zero_rows);

  if (pu2_pred_recon) {
    FillRandom(pu2_pred_recon, static_cast<size_t>(trans_size * trans_size),
               static_cast<UWORD16>(0),
               static_cast<UWORD16>((1 << bit_depth) - 1), 1337);
  }
}

const std::vector<ITransTestCase>& GetITransTestCases() {
  static const std::vector<ITransTestCase> kTestCases = {
      {4, 0, {{4, 4}, {4, 2}, {2, 4}}},
      {4, 1, {{4, 4}, {4, 2}, {2, 4}}},
      {8, 0, {{4, 4}, {8, 4}, {4, 8}, {8, 8}}},
      {16,
       0,
       {{4, 4},
        {8, 4},
        {4, 8},
        {8, 8},
        {16, 4},
        {4, 16},
        {16, 8},
        {8, 16},
        {16, 16}}},
      {32,
       0,
       {{4, 4},
        {8, 4},
        {4, 8},
        {8, 8},
        {16, 4},
        {4, 16},
        {16, 8},
        {8, 16},
        {16, 16},
        {32, 8},
        {8, 32},
        {32, 16},
        {16, 32},
        {32, 32}}},
  };
  return kTestCases;
}

template <typename T>
ITransReconBuffers<T> CreateITransReconBuffers(const ITransReconConfig& cfg) {
  const int trans_size = cfg.trans_size;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  // x86 32x32 uses 3 * 32 * 32 of scratch, and stage 2 reads 4 past the end
  const int x86_tmp_overread = 4;
  int tmp_size = (trans_size == 32) ? 3 * trans_size * trans_size + x86_tmp_overread
                                    : trans_size * trans_size;
#else
  int tmp_size = trans_size * trans_size;
#endif
  // Row padding catches wrong row stepping and writes past the row
  const int stride_pad = 16;
  const T max_val = static_cast<T>((1 << cfg.bit_depth) - 1);
  // Small coefficients keep most outputs unclipped; full range exercises saturation
  const int coeff_min = cfg.full_range ? -32768 : -512;
  const int coeff_max = cfg.full_range ? 32767 : 512;

  ITransReconBuffers<T> b;
  b.src_strd = trans_size;
  b.pred_strd = (cfg.is_chroma ? 2 : 1) * trans_size + stride_pad;
  b.dst_strd = b.pred_strd;
  b.offset = cfg.plane;

  // Guard row below the block catches writes and covers SIMD over-reads
  b.src.resize(trans_size * trans_size);
  b.tmp.resize(tmp_size);
  b.dst.resize(b.dst_strd * (trans_size + 1));

  std::mt19937 rng(0);
  std::uniform_int_distribution<int> coeff(coeff_min, coeff_max);
  std::uniform_int_distribution<int> keep(0, 3);
  for (int r = 0; r < cfg.nz_rows; ++r) {
    for (int c = 0; c < cfg.nz_cols; ++c) {
      // Sparse: keep 1 in 4, always the corner so the region size stays the same
      bool corner = (r == cfg.nz_rows - 1) && (c == cfg.nz_cols - 1);
      if (cfg.sparse && !corner && keep(rng) != 0) continue;
      WORD16 val = static_cast<WORD16>(coeff(rng));
      // A zero corner would shrink the region, so redraw until it is non-zero
      while (corner && val == 0) val = static_cast<WORD16>(coeff(rng));
      b.src[r * b.src_strd + c] = val;
    }
  }

  // Bits above trans_size stay set
  b.zero_cols = static_cast<WORD32>(0xFFFFFFFFu);
  b.zero_rows = static_cast<WORD32>(0xFFFFFFFFu);
  for (int r = 0; r < trans_size; ++r) {
    for (int c = 0; c < trans_size; ++c) {
      if (b.src[r * b.src_strd + c] != 0) {
        b.zero_cols &= ~static_cast<WORD32>(1u << c);
        b.zero_rows &= ~static_cast<WORD32>(1u << r);
      }
    }
  }

  // Random pred (in dst) and stale tmp, so stray writes and reads are caught
  FillRandom(b.dst, static_cast<T>(0), max_val, 42);
  FillRandom(b.tmp, static_cast<WORD16>(-32768), static_cast<WORD16>(32767), 1337);
  return b;
}

// Explicit instantiations for 8-bit and HBD
template ITransReconBuffers<UWORD8> CreateITransReconBuffers<UWORD8>(const ITransReconConfig& cfg);
template ITransReconBuffers<UWORD16> CreateITransReconBuffers<UWORD16>(
    const ITransReconConfig& cfg);
