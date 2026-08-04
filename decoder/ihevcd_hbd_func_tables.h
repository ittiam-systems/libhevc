/******************************************************************************
*
* Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore
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
/**
*******************************************************************************
* @file
*  ihevcd_hbd_func_tables.h
*
* @brief
*  HBD function ptr tables for C function,  or Neon intrinsics
* or Cortex A8 intrinsics or Neon  assembly or cortex a8 assembly or intel
* x86  instrinsics
*
* @remarks
*  None
*
*******************************************************************************
*/

#ifndef __ihevcd_hbd_func_tables_H__
#define __ihevcd_hbd_func_tables_H__

#include "ihevcd_defs.h"
#include "ihevcd_structs.h"

/* Extern variable declarations */
extern const pf_hbd_itrans_recon gafp_ihevcd_hbd_itrans_recon[8];

extern const pf_hbd_itrans_recon gafp_ihevcd_hbd_itrans_recon_av8[8];

extern const pf_hbd_recon gafp_ihevcd_hbd_recon[8];

extern const pf_hbd_itrans_recon_dc gafp_ihevcd_hbd_itrans_recon_dc[2];

extern const pf_hbd_intra_pred_luma gafp_ihevcd_hbd_intra_pred_luma[11];

extern const pf_hbd_intra_pred_chroma gafp_ihevcd_hbd_intra_pred_chroma[11];

extern const pf_hbd_intra_pred_chroma gafp_ihevcd_hbd_intra_pred_chroma_av8[11];

extern const pf_hbd_sao_luma gapf_hbd_sao_luma[4];

extern const pf_hbd_sao_chroma gapf_hbd_sao_chroma[4];

extern const pf_hbd_inter_pred gapf_hbd_inter_pred[22];

extern const pf_hbd_itrans_recon gafp_ihevcd_hbd_itrans_recon_sse42[8];

extern const pf_hbd_intra_pred_luma gafp_ihevcd_hbd_intra_pred_luma_sse42[11];

extern const pf_hbd_intra_pred_chroma gafp_ihevcd_hbd_intra_pred_chroma_sse42[11];

extern const pf_hbd_sao_luma gapf_hbd_sao_luma_sse42[4];

extern const pf_hbd_sao_chroma gapf_hbd_sao_chroma_sse42[4];

extern const pf_hbd_inter_pred gapf_hbd_inter_pred_sse42[22];

extern const pf_hbd_sao_luma gapf_hbd_sao_luma_av8[4];

extern const pf_hbd_sao_chroma gapf_hbd_sao_chroma_av8[4];

extern const pf_hbd_inter_pred gapf_hbd_inter_pred_av8[22];

extern const pf_hbd_intra_pred_luma gafp_ihevcd_hbd_intra_pred_luma_av8[11];

extern const pf_hbd_itrans_recon gafp_ihevcd_hbd_itrans_recon_a9q[8];

extern const pf_hbd_sao_luma gapf_hbd_sao_luma_a9q[4];

extern const pf_hbd_inter_pred gapf_hbd_inter_pred_a9q[22];

extern const pf_hbd_intra_pred_luma gafp_ihevcd_hbd_intra_pred_luma_a9q[11];

extern const pf_hbd_intra_pred_chroma gafp_ihevcd_hbd_intra_pred_chroma_a9q[11];

extern const pf_hbd_intra_pred_luma gafp_ihevcd_hbd_intra_pred_luma_neonintr[11];

extern const pf_hbd_intra_pred_chroma gafp_ihevcd_hbd_intra_pred_chroma_neonintr[11];

extern const pf_hbd_sao_luma gapf_hbd_sao_luma_neonintr[4];

extern const pf_hbd_inter_pred gapf_hbd_inter_pred_neonintr[22];

#endif  /* __ihevcd_hbd_func_tables_H__ */

