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
*  ihevcd_hbd_func_tables.c
*
* @brief
*  Contains function pointer tables to initialize function pointers used in hevc
*
* @author
*  Naveen
*
* @par List of Functions:
* @remarks
*  None
*
*******************************************************************************
*/
/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "ihevc_typedefs.h"
#include "iv.h"
#include "ivd.h"
#include "ihevc_defs.h"
#include "ihevc_debug.h"
#include "ihevc_structs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_cabac_tables.h"
#include "ihevc_disp_mgr.h"
#include "ihevc_buf_mgr.h"
#include "ihevc_dpb_mgr.h"
#include "ihevc_error.h"

#include "ihevcd_defs.h"
#include "ihevcd_function_selector.h"
#include "ihevcd_structs.h"

const pf_hbd_itrans_recon gafp_ihevcd_hbd_itrans_recon[] =
{
    (pf_hbd_itrans_recon)&ihevc_hbd_itrans_recon_4x4_ttype1,
    (pf_hbd_itrans_recon)&ihevc_hbd_itrans_recon_4x4,
    (pf_hbd_itrans_recon)&ihevc_hbd_itrans_recon_8x8,
    (pf_hbd_itrans_recon)&ihevc_hbd_itrans_recon_16x16,
    (pf_hbd_itrans_recon)&ihevc_hbd_itrans_recon_32x32,
    (pf_hbd_itrans_recon)&ihevc_hbd_chroma_itrans_recon_4x4,
    (pf_hbd_itrans_recon)&ihevc_hbd_chroma_itrans_recon_8x8,
    (pf_hbd_itrans_recon)&ihevc_hbd_chroma_itrans_recon_16x16
};

const pf_hbd_recon gafp_ihevcd_hbd_recon[] =
{
    (pf_hbd_recon)&ihevc_hbd_recon_4x4_ttype1,
    (pf_hbd_recon)&ihevc_hbd_recon_4x4,
    (pf_hbd_recon)&ihevc_hbd_recon_8x8,
    (pf_hbd_recon)&ihevc_hbd_recon_16x16,
    (pf_hbd_recon)&ihevc_hbd_recon_32x32,
    (pf_hbd_recon)&ihevc_hbd_chroma_recon_4x4,
    (pf_hbd_recon)&ihevc_hbd_chroma_recon_8x8,
    (pf_hbd_recon)&ihevc_hbd_chroma_recon_16x16
};

const pf_hbd_itrans_recon_dc gafp_ihevcd_hbd_itrans_recon_dc[] =
{
    (pf_hbd_itrans_recon_dc)&ihevcd_hbd_itrans_recon_dc_luma,
    (pf_hbd_itrans_recon_dc)&ihevcd_hbd_itrans_recon_dc_chroma
};

const pf_hbd_intra_pred_luma gafp_ihevcd_hbd_intra_pred_luma[] =
{
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_planar,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_planar,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_dc,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_mode2,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_mode_3_to_9,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_horz,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_mode_11_to_17,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_mode_18_34,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_mode_19_to_25,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_ver,
    (pf_hbd_intra_pred_luma)&ihevc_hbd_intra_pred_luma_mode_27_to_33
};

const pf_hbd_intra_pred_chroma gafp_ihevcd_hbd_intra_pred_chroma[] =
{
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_planar,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_planar,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_dc,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_mode2,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_mode_3_to_9,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_horz,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_mode_11_to_17,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_mode_18_34,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_mode_19_to_25,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_ver,
    (pf_hbd_intra_pred_chroma)&ihevc_hbd_intra_pred_chroma_mode_27_to_33
};

const pf_hbd_sao_luma gapf_hbd_sao_luma[] =
{
    (pf_hbd_sao_luma)&ihevc_hbd_sao_edge_offset_class0,
    (pf_hbd_sao_luma)&ihevc_hbd_sao_edge_offset_class1,
    (pf_hbd_sao_luma)&ihevc_hbd_sao_edge_offset_class2,
    (pf_hbd_sao_luma)&ihevc_hbd_sao_edge_offset_class3
};

const pf_hbd_sao_chroma gapf_hbd_sao_chroma[] =
{
    (pf_hbd_sao_chroma)&ihevc_hbd_sao_edge_offset_class0_chroma,
    (pf_hbd_sao_chroma)&ihevc_hbd_sao_edge_offset_class1_chroma,
    (pf_hbd_sao_chroma)&ihevc_hbd_sao_edge_offset_class2_chroma,
    (pf_hbd_sao_chroma)&ihevc_hbd_sao_edge_offset_class3_chroma
};

const pf_hbd_inter_pred gapf_hbd_inter_pred[22] =
{
      NULL,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_copy,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_vert,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_horz,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_horz_w16out,

      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_copy_w16out,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_vert_w16out,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_horz_w16out,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_horz_w16out,

      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_vert_w16inp,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_luma_vert_w16inp_w16out,

      NULL,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_copy,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_vert,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_horz,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_horz_w16out,

      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_copy_w16out,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_vert_w16out,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_horz_w16out,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_horz_w16out,

      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_vert_w16inp,
      (pf_hbd_inter_pred)&ihevc_hbd_inter_pred_chroma_vert_w16inp_w16out
};

