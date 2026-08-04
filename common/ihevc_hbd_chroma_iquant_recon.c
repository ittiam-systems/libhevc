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
 *  ihevc_chroma_iquant_recon.c
 *
 * @brief
 *  Contains function definitions for inverse  quantization and
 * reconstruction  of chroma interleaved data.
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *   - ihevc_chroma_iquant_recon_4x4()
 *   - ihevc_chroma_iquant_recon_8x8()
 *   - ihevc_chroma_iquant_recon_16x16()
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

#include <stdio.h>
#include <string.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_chroma_iquant_recon.h"
#include "ihevc_function_selector.h"
#include "ihevc_trans_macros.h"

/* All the functions work one component(U or V) of interleaved data depending upon pointers passed to it */
/* Data visualization */
/* U V U V U V U V */
/* U V U V U V U V */
/* U V U V U V U V */
/* U V U V U V U V */
/* If the pointer points to first byte of above stream (U) , functions will operate on U component */
/* If the pointer points to second byte of above stream (V) , functions will operate on V component */

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs inverse quantization and  reconstruction for 4x4
 * input block
 *
 * @par Description:
 *  This function performs inverse quantization and  reconstruction for 4x4
 * input block
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pu2_pred
 *  Prediction 4x4 block
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pu2_dst
 *  Output 4x4 block
 *
 * @param[in] i4_qp_div
 *  Quantization parameter / 6
 *
 * @param[in] i4_qp_rem
 *  Quantization parameter % 6
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_zero_cols
 *  Zero columns in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

void ihevc_hbd_chroma_iquant_recon_4x4(WORD16 *pi2_src,
                                   UWORD16 *pu2_pred,
                                   WORD16 *pi2_dequant_coeff,
                                   UWORD16 *pu2_dst,
                                   WORD32 i4_qp_div,/* qpscaled / 6 */
                                   WORD32 i4_qp_rem,/* qpscaled % 6 */
                                   WORD32 i4_src_strd,
                                   WORD32 i4_pred_strd,
                                   WORD32 i4_dst_strd,
                                   WORD32 i4_zero_cols,
                                   UWORD8 u1_bit_depth)
{
    WORD16 clip_limit;
    /* Inverse Quant and recon */
    WORD32 i, j;
    WORD32 shift_iq;
    WORD32 trans_size;
    WORD32 shift_skip_trans,add_skip_trans;
    /* Inverse Quantization constants */
    {
        WORD32 log2_trans_size;

        log2_trans_size = 2;
        shift_iq = u1_bit_depth + log2_trans_size - 5;
    }

    trans_size = TRANS_SIZE_4;

    for(i = 0; i < trans_size; i++)
    {
        /* Checking for Zero Cols */
        if((i4_zero_cols & 1) == 1)
        {
            for(j = 0; j < trans_size; j++)
                pu2_dst[j * i4_dst_strd] = pu2_pred[j * i4_pred_strd];
        }
        else
        {
            clip_limit = (1 << u1_bit_depth) -1;
            shift_skip_trans = 13 - u1_bit_depth;
            add_skip_trans = (1 << (shift_skip_trans -1));
            for(j = 0; j < trans_size; j++)
            {
                WORD32 iquant_out;

                IQUANT_4x4(iquant_out,
                    pi2_src[j*i4_src_strd],
                    pi2_dequant_coeff[j*trans_size]* g_ihevc_iquant_scales[i4_qp_rem],
                    shift_iq, i4_qp_div);
                iquant_out = (iquant_out + add_skip_trans) >> shift_skip_trans;
                pu2_dst[j * i4_dst_strd] =
                    CLIP3(( iquant_out + pu2_pred[j*i4_pred_strd]), 0, clip_limit);
            }
        }
        pi2_src++;
        pi2_dequant_coeff++;
        pu2_pred += 2;
        pu2_dst += 2;

        i4_zero_cols = i4_zero_cols >> 1;
    }
}
/**
 *******************************************************************************
 *
 * @brief
 *  This function performs inverse quantization and  reconstruction for 8x8
 * input block
 *
 * @par Description:
 *  This function performs inverse quantization and  reconstruction for 8x8
 * input block
 *
 * @param[in] pi2_src
 *  Input 8x8 coefficients
 *
 * @param[in] pu2_pred
 *  Prediction 8x8 block
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pu2_dst
 *  Output 8x8 block
 *
 * @param[in] i4_qp_div
 *  Quantization parameter / 6
 *
 * @param[in] i4_qp_rem
 *  Quantization parameter % 6
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_zero_cols
 *  Zero columns in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

void ihevc_hbd_chroma_iquant_recon_8x8(WORD16 *pi2_src,
                                   UWORD16 *pu2_pred,
                                   WORD16 *pi2_dequant_coeff,
                                   UWORD16 *pu2_dst,
                                   WORD32 i4_qp_div,/* qpscaled / 6 */
                                   WORD32 i4_qp_rem,/* qpscaled % 6 */
                                   WORD32 i4_src_strd,
                                   WORD32 i4_pred_strd,
                                   WORD32 i4_dst_strd,
                                   WORD32 i4_zero_cols,
                                   UWORD8 u1_bit_depth)
{

    WORD16 clip_limit;
    /* Inverse Quant and recon */
    WORD32 i, j;
    WORD32 shift_iq;
    WORD32 shift_skip_trans,add_skip_trans;
    WORD32 trans_size;
    /* Inverse Quantization constants */
    {
        WORD32 log2_trans_size;

        log2_trans_size = 3;
        shift_iq = u1_bit_depth + log2_trans_size - 5;
    }

    trans_size = TRANS_SIZE_8;

    for(i = 0; i < trans_size; i++)
    {
        /* Checking for Zero Cols */
        if((i4_zero_cols & 1) == 1)
        {
            for(j = 0; j < trans_size; j++)
                pu2_dst[j * i4_dst_strd] = pu2_pred[j * i4_pred_strd];
        }
        else
        {
            clip_limit = (1 << u1_bit_depth) -1;
            shift_skip_trans = 13 - u1_bit_depth;
            add_skip_trans = (1 << (shift_skip_trans -1));
            for(j = 0; j < trans_size; j++)
            {
                WORD32 iquant_out;
                IQUANT(iquant_out,
                    pi2_src[j*i4_src_strd],
                    pi2_dequant_coeff[j*trans_size]* g_ihevc_iquant_scales[i4_qp_rem],
                    shift_iq, i4_qp_div);
                iquant_out = (iquant_out + add_skip_trans) >> shift_skip_trans;
                pu2_dst[j * i4_dst_strd] =
                    CLIP3( (iquant_out + pu2_pred[j*i4_pred_strd] ), 0, clip_limit);
            }
        }
        pi2_src++;
        pi2_dequant_coeff++;
        pu2_pred += 2;
        pu2_dst += 2;

        i4_zero_cols = i4_zero_cols >> 1;
    }
}
/**
 *******************************************************************************
 *
 * @brief
 *  This function performs inverse quantization and  reconstruction for 16x16
 * input block
 *
 * @par Description:
 *  This function performs inverse quantization and  reconstruction for 16x16
 * input block
 *
 * @param[in] pi2_src
 *  Input 16x16 coefficients
 *
 * @param[in] pu2_pred
 *  Prediction 16x16 block
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pu2_dst
 *  Output 16x16 block
 *
 * @param[in] i4_qp_div
 *  Quantization parameter / 6
 *
 * @param[in] i4_qp_rem
 *  Quantization parameter % 6
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_zero_cols
 *  Zero columns in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

void ihevc_hbd_chroma_iquant_recon_16x16(WORD16 *pi2_src,
                                     UWORD16 *pu2_pred,
                                     WORD16 *pi2_dequant_coeff,
                                     UWORD16 *pu2_dst,
                                     WORD32 i4_qp_div,/* qpscaled / 6 */
                                     WORD32 i4_qp_rem,/* qpscaled % 6 */
                                     WORD32 i4_src_strd,
                                     WORD32 i4_pred_strd,
                                     WORD32 i4_dst_strd,
                                     WORD32 i4_zero_cols,
                                     UWORD8 u1_bit_depth)
{
    WORD16 clip_limit;
    /* Inverse Quant and recon */
    WORD32 i, j;
    WORD32 shift_iq;
    WORD32 shift_skip_trans,add_skip_trans;
    WORD32 trans_size;
    /* Inverse Quantization constants */
    {
        WORD32 log2_trans_size;

        log2_trans_size = 4;
        shift_iq = u1_bit_depth + log2_trans_size - 5;
    }

    trans_size = TRANS_SIZE_16;

    for(i = 0; i < trans_size; i++)
    {
        /* Checking for Zero Cols */
        if((i4_zero_cols & 1) == 1)
        {
            for(j = 0; j < trans_size; j++)
                pu2_dst[j * i4_dst_strd] = pu2_pred[j * i4_pred_strd];
        }
        else
        {
            clip_limit = (1 << u1_bit_depth) -1;
            shift_skip_trans = 13 - u1_bit_depth;
            add_skip_trans = (1 << (shift_skip_trans -1));
            for(j = 0; j < trans_size; j++)
            {
                WORD32 iquant_out;
                IQUANT(iquant_out,
                    pi2_src[j*i4_src_strd],
                    pi2_dequant_coeff[j*trans_size] * g_ihevc_iquant_scales[i4_qp_rem],
                    shift_iq, i4_qp_div);
                iquant_out = (iquant_out + add_skip_trans) >> shift_skip_trans;
                pu2_dst[j * i4_dst_strd] =
                    CLIP3( (iquant_out + pu2_pred[j*i4_pred_strd]), 0, clip_limit);
            }
        }
        pi2_src++;
        pi2_dequant_coeff++;
        pu2_pred += 2;
        pu2_dst += 2;

        i4_zero_cols = i4_zero_cols >> 1;
    }
}

