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
/**
*******************************************************************************
* @file
*  ihevcd_hbd_itrans_recon_dc_neon_intr.c
*
* @brief
*  Contains ARM NEON intrinsics functions for High Bit Depth DC inverse transform
*  and reconstruction (Luma and Chroma)
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevcd_hbd_itrans_recon_dc_luma_neonintr()
*  - ihevcd_hbd_itrans_recon_dc_chroma_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/

#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_macros.h"
#include "ihevcd_itrans_recon_dc.h"
#include <arm_neon.h>

void ihevcd_hbd_itrans_recon_dc_luma_neonintr(UWORD16 *pu2_pred,
                                              UWORD16 *pu2_dst,
                                              WORD32 pred_strd,
                                              WORD32 dst_strd,
                                              WORD32 log2_trans_size,
                                              WORD16 i2_coeff_value,
                                              WORD32 bit_depth)
{
    WORD32 row, col;
    WORD32 add, shift;
    WORD32 dc_value, quant_out;
    WORD32 trans_size;
    UWORD16 max_val;

    trans_size = (1 << log2_trans_size);
    quant_out = (WORD32)i2_coeff_value;
    shift = IT_SHIFT_STAGE_1;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((quant_out * 64 + add) >> shift);
    shift = 20 - bit_depth;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((dc_value * 64 + add) >> shift);
    max_val = (1 << bit_depth) - 1;

    if(trans_size == 4)
    {
        int16x4_t v_dc_4 = vdup_n_s16((int16_t)dc_value);
        int16x4_t v_zero_4 = vdup_n_s16(0);
        int16x4_t v_max_4 = vdup_n_s16((int16_t)max_val);

        for(row = 0; row < 4; row++)
        {
            int16x4_t v_p = vreinterpret_s16_u16(vld1_u16(pu2_pred));
            int16x4_t v_res = vqadd_s16(v_p, v_dc_4);
            v_res = vmin_s16(vmax_s16(v_res, v_zero_4), v_max_4);
            vst1_u16(pu2_dst, vreinterpret_u16_s16(v_res));
            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
    }
    else
    {
        int16x8_t v_dc = vdupq_n_s16((int16_t)dc_value);
        int16x8_t v_zero = vdupq_n_s16(0);
        int16x8_t v_max = vdupq_n_s16((int16_t)max_val);

        for(row = 0; row < trans_size; row++)
        {
            for(col = 0; col < trans_size; col += 8)
            {
                int16x8_t v_p = vreinterpretq_s16_u16(vld1q_u16(pu2_pred + col));
                int16x8_t v_res = vqaddq_s16(v_p, v_dc);
                v_res = vminq_s16(vmaxq_s16(v_res, v_zero), v_max);
                vst1q_u16(pu2_dst + col, vreinterpretq_u16_s16(v_res));
            }
            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
    }
}

void ihevcd_hbd_itrans_recon_dc_chroma_neonintr(UWORD16 *pu2_pred,
                                                UWORD16 *pu2_dst,
                                                WORD32 pred_strd,
                                                WORD32 dst_strd,
                                                WORD32 log2_trans_size,
                                                WORD16 i2_coeff_value,
                                                WORD32 bit_depth)
{
    WORD32 row, col;
    WORD32 add, shift;
    WORD32 dc_value, quant_out;
    WORD32 trans_size;
    UWORD16 max_val;

    trans_size = (1 << log2_trans_size);
    quant_out = (WORD32)i2_coeff_value;
    shift = IT_SHIFT_STAGE_1;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((quant_out * 64 + add) >> shift);
    shift = 20 - bit_depth;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((dc_value * 64 + add) >> shift);
    max_val = (1 << bit_depth) - 1;

    if(trans_size == 4)
    {
        int16x4_t v_dc_4 = vdup_n_s16((int16_t)dc_value);
        int16x4_t v_zero_4 = vdup_n_s16(0);
        int16x4_t v_max_4 = vdup_n_s16((int16_t)max_val);

        for(row = 0; row < 4; row++)
        {
            uint16x4x2_t p = vld2_u16(pu2_pred);
            int16x4_t v_res = vqadd_s16(vreinterpret_s16_u16(p.val[0]), v_dc_4);
            v_res = vmin_s16(vmax_s16(v_res, v_zero_4), v_max_4);
            uint16x4x2_t d = vld2_u16(pu2_dst);
            d.val[0] = vreinterpret_u16_s16(v_res);
            vst2_u16(pu2_dst, d);

            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
    }
    else
    {
        int16x8_t v_dc = vdupq_n_s16((int16_t)dc_value);
        int16x8_t v_zero = vdupq_n_s16(0);
        int16x8_t v_max = vdupq_n_s16((int16_t)max_val);

        for(row = 0; row < trans_size; row++)
        {
            for(col = 0; col < trans_size; col += 8)
            {
                uint16x8x2_t p = vld2q_u16(&pu2_pred[col * 2]);
                int16x8_t v_res = vqaddq_s16(vreinterpretq_s16_u16(p.val[0]), v_dc);
                v_res = vminq_s16(vmaxq_s16(v_res, v_zero), v_max);
                uint16x8x2_t d = vld2q_u16(&pu2_dst[col * 2]);
                d.val[0] = vreinterpretq_u16_s16(v_res);
                vst2q_u16(&pu2_dst[col * 2], d);
            }
            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
    }
}
