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
*  ihevcd_fmt_conv_neon_intr.c
*
* @brief
*  Contains ARM NEON intrinsic functions for format conversion
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevcd_fmt_conv_444sp_to_420p_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/

#include <arm_neon.h>

#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevcd_fmt_conv.h"

/**
*******************************************************************************
*
* @brief Function to convert a YUV 4:4:4 semi-planar buffer to a YUV 4:2:0
* planar buffer using ARM NEON intrinsics
*
* @par Description
* Copies the 8-bit luma plane directly and converts the 8-bit interleaved UV
* 4:4:4 semi-planar chroma plane into separate U and V 4:2:0 planar buffers by
* 2x2 box-filter downsampling with rounding.
*
* @param[in] pu1_y_src
*   Input Y pointer
*
* @param[in] pu1_uv_src
*   Input UV pointer (interleaved U, V)
*
* @param[out] pu1_y_dst
*   Output Y pointer
*
* @param[out] pu1_u_dst
*   Output U pointer
*
* @param[out] pu1_v_dst
*   Output V pointer
*
* @param[in] wd
*   Width
*
* @param[in] ht
*   Height
*
* @param[in] src_y_strd
*   Input Y stride
*
* @param[in] src_uv_strd
*   Input UV stride
*
* @param[in] dst_y_strd
*   Output Y stride
*
* @param[in] dst_uv_strd
*   Output U or V stride
*
* @returns None
*
* @remarks None
*
*******************************************************************************
*/
void ihevcd_fmt_conv_444sp_to_420p_neonintr(UWORD8 *pu1_y_src,
                                            UWORD8 *pu1_uv_src,
                                            UWORD8 *pu1_y_dst,
                                            UWORD8 *pu1_u_dst,
                                            UWORD8 *pu1_v_dst,
                                            WORD32 wd,
                                            WORD32 ht,
                                            WORD32 src_y_strd,
                                            WORD32 src_uv_strd,
                                            WORD32 dst_y_strd,
                                            WORD32 dst_uv_strd)
{
    WORD32 i, j;

    /* Copy luma */
    for(i = 0; i < ht; i++)
    {
        for(j = 0; j <= (wd - 64); j += 64)
        {
            uint8x16_t v_y0 = vld1q_u8(pu1_y_src + j);
            uint8x16_t v_y1 = vld1q_u8(pu1_y_src + j + 16);
            uint8x16_t v_y2 = vld1q_u8(pu1_y_src + j + 32);
            uint8x16_t v_y3 = vld1q_u8(pu1_y_src + j + 48);

            vst1q_u8(pu1_y_dst + j, v_y0);
            vst1q_u8(pu1_y_dst + j + 16, v_y1);
            vst1q_u8(pu1_y_dst + j + 32, v_y2);
            vst1q_u8(pu1_y_dst + j + 48, v_y3);
        }

        if(j <= (wd - 32))
        {
            uint8x16_t v_y0 = vld1q_u8(pu1_y_src + j);
            uint8x16_t v_y1 = vld1q_u8(pu1_y_src + j + 16);

            vst1q_u8(pu1_y_dst + j, v_y0);
            vst1q_u8(pu1_y_dst + j + 16, v_y1);
            j += 32;
        }

        if(j <= (wd - 16))
        {
            vst1q_u8(pu1_y_dst + j, vld1q_u8(pu1_y_src + j));
            j += 16;
        }

        if(j <= (wd - 8))
        {
            vst1_u8(pu1_y_dst + j, vld1_u8(pu1_y_src + j));
            j += 8;
        }

        for(; j < wd; j++)
        {
            pu1_y_dst[j] = pu1_y_src[j];
        }

        pu1_y_src += src_y_strd;
        pu1_y_dst += dst_y_strd;
    }

    /* De-interleave U and V and downsample 2:1 horizontally and vertically */
    for(i = 0; (i + 1) < ht; i += 2)
    {
        UWORD8 *pu1_uv_src0 = pu1_uv_src;
        UWORD8 *pu1_uv_src1 = pu1_uv_src + src_uv_strd;

        for(j = 0; j <= (wd - 32); j += 32)
        {
            uint8x16x2_t v_uv0_lo = vld2q_u8(pu1_uv_src0 + j * 2);
            uint8x16x2_t v_uv0_hi = vld2q_u8(pu1_uv_src0 + j * 2 + 32);
            uint8x16x2_t v_uv1_lo = vld2q_u8(pu1_uv_src1 + j * 2);
            uint8x16x2_t v_uv1_hi = vld2q_u8(pu1_uv_src1 + j * 2 + 32);

            uint16x8_t v_u_sum_lo = vpadalq_u8(vpaddlq_u8(v_uv0_lo.val[0]), v_uv1_lo.val[0]);
            uint16x8_t v_u_sum_hi = vpadalq_u8(vpaddlq_u8(v_uv0_hi.val[0]), v_uv1_hi.val[0]);
            uint16x8_t v_v_sum_lo = vpadalq_u8(vpaddlq_u8(v_uv0_lo.val[1]), v_uv1_lo.val[1]);
            uint16x8_t v_v_sum_hi = vpadalq_u8(vpaddlq_u8(v_uv0_hi.val[1]), v_uv1_hi.val[1]);

            uint8x16_t v_u = vcombine_u8(vrshrn_n_u16(v_u_sum_lo, 2), vrshrn_n_u16(v_u_sum_hi, 2));
            uint8x16_t v_v = vcombine_u8(vrshrn_n_u16(v_v_sum_lo, 2), vrshrn_n_u16(v_v_sum_hi, 2));

            vst1q_u8(pu1_u_dst + (j >> 1), v_u);
            vst1q_u8(pu1_v_dst + (j >> 1), v_v);
        }

        if(j <= (wd - 16))
        {
            uint8x16x2_t v_uv0 = vld2q_u8(pu1_uv_src0 + j * 2);
            uint8x16x2_t v_uv1 = vld2q_u8(pu1_uv_src1 + j * 2);

            uint16x8_t v_u_sum = vpadalq_u8(vpaddlq_u8(v_uv0.val[0]), v_uv1.val[0]);
            uint16x8_t v_v_sum = vpadalq_u8(vpaddlq_u8(v_uv0.val[1]), v_uv1.val[1]);

            vst1_u8(pu1_u_dst + (j >> 1), vrshrn_n_u16(v_u_sum, 2));
            vst1_u8(pu1_v_dst + (j >> 1), vrshrn_n_u16(v_v_sum, 2));
            j += 16;
        }

        for(; j < wd; j += 2)
        {
            WORD32 cb_sum = pu1_uv_src0[j * 2] + pu1_uv_src1[j * 2];
            WORD32 cr_sum = pu1_uv_src0[j * 2 + 1] + pu1_uv_src1[j * 2 + 1];

            if((j + 1) < wd)
            {
                cb_sum += pu1_uv_src0[(j + 1) * 2] + pu1_uv_src1[(j + 1) * 2];
                cr_sum += pu1_uv_src0[(j + 1) * 2 + 1] + pu1_uv_src1[(j + 1) * 2 + 1];

                pu1_u_dst[j >> 1] = (cb_sum + 2) >> 2;
                pu1_v_dst[j >> 1] = (cr_sum + 2) >> 2;
            }
            else
            {
                pu1_u_dst[j >> 1] = (cb_sum + 1) >> 1;
                pu1_v_dst[j >> 1] = (cr_sum + 1) >> 1;
            }
        }

        pu1_u_dst += dst_uv_strd;
        pu1_v_dst += dst_uv_strd;
        pu1_uv_src += (src_uv_strd * 2);
    }

    if(i < ht)
    {
        for(j = 0; j <= (wd - 32); j += 32)
        {
            uint8x16x2_t v_uv0_lo = vld2q_u8(pu1_uv_src + j * 2);
            uint8x16x2_t v_uv0_hi = vld2q_u8(pu1_uv_src + j * 2 + 32);

            uint16x8_t v_u_sum_lo = vpaddlq_u8(v_uv0_lo.val[0]);
            uint16x8_t v_u_sum_hi = vpaddlq_u8(v_uv0_hi.val[0]);
            uint16x8_t v_v_sum_lo = vpaddlq_u8(v_uv0_lo.val[1]);
            uint16x8_t v_v_sum_hi = vpaddlq_u8(v_uv0_hi.val[1]);

            uint8x16_t v_u = vcombine_u8(vrshrn_n_u16(v_u_sum_lo, 1), vrshrn_n_u16(v_u_sum_hi, 1));
            uint8x16_t v_v = vcombine_u8(vrshrn_n_u16(v_v_sum_lo, 1), vrshrn_n_u16(v_v_sum_hi, 1));

            vst1q_u8(pu1_u_dst + (j >> 1), v_u);
            vst1q_u8(pu1_v_dst + (j >> 1), v_v);
        }

        if(j <= (wd - 16))
        {
            uint8x16x2_t v_uv0 = vld2q_u8(pu1_uv_src + j * 2);

            uint16x8_t v_u_sum = vpaddlq_u8(v_uv0.val[0]);
            uint16x8_t v_v_sum = vpaddlq_u8(v_uv0.val[1]);

            vst1_u8(pu1_u_dst + (j >> 1), vrshrn_n_u16(v_u_sum, 1));
            vst1_u8(pu1_v_dst + (j >> 1), vrshrn_n_u16(v_v_sum, 1));
            j += 16;
        }

        for(; j < wd; j += 2)
        {
            WORD32 cb_sum = pu1_uv_src[j * 2];
            WORD32 cr_sum = pu1_uv_src[j * 2 + 1];

            if((j + 1) < wd)
            {
                cb_sum = (cb_sum + pu1_uv_src[(j + 1) * 2] + 1) >> 1;
                cr_sum = (cr_sum + pu1_uv_src[(j + 1) * 2 + 1] + 1) >> 1;
            }
            pu1_u_dst[j >> 1] = cb_sum;
            pu1_v_dst[j >> 1] = cr_sum;
        }
    }
}
