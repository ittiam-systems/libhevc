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
*  - ihevcd_hbd_fmt_conv_444sp_to_420p_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/

#include <arm_neon.h>
#include <string.h>

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

/**
*******************************************************************************
*
* @brief Function to convert a High Bit Depth 444SP buffer to 420P buffer using NEON
*
* @par   Description
* This function handles format conversion from a 4:4:4 semi-planar (interleaved UV)
* buffer to a 4:2:0 planar output buffer for high bit depth video.
* Luma is optionally copied. Chroma is de-interleaved and vertically and horizontally
* downsampled by averaging 2x2 chroma blocks with rounding: (c00 + c01 + c10 + c11 + 2) >> 2.
*
* @param[in] pu2_y_src
*   Input Y pointer
*
* @param[in] pu2_uv_src
*   Input UV pointer (UV is interleaved)
*
* @param[out] pu2_y_dst
*   Output Y pointer
*
* @param[out] pu2_u_dst
*   Output U pointer
*
* @param[out] pu2_v_dst
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
*   Output UV stride
*
* @param[in] is_u_first
*   Flag to indicate if U is first in interleaved chroma
*
* @param[in] disable_luma_copy
*   Flag to disable luma copy
*
* @returns None
*
*******************************************************************************
*/
void ihevcd_hbd_fmt_conv_444sp_to_420p_neonintr(UWORD16 *pu2_y_src,
                                                UWORD16 *pu2_uv_src,
                                                UWORD16 *pu2_y_dst,
                                                UWORD16 *pu2_u_dst,
                                                UWORD16 *pu2_v_dst,
                                                WORD32 wd,
                                                WORD32 ht,
                                                WORD32 src_y_strd,
                                                WORD32 src_uv_strd,
                                                WORD32 dst_y_strd,
                                                WORD32 dst_uv_strd,
                                                WORD32 is_u_first,
                                                WORD32 disable_luma_copy)
{
    WORD32 num_rows, num_cols, src_strd, dst_strd, cols;
    WORD32 i, j;

    if (0 == disable_luma_copy)
    {
        /* copy luma */
        UWORD16* pu2_src = (UWORD16*)pu2_y_src;
        UWORD16* pu2_dst = (UWORD16*)pu2_y_dst;

        num_rows = ht;
        num_cols = wd;

        src_strd = src_y_strd;
        dst_strd = dst_y_strd;
        for (i = 0; i < num_rows; i++)
        {
            memcpy(pu2_dst, pu2_src, num_cols * sizeof(UWORD16));
            pu2_dst += dst_strd;
            pu2_src += src_strd;
        }
    }

    /* de-interleave U and V, downsample horizontally and vertically, and copy to destination */
    if (!is_u_first)
    {
        UWORD16* temp = pu2_u_dst;
        pu2_u_dst = pu2_v_dst;
        pu2_v_dst = temp;
    }

    {
        uint16x8x2_t r0_0, r0_1, r0_2, r0_3;
        uint16x8x2_t r1_0, r1_1, r1_2, r1_3;
        uint32x4_t sum_u0, sum_u1, sum_v0, sum_v1;
        uint16x8_t u_out, v_out;
        WORD32 col_offset = 0;

        num_rows = ht >> 1;
        num_cols = wd >> 1;

        src_strd = src_uv_strd;
        dst_strd = dst_uv_strd;

        cols = num_cols >> 4;

        if (cols > 0)
        {
            UWORD16* pu2_uv_src_row = pu2_uv_src;
            UWORD16* pu2_u_dst_row = pu2_u_dst;
            UWORD16* pu2_v_dst_row = pu2_v_dst;

            /* 2-row unrolled loop across vertical row pairs */
            for (i = 0; i < (num_rows >> 1); i++)
            {
                UWORD16* pu2_uv_src_temp = pu2_uv_src_row;
                UWORD16* pu2_u_dst_temp = pu2_u_dst_row;
                UWORD16* pu2_v_dst_temp = pu2_v_dst_row;

                for (j = 0; j < cols; j++)
                {
                    /**** Output Row 0 (Input Rows 0 & 1) ****/
                    /* First 8 output columns (16 input columns) */
                    r0_0 = vld2q_u16(pu2_uv_src_temp);
                    r0_1 = vld2q_u16(pu2_uv_src_temp + 16);
                    r1_0 = vld2q_u16(pu2_uv_src_temp + src_strd);
                    r1_1 = vld2q_u16(pu2_uv_src_temp + src_strd + 16);

                    sum_u0 = vpadalq_u16(vpaddlq_u16(r0_0.val[0]), r1_0.val[0]);
                    sum_u1 = vpadalq_u16(vpaddlq_u16(r0_1.val[0]), r1_1.val[0]);
                    u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 2), vrshrn_n_u32(sum_u1, 2));
                    vst1q_u16(pu2_u_dst_temp, u_out);

                    sum_v0 = vpadalq_u16(vpaddlq_u16(r0_0.val[1]), r1_0.val[1]);
                    sum_v1 = vpadalq_u16(vpaddlq_u16(r0_1.val[1]), r1_1.val[1]);
                    v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 2), vrshrn_n_u32(sum_v1, 2));
                    vst1q_u16(pu2_v_dst_temp, v_out);

                    /* Next 8 output columns (16 input columns) */
                    r0_2 = vld2q_u16(pu2_uv_src_temp + 32);
                    r0_3 = vld2q_u16(pu2_uv_src_temp + 48);
                    r1_2 = vld2q_u16(pu2_uv_src_temp + src_strd + 32);
                    r1_3 = vld2q_u16(pu2_uv_src_temp + src_strd + 48);

                    sum_u0 = vpadalq_u16(vpaddlq_u16(r0_2.val[0]), r1_2.val[0]);
                    sum_u1 = vpadalq_u16(vpaddlq_u16(r0_3.val[0]), r1_3.val[0]);
                    u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 2), vrshrn_n_u32(sum_u1, 2));
                    vst1q_u16(pu2_u_dst_temp + 8, u_out);

                    sum_v0 = vpadalq_u16(vpaddlq_u16(r0_2.val[1]), r1_2.val[1]);
                    sum_v1 = vpadalq_u16(vpaddlq_u16(r0_3.val[1]), r1_3.val[1]);
                    v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 2), vrshrn_n_u32(sum_v1, 2));
                    vst1q_u16(pu2_v_dst_temp + 8, v_out);

                    /**** Output Row 1 (Input Rows 2 & 3) ****/
                    /* First 8 output columns (16 input columns) */
                    r0_0 = vld2q_u16(pu2_uv_src_temp + 2 * src_strd);
                    r0_1 = vld2q_u16(pu2_uv_src_temp + 2 * src_strd + 16);
                    r1_0 = vld2q_u16(pu2_uv_src_temp + 3 * src_strd);
                    r1_1 = vld2q_u16(pu2_uv_src_temp + 3 * src_strd + 16);

                    sum_u0 = vpadalq_u16(vpaddlq_u16(r0_0.val[0]), r1_0.val[0]);
                    sum_u1 = vpadalq_u16(vpaddlq_u16(r0_1.val[0]), r1_1.val[0]);
                    u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 2), vrshrn_n_u32(sum_u1, 2));
                    vst1q_u16(pu2_u_dst_temp + dst_strd, u_out);

                    sum_v0 = vpadalq_u16(vpaddlq_u16(r0_0.val[1]), r1_0.val[1]);
                    sum_v1 = vpadalq_u16(vpaddlq_u16(r0_1.val[1]), r1_1.val[1]);
                    v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 2), vrshrn_n_u32(sum_v1, 2));
                    vst1q_u16(pu2_v_dst_temp + dst_strd, v_out);

                    /* Next 8 output columns (16 input columns) */
                    r0_2 = vld2q_u16(pu2_uv_src_temp + 2 * src_strd + 32);
                    r0_3 = vld2q_u16(pu2_uv_src_temp + 2 * src_strd + 48);
                    r1_2 = vld2q_u16(pu2_uv_src_temp + 3 * src_strd + 32);
                    r1_3 = vld2q_u16(pu2_uv_src_temp + 3 * src_strd + 48);

                    sum_u0 = vpadalq_u16(vpaddlq_u16(r0_2.val[0]), r1_2.val[0]);
                    sum_u1 = vpadalq_u16(vpaddlq_u16(r0_3.val[0]), r1_3.val[0]);
                    u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 2), vrshrn_n_u32(sum_u1, 2));
                    vst1q_u16(pu2_u_dst_temp + dst_strd + 8, u_out);

                    sum_v0 = vpadalq_u16(vpaddlq_u16(r0_2.val[1]), r1_2.val[1]);
                    sum_v1 = vpadalq_u16(vpaddlq_u16(r0_3.val[1]), r1_3.val[1]);
                    v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 2), vrshrn_n_u32(sum_v1, 2));
                    vst1q_u16(pu2_v_dst_temp + dst_strd + 8, v_out);

                    pu2_u_dst_temp += 16;
                    pu2_v_dst_temp += 16;
                    pu2_uv_src_temp += 64;
                }

                pu2_u_dst_row += 2 * dst_strd;
                pu2_v_dst_row += 2 * dst_strd;
                pu2_uv_src_row += 4 * src_strd;
            }

            /* Single remaining output row pair if num_rows is odd */
            if (num_rows & 1)
            {
                UWORD16* pu2_uv_src_temp = pu2_uv_src_row;
                UWORD16* pu2_u_dst_temp = pu2_u_dst_row;
                UWORD16* pu2_v_dst_temp = pu2_v_dst_row;

                for (j = 0; j < cols; j++)
                {
                    r0_0 = vld2q_u16(pu2_uv_src_temp);
                    r0_1 = vld2q_u16(pu2_uv_src_temp + 16);
                    r1_0 = vld2q_u16(pu2_uv_src_temp + src_strd);
                    r1_1 = vld2q_u16(pu2_uv_src_temp + src_strd + 16);

                    sum_u0 = vpadalq_u16(vpaddlq_u16(r0_0.val[0]), r1_0.val[0]);
                    sum_u1 = vpadalq_u16(vpaddlq_u16(r0_1.val[0]), r1_1.val[0]);
                    u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 2), vrshrn_n_u32(sum_u1, 2));
                    vst1q_u16(pu2_u_dst_temp, u_out);

                    sum_v0 = vpadalq_u16(vpaddlq_u16(r0_0.val[1]), r1_0.val[1]);
                    sum_v1 = vpadalq_u16(vpaddlq_u16(r0_1.val[1]), r1_1.val[1]);
                    v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 2), vrshrn_n_u32(sum_v1, 2));
                    vst1q_u16(pu2_v_dst_temp, v_out);

                    r0_2 = vld2q_u16(pu2_uv_src_temp + 32);
                    r0_3 = vld2q_u16(pu2_uv_src_temp + 48);
                    r1_2 = vld2q_u16(pu2_uv_src_temp + src_strd + 32);
                    r1_3 = vld2q_u16(pu2_uv_src_temp + src_strd + 48);

                    sum_u0 = vpadalq_u16(vpaddlq_u16(r0_2.val[0]), r1_2.val[0]);
                    sum_u1 = vpadalq_u16(vpaddlq_u16(r0_3.val[0]), r1_3.val[0]);
                    u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 2), vrshrn_n_u32(sum_u1, 2));
                    vst1q_u16(pu2_u_dst_temp + 8, u_out);

                    sum_v0 = vpadalq_u16(vpaddlq_u16(r0_2.val[1]), r1_2.val[1]);
                    sum_v1 = vpadalq_u16(vpaddlq_u16(r0_3.val[1]), r1_3.val[1]);
                    v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 2), vrshrn_n_u32(sum_v1, 2));
                    vst1q_u16(pu2_v_dst_temp + 8, v_out);

                    pu2_u_dst_temp += 16;
                    pu2_v_dst_temp += 16;
                    pu2_uv_src_temp += 64;
                }
            }

            col_offset = cols << 4;
            num_cols &= 0x0F;
        }

        /* 8-column vector branch before scalar loop */
        if (num_cols >= 8)
        {
            UWORD16* pu2_uv_src_temp = pu2_uv_src + 4 * col_offset;
            UWORD16* pu2_u_dst_temp = pu2_u_dst + col_offset;
            UWORD16* pu2_v_dst_temp = pu2_v_dst + col_offset;

            for (i = 0; i < num_rows; i++)
            {
                r0_0 = vld2q_u16(pu2_uv_src_temp);
                r0_1 = vld2q_u16(pu2_uv_src_temp + 16);
                r1_0 = vld2q_u16(pu2_uv_src_temp + src_strd);
                r1_1 = vld2q_u16(pu2_uv_src_temp + src_strd + 16);

                sum_u0 = vpadalq_u16(vpaddlq_u16(r0_0.val[0]), r1_0.val[0]);
                sum_u1 = vpadalq_u16(vpaddlq_u16(r0_1.val[0]), r1_1.val[0]);
                u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 2), vrshrn_n_u32(sum_u1, 2));
                vst1q_u16(pu2_u_dst_temp, u_out);

                sum_v0 = vpadalq_u16(vpaddlq_u16(r0_0.val[1]), r1_0.val[1]);
                sum_v1 = vpadalq_u16(vpaddlq_u16(r0_1.val[1]), r1_1.val[1]);
                v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 2), vrshrn_n_u32(sum_v1, 2));
                vst1q_u16(pu2_v_dst_temp, v_out);

                pu2_u_dst_temp += dst_strd;
                pu2_v_dst_temp += dst_strd;
                pu2_uv_src_temp += 2 * src_strd;
            }

            col_offset += 8;
            num_cols -= 8;
        }

        /* Scalar loop for remaining columns (1 to 7) */
        if (num_cols > 0)
        {
            UWORD16* pu2_uv_src_temp = pu2_uv_src + 4 * col_offset;
            UWORD16* pu2_u_dst_temp = pu2_u_dst + col_offset;
            UWORD16* pu2_v_dst_temp = pu2_v_dst + col_offset;

            for (i = 0; i < num_rows; i++)
            {
                for (j = 0; j < num_cols; j++)
                {
                    WORD32 cb_sum = pu2_uv_src_temp[j * 4] +
                        pu2_uv_src_temp[j * 4 + 2] +
                        pu2_uv_src_temp[j * 4 + src_strd] +
                        pu2_uv_src_temp[j * 4 + src_strd + 2];
                    WORD32 cr_sum = pu2_uv_src_temp[j * 4 + 1] +
                        pu2_uv_src_temp[j * 4 + 3] +
                        pu2_uv_src_temp[j * 4 + src_strd + 1] +
                        pu2_uv_src_temp[j * 4 + src_strd + 3];

                    pu2_u_dst_temp[j] = (cb_sum + 2) >> 2;
                    pu2_v_dst_temp[j] = (cr_sum + 2) >> 2;
                }

                pu2_u_dst_temp += dst_strd;
                pu2_v_dst_temp += dst_strd;
                pu2_uv_src_temp += 2 * src_strd;
            }
        }

        /* Trailing column if width is odd (count = 2 for vertical pair) */
        if (wd & 1)
        {
            WORD32 last_col = wd >> 1;
            UWORD16* pu2_uv_src_temp = pu2_uv_src + (wd - 1) * 2;
            UWORD16* pu2_u_dst_temp = pu2_u_dst + last_col;
            UWORD16* pu2_v_dst_temp = pu2_v_dst + last_col;

            for (i = 0; i < num_rows; i++)
            {
                WORD32 cb_sum = pu2_uv_src_temp[0] + pu2_uv_src_temp[src_strd];
                WORD32 cr_sum = pu2_uv_src_temp[1] + pu2_uv_src_temp[src_strd + 1];

                *pu2_u_dst_temp = (cb_sum + 1) >> 1;
                *pu2_v_dst_temp = (cr_sum + 1) >> 1;

                pu2_u_dst_temp += dst_strd;
                pu2_v_dst_temp += dst_strd;
                pu2_uv_src_temp += 2 * src_strd;
            }
        }

        /* Trailing row if height is odd */
        if (ht & 1)
        {
            WORD32 last_row = ht >> 1;
            UWORD16* pu2_uv_src_last = pu2_uv_src + (ht - 1) * src_strd;
            UWORD16* pu2_u_dst_last = pu2_u_dst + last_row * dst_strd;
            UWORD16* pu2_v_dst_last = pu2_v_dst + last_row * dst_strd;
            WORD32 total_cols = wd >> 1;

            j = 0;
            for (; j <= total_cols - 8; j += 8)
            {
                r0_0 = vld2q_u16(pu2_uv_src_last + 4 * j);
                r0_1 = vld2q_u16(pu2_uv_src_last + 4 * j + 16);

                sum_u0 = vpaddlq_u16(r0_0.val[0]);
                sum_u1 = vpaddlq_u16(r0_1.val[0]);
                u_out = vcombine_u16(vrshrn_n_u32(sum_u0, 1), vrshrn_n_u32(sum_u1, 1));
                vst1q_u16(pu2_u_dst_last + j, u_out);

                sum_v0 = vpaddlq_u16(r0_0.val[1]);
                sum_v1 = vpaddlq_u16(r0_1.val[1]);
                v_out = vcombine_u16(vrshrn_n_u32(sum_v0, 1), vrshrn_n_u32(sum_v1, 1));
                vst1q_u16(pu2_v_dst_last + j, v_out);
            }
            for (; j < total_cols; j++)
            {
                WORD32 cb_sum = pu2_uv_src_last[j * 4] + pu2_uv_src_last[j * 4 + 2];
                WORD32 cr_sum = pu2_uv_src_last[j * 4 + 1] + pu2_uv_src_last[j * 4 + 3];

                pu2_u_dst_last[j] = (cb_sum + 1) >> 1;
                pu2_v_dst_last[j] = (cr_sum + 1) >> 1;
            }

            /* Corner pixel if both wd and ht are odd (count = 1) */
            if (wd & 1)
            {
                pu2_u_dst_last[total_cols] = pu2_uv_src_last[(wd - 1) * 2];
                pu2_v_dst_last[total_cols] = pu2_uv_src_last[(wd - 1) * 2 + 1];
            }
        }
    }
}
