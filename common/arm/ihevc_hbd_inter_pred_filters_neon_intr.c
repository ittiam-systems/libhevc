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
*  ihevc_hbd_inter_pred_filters_neon_intr.c
*
* @brief
*  Contains function definitions for HBD inter prediction interpolation filters
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_inter_pred_luma_copy_neonintr()
*  - ihevc_hbd_inter_pred_luma_copy_w16out_neonintr()
*  - ihevc_hbd_inter_pred_luma_horz_neonintr()
*  - ihevc_hbd_inter_pred_luma_vert_neonintr()
*  - ihevc_hbd_inter_pred_luma_horz_w16out_neonintr()
*  - ihevc_hbd_inter_pred_luma_vert_w16out_neonintr()
*  - ihevc_hbd_inter_pred_luma_vert_w16inp_neonintr()
*  - ihevc_hbd_inter_pred_luma_vert_w16inp_w16out_neonintr()
*  - ihevc_hbd_inter_pred_chroma_copy_neonintr()
*  - ihevc_hbd_inter_pred_chroma_copy_w16out_neonintr()
*  - ihevc_hbd_inter_pred_chroma_horz_neonintr()
*  - ihevc_hbd_inter_pred_chroma_vert_neonintr()
*  - ihevc_hbd_inter_pred_chroma_horz_w16out_neonintr()
*  - ihevc_hbd_inter_pred_chroma_vert_w16out_neonintr()
*  - ihevc_hbd_inter_pred_chroma_vert_w16inp_neonintr()
*  - ihevc_hbd_inter_pred_chroma_vert_w16inp_w16out_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/

/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <arm_neon.h>

#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_inter_pred.h"

/**
*******************************************************************************
*
* @brief
*   Interprediction luma function for copy
*
* @par Description:
*   Copies the array of width 'wd' and height 'ht' from the location pointed
*   by 'pu2_src' to the location pointed by 'pu2_dst'
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients (unused)
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples (unused)
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_copy_neonintr(UWORD16 *pu2_src,
                                             UWORD16 *pu2_dst,
                                             WORD32 src_strd,
                                             WORD32 dst_strd,
                                             WORD8 *pi1_coeff,
                                             WORD32 ht,
                                             WORD32 wd,
                                             UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 src_strd2 = src_strd << 1;
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = src_strd << 2;
    WORD32 dst_strd2 = dst_strd << 1;
    WORD32 dst_strd3 = dst_strd2 + dst_strd;
    WORD32 dst_strd4 = dst_strd << 2;

    UNUSED(pi1_coeff);
    UNUSED(bit_depth);

    if (!(wd & 15))
    {
        for (row = 0; row < ht; row += 4)
        {
            for (col = 0; col < wd; col += 16)
            {
                uint16x8_t r0_0 = vld1q_u16(pu2_src + col);
                uint16x8_t r0_1 = vld1q_u16(pu2_src + col + 8);
                uint16x8_t r1_0 = vld1q_u16(pu2_src + src_strd + col);
                uint16x8_t r1_1 = vld1q_u16(pu2_src + src_strd + col + 8);
                uint16x8_t r2_0 = vld1q_u16(pu2_src + src_strd2 + col);
                uint16x8_t r2_1 = vld1q_u16(pu2_src + src_strd2 + col + 8);
                uint16x8_t r3_0 = vld1q_u16(pu2_src + src_strd3 + col);
                uint16x8_t r3_1 = vld1q_u16(pu2_src + src_strd3 + col + 8);

                vst1q_u16(pu2_dst + col, r0_0);
                vst1q_u16(pu2_dst + col + 8, r0_1);
                vst1q_u16(pu2_dst + dst_strd + col, r1_0);
                vst1q_u16(pu2_dst + dst_strd + col + 8, r1_1);
                vst1q_u16(pu2_dst + dst_strd2 + col, r2_0);
                vst1q_u16(pu2_dst + dst_strd2 + col + 8, r2_1);
                vst1q_u16(pu2_dst + dst_strd3 + col, r3_0);
                vst1q_u16(pu2_dst + dst_strd3 + col + 8, r3_1);
            }
            pu2_src += src_strd4;
            pu2_dst += dst_strd4;
        }
    }
    else if (!(wd & 7))
    {
        for (row = 0; row < ht; row += 4)
        {
            for (col = 0; col < wd; col += 8)
            {
                uint16x8_t r0 = vld1q_u16(pu2_src + col);
                uint16x8_t r1 = vld1q_u16(pu2_src + src_strd + col);
                uint16x8_t r2 = vld1q_u16(pu2_src + src_strd2 + col);
                uint16x8_t r3 = vld1q_u16(pu2_src + src_strd3 + col);

                vst1q_u16(pu2_dst + col, r0);
                vst1q_u16(pu2_dst + dst_strd + col, r1);
                vst1q_u16(pu2_dst + dst_strd2 + col, r2);
                vst1q_u16(pu2_dst + dst_strd3 + col, r3);
            }
            pu2_src += src_strd4;
            pu2_dst += dst_strd4;
        }
    }
    else
    {
        for (row = 0; row < ht; row += 4)
        {
            for (col = 0; col < wd; col += 4)
            {
                uint16x4_t r0 = vld1_u16(pu2_src + col);
                uint16x4_t r1 = vld1_u16(pu2_src + src_strd + col);
                uint16x4_t r2 = vld1_u16(pu2_src + src_strd2 + col);
                uint16x4_t r3 = vld1_u16(pu2_src + src_strd3 + col);

                vst1_u16(pu2_dst + col, r0);
                vst1_u16(pu2_dst + dst_strd + col, r1);
                vst1_u16(pu2_dst + dst_strd2 + col, r2);
                vst1_u16(pu2_dst + dst_strd3 + col, r3);
            }
            pu2_src += src_strd4;
            pu2_dst += dst_strd4;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     Interprediction luma filter for horizontal input
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and writes to the location pointed
*    by 'pu2_dst'. The output is downshifted by 6 and clipped to bit_depth.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_horz_neonintr(UWORD16 *pu2_src,
                                             UWORD16 *pu2_dst,
                                             WORD32 src_strd,
                                             WORD32 dst_strd,
                                             WORD8 *pi1_coeff,
                                             WORD32 ht,
                                             WORD32 wd,
                                             UWORD8 bit_depth)
{
    int8x8_t coeffS1 = vld1_s8(pi1_coeff);
    pu2_src -= 3;
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 x, y;

    uint16x8_t aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2, aSrc4U2, aSrc5U2, aSrc6U2, aSrc7U2;
    uint16x8_t bSrc0U2, bSrc1U2, bSrc2U2, bSrc3U2, bSrc4U2, bSrc5U2, bSrc6U2, bSrc7U2;
    uint16x8_t mul0U2, mul1U2, mul2U2, mul3U2, mul4U2, mul5U2, res0U2, res1U2;
    uint16x4_t sft0U2, sft1U2, sft2U2, sft3U2, mul6U2;
    uint32x4_t add0U4, add1U4, add2U4, add3U4;
    int32x4_t  sub0U4, sub1U4, sub2U4, sub3U4, mul0S4;
    int16x4_t  src0S2, src1S2, src2S2, src3S2, src4S2, src5S2, src6S2, src7S2;
    int16x4_t  coeff0S2, coeff1S2, coeff2S2, coeff3S2, coeff4S2, coeff5S2, coeff6S2, coeff7S2;
    uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2, coeff4U2, coeff5U2, coeff6U2, coeff7U2;

#define CALCULATE_2ROWS_WD8x(pSrc, pDst)                                 \
        aSrc0U2 = vld1q_u16(pSrc);                                       \
        aSrc1U2 = vld1q_u16(pSrc + 1);                                   \
        bSrc0U2 = vld1q_u16(pSrc + src_strd);                            \
        bSrc1U2 = vld1q_u16(pSrc + src_strd + 1);                        \
        mul0U2 = vmulq_u16(aSrc0U2, coeff0U2);                           \
        mul1U2 = vmulq_u16(aSrc1U2, coeff1U2);                           \
        mul3U2 = vmulq_u16(bSrc0U2, coeff0U2);                           \
        mul4U2 = vmulq_u16(bSrc1U2, coeff1U2);                           \
        aSrc2U2 = vld1q_u16(pSrc + 2);                                   \
        aSrc3U2 = vld1q_u16(pSrc + 3);                                   \
        bSrc2U2 = vld1q_u16(pSrc + src_strd + 2);                        \
        bSrc3U2 = vld1q_u16(pSrc + src_strd + 3);                        \
        mul0U2 = vmlaq_u16(mul0U2, aSrc2U2, coeff2U2);                   \
        mul1U2 = vmlaq_u16(mul1U2, aSrc3U2, coeff3U2);                   \
        mul3U2 = vmlaq_u16(mul3U2, bSrc2U2, coeff2U2);                   \
        mul4U2 = vmlaq_u16(mul4U2, bSrc3U2, coeff3U2);                   \
        aSrc4U2 = vld1q_u16(pSrc + 4);                                   \
        aSrc5U2 = vld1q_u16(pSrc + 5);                                   \
        bSrc4U2 = vld1q_u16(pSrc + src_strd + 4);                        \
        bSrc5U2 = vld1q_u16(pSrc + src_strd + 5);                        \
        mul2U2 = vmulq_u16(aSrc4U2, coeff4U2);                           \
        mul0U2 = vmlaq_u16(mul0U2, aSrc5U2, coeff5U2);                   \
        mul5U2 = vmulq_u16(bSrc4U2, coeff4U2);                           \
        mul3U2 = vmlaq_u16(mul3U2, bSrc5U2, coeff5U2);                   \
        aSrc6U2 = vld1q_u16(pSrc + 6);                                   \
        aSrc7U2 = vld1q_u16(pSrc + 7);                                   \
        bSrc6U2 = vld1q_u16(pSrc + src_strd + 6);                        \
        bSrc7U2 = vld1q_u16(pSrc + src_strd + 7);                        \
        mul2U2 = vmlaq_u16(mul2U2, aSrc6U2, coeff6U2);                   \
        mul0U2 = vmlaq_u16(mul0U2, aSrc7U2, coeff7U2);                   \
        mul5U2 = vmlaq_u16(mul5U2, bSrc6U2, coeff6U2);                   \
        mul3U2 = vmlaq_u16(mul3U2, bSrc7U2, coeff7U2);                   \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));  \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));\
        add2U4 = vaddl_u16(vget_low_u16(mul4U2), vget_low_u16(mul5U2));  \
        add3U4 = vaddl_u16(vget_high_u16(mul4U2), vget_high_u16(mul5U2));\
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2))); \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16( \
                                         vreinterpretq_s16_u16(mul0U2)));\
        sub2U4 = vsubw_s16(vreinterpretq_s32_u32(add2U4),                \
                            vreinterpret_s16_u16(vget_low_u16(mul3U2))); \
        sub3U4 = vsubw_s16(vreinterpretq_s32_u32(add3U4), vget_high_s16( \
                                        vreinterpretq_s16_u16(mul3U2))); \
        sft0U2 = vqrshrun_n_s32(sub0U4, 6);                              \
        sft1U2 = vqrshrun_n_s32(sub1U4, 6);                              \
        sft2U2 = vqrshrun_n_s32(sub2U4, 6);                              \
        sft3U2 = vqrshrun_n_s32(sub3U4, 6);                              \
        res0U2 = vcombine_u16(sft0U2, sft1U2);                           \
        res1U2 = vcombine_u16(sft2U2, sft3U2);                           \
        res0U2 = vminq_u16(res0U2, max_10bit);                           \
        res1U2 = vminq_u16(res1U2, max_10bit);                           \
        vst1q_u16(pDst, res0U2);                                         \
        vst1q_u16(pDst + dst_strd, res1U2);

#define CALCULATE_ROW_WD4x(pSrc, pDst)                                   \
        src0S2 = vld1_s16((const int16_t*)(pSrc));                       \
        src1S2 = vld1_s16((const int16_t*)(pSrc + 1));                   \
        src2S2 = vld1_s16((const int16_t*)(pSrc + 2));                   \
        src3S2 = vld1_s16((const int16_t*)(pSrc + 3));                   \
        src4S2 = vld1_s16((const int16_t*)(pSrc + 4));                   \
        src5S2 = vld1_s16((const int16_t*)(pSrc + 5));                   \
        src6S2 = vld1_s16((const int16_t*)(pSrc + 6));                   \
        src7S2 = vld1_s16((const int16_t*)(pSrc + 7));                   \
        mul0S4 = vmull_s16(src1S2, coeff1S2);                            \
        mul0S4 = vmlal_s16(mul0S4, src0S2, coeff0S2);                    \
        mul0S4 = vmlal_s16(mul0S4, src3S2, coeff3S2);                    \
        mul0S4 = vmlal_s16(mul0S4, src2S2, coeff2S2);                    \
        mul0S4 = vmlal_s16(mul0S4, src4S2, coeff4S2);                    \
        mul0S4 = vmlal_s16(mul0S4, src5S2, coeff5S2);                    \
        mul0S4 = vmlal_s16(mul0S4, src6S2, coeff6S2);                    \
        mul0S4 = vmlal_s16(mul0S4, src7S2, coeff7S2);                    \
        mul6U2 = vqrshrun_n_s32(mul0S4, 6);                              \
        mul6U2 = vmin_u16(mul6U2, max_10bit);                            \
        vst1_u16(pDst, mul6U2);

    if (!(wd & 7))
    {
        uint16x8_t max_10bit = vdupq_n_u16((1 << bit_depth) - 1);
        uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(vabs_s8(coeffS1)));
        coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
        coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
        coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
        coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);
        coeff4U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 0);
        coeff5U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 1);
        coeff6U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 2);
        coeff7U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 3);

        if (!(wd & 31))
        {
            for (y = 0; y < ht; y += 2)
            {
                for (x = 0; x < wd; x += 32)
                {
                    CALCULATE_2ROWS_WD8x(pu2_src + x, pu2_dst + x)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 8, pu2_dst + x + 8)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 16, pu2_dst + x + 16)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 24, pu2_dst + x + 24)
                }
                pu2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
        else if (!(wd & 15))
        {
            for (y = 0; y < ht; y += 2)
            {
                for (x = 0; x < wd; x += 16)
                {
                    CALCULATE_2ROWS_WD8x(pu2_src + x, pu2_dst + x)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 8, pu2_dst + x + 8)
                }
                pu2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
        else
        {
            for (y = 0; y < ht; y += 4)
            {
                for (x = 0; x < wd; x += 8)
                {
                    CALCULATE_2ROWS_WD8x(pu2_src + x, pu2_dst + x)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + src_strd2,
                                         pu2_dst + x + dst_strd2)
                }
                pu2_src += (src_strd2 + src_strd2);
                pu2_dst += (dst_strd2 + dst_strd2);
            }
        }
    }
    else if (!(wd & 3))
    {
        uint16x4_t max_10bit = vdup_n_u16((1 << bit_depth) - 1);
        int16x8_t coeffS2 = vmovl_s8(coeffS1);
        coeff0S2 = vdup_lane_s16(vget_low_s16(coeffS2), 0);
        coeff1S2 = vdup_lane_s16(vget_low_s16(coeffS2), 1);
        coeff2S2 = vdup_lane_s16(vget_low_s16(coeffS2), 2);
        coeff3S2 = vdup_lane_s16(vget_low_s16(coeffS2), 3);
        coeff4S2 = vdup_lane_s16(vget_high_s16(coeffS2), 0);
        coeff5S2 = vdup_lane_s16(vget_high_s16(coeffS2), 1);
        coeff6S2 = vdup_lane_s16(vget_high_s16(coeffS2), 2);
        coeff7S2 = vdup_lane_s16(vget_high_s16(coeffS2), 3);

        for (y = 0; y < ht; y += 2)
        {
            for (x = 0; x < wd; x += 4)
            {
                CALCULATE_ROW_WD4x(pu2_src + x, pu2_dst + x)
                CALCULATE_ROW_WD4x(pu2_src + x + src_strd, pu2_dst + x + dst_strd)
            }
            pu2_src += src_strd2;
            pu2_dst += dst_strd2;
        }
    }
#undef CALCULATE_2ROWS_WD8x
#undef CALCULATE_ROW_WD4x
}

/**
*******************************************************************************
*
* @brief
*    Interprediction luma filter for vertical input
*
* @par Description:
*   Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*   the elements pointed by 'pu2_src' and writes to the location pointed by
*   'pu2_dst'. The output is downshifted by 6 and clipped to bit_depth.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_vert_neonintr(UWORD16 *pu2_src,
                                             UWORD16 *pu2_dst,
                                             WORD32 src_strd,
                                             WORD32 dst_strd,
                                             WORD8 *pi1_coeff,
                                             WORD32 ht,
                                             WORD32 wd,
                                             UWORD8 bit_depth)
{
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = (src_strd << 2);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 dst_strd3 = dst_strd2 + dst_strd;
    WORD32 dst_strd4 = (dst_strd << 2);
    WORD32 x, y;

    int8x8_t  coeffS1 = vabs_s8(vld1_s8(pi1_coeff));
    uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(coeffS1));

    uint16x8_t src0U2, src1U2, src2U2, src3U2, src4U2, src5U2;
    uint16x8_t src6U2, src7U2, src8U2, src9U2, src10U2;
    uint16x8_t res0U2, res1U2, res2U2, sol0U2, sol1U2;
    uint16x4_t sft0U2, sft1U2;
    uint32x4_t add0U4, add1U4;
    int32x4_t  sub0U4, sub1U4;

#define CALCULATE_ROW_WD8x(src0, src1, src2, src3, src4,                    \
                                             src5, src6, src7, out)         \
            res0U2 = vmulq_u16(src0, coeff0U2);                             \
            res1U2 = vmulq_u16(src1, coeff1U2);                             \
            res0U2 = vmlaq_u16(res0U2, src2, coeff2U2);                     \
            res1U2 = vmlaq_u16(res1U2, src3, coeff3U2);                     \
            res2U2 = vmulq_u16(src4, coeff4U2);                             \
            res0U2 = vmlaq_u16(res0U2, src5, coeff5U2);                     \
            res2U2 = vmlaq_u16(res2U2, src6, coeff6U2);                     \
            res0U2 = vmlaq_u16(res0U2, src7, coeff7U2);                     \
            add0U4 = vaddl_u16(vget_low_u16(res1U2), vget_low_u16(res2U2)); \
            add1U4 = vaddl_u16(vget_high_u16(res1U2), vget_high_u16(res2U2));\
            sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),               \
                                vreinterpret_s16_u16(vget_low_u16(res0U2)));\
            sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(\
                                            vreinterpretq_s16_u16(res0U2)));\
            sft0U2 = vqrshrun_n_s32(sub0U4, 6);                             \
            sft1U2 = vqrshrun_n_s32(sub1U4, 6);                             \
            sol0U2 = vcombine_u16(sft0U2, sft1U2);                          \
            sol1U2 = vminq_u16(sol0U2, max_10bit);                          \
            vst1q_u16(out, sol1U2);

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, src4,                    \
                                             src5, src6, src7, out)         \
            mul0S4 = vmull_s16(src1, coeff1S2);                             \
            mul0S4 = vmlal_s16(mul0S4, src0, coeff0S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src3, coeff3S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src2, coeff2S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src4, coeff4S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src5, coeff5S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src6, coeff6S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src7, coeff7S2);                     \
            mul0U2 = vqrshrun_n_s32(mul0S4, 6);                             \
            mul0U2 = vmin_u16(mul0U2, max_10bit);                           \
            vst1_u16(out, mul0U2);

    if (!(wd & 7))
    {
        uint16x8_t max_10bit = vdupq_n_u16((1 << bit_depth) - 1);
        uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2;
        uint16x8_t coeff4U2, coeff5U2, coeff6U2, coeff7U2;
        coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
        coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
        coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
        coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);
        coeff4U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 0);
        coeff5U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 1);
        coeff6U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 2);
        coeff7U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 3);

        if (!(wd & 15))
        {
            for (y = 0; y < ht; y += 4)
            {
                for (x = 0; x < wd; x += 16)
                {
                    src0U2 = vld1q_u16(pu2_src + x - src_strd3);
                    src1U2 = vld1q_u16(pu2_src + x - src_strd2);
                    src2U2 = vld1q_u16(pu2_src + x - src_strd);
                    src3U2 = vld1q_u16(pu2_src + x);
                    src4U2 = vld1q_u16(pu2_src + x + src_strd);
                    src5U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src6U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src7U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd);
                    src9U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd2);
                    src10U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd3);
                    CALCULATE_ROW_WD8x(src0U2, src1U2, src2U2, src3U2,
                                       src4U2, src5U2, src6U2, src7U2, pu2_dst + x)
                    CALCULATE_ROW_WD8x(src1U2, src2U2, src3U2, src4U2,
                                       src5U2, src6U2, src7U2, src8U2,
                                       pu2_dst + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2U2, src3U2, src4U2, src5U2,
                                       src6U2, src7U2, src8U2, src9U2,
                                       pu2_dst + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3U2, src4U2, src5U2, src6U2,
                                         src7U2, src8U2, src9U2, src10U2,
                                         pu2_dst + x + dst_strd3)

                    src0U2 = vld1q_u16(pu2_src + x + 8 - src_strd3);
                    src1U2 = vld1q_u16(pu2_src + x + 8 - src_strd2);
                    src2U2 = vld1q_u16(pu2_src + x + 8 - src_strd);
                    src3U2 = vld1q_u16(pu2_src + x + 8);
                    src4U2 = vld1q_u16(pu2_src + x + 8 + src_strd);
                    src5U2 = vld1q_u16(pu2_src + x + 8 + src_strd2);
                    src6U2 = vld1q_u16(pu2_src + x + 8 + src_strd3);
                    src7U2 = vld1q_u16(pu2_src + x + 8 + src_strd4);
                    src8U2 = vld1q_u16(pu2_src + x + 8 + src_strd4 + src_strd);
                    src9U2 = vld1q_u16(pu2_src + x + 8 + src_strd4 + src_strd2);
                    src10U2 = vld1q_u16(pu2_src + x + 8 + src_strd4 + src_strd3);

                    CALCULATE_ROW_WD8x(src0U2, src1U2, src2U2, src3U2,
                                       src4U2, src5U2, src6U2, src7U2,
                                       pu2_dst + 8 + x)
                    CALCULATE_ROW_WD8x(src1U2, src2U2, src3U2, src4U2,
                                       src5U2, src6U2, src7U2, src8U2,
                                       pu2_dst + 8 + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2U2, src3U2, src4U2, src5U2,
                                       src6U2, src7U2, src8U2, src9U2,
                                       pu2_dst + 8 + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3U2, src4U2, src5U2, src6U2,
                                         src7U2, src8U2, src9U2, src10U2,
                                         pu2_dst + 8 + x + dst_strd3)
                }
                pu2_src += src_strd4;
                pu2_dst += dst_strd4;
            }
        }
        else
        {
            for (y = 0; y < ht; y += 4)
            {
                for (x = 0; x < wd; x += 8)
                {
                    src0U2 = vld1q_u16(pu2_src + x - src_strd3);
                    src1U2 = vld1q_u16(pu2_src + x - src_strd2);
                    src2U2 = vld1q_u16(pu2_src + x - src_strd);
                    src3U2 = vld1q_u16(pu2_src + x);
                    src4U2 = vld1q_u16(pu2_src + x + src_strd);
                    src5U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src6U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src7U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd);
                    src9U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd2);
                    src10U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd3);

                    CALCULATE_ROW_WD8x(src0U2, src1U2, src2U2, src3U2,
                                       src4U2, src5U2, src6U2, src7U2, pu2_dst + x)
                    CALCULATE_ROW_WD8x(src1U2, src2U2, src3U2, src4U2,
                                       src5U2, src6U2, src7U2, src8U2,
                                       pu2_dst + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2U2, src3U2, src4U2, src5U2,
                                       src6U2, src7U2, src8U2, src9U2,
                                       pu2_dst + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3U2, src4U2, src5U2, src6U2,
                                         src7U2, src8U2, src9U2, src10U2,
                                         pu2_dst + x + dst_strd3)
                }
                pu2_src += src_strd4;
                pu2_dst += dst_strd4;
            }
        }
    }
    else if (!(wd & 3))
    {
        int16x4_t src0S2, src1S2, src2S2, src3S2, src4S2, src5S2;
        int16x4_t src6S2, src7S2, src8S2, src9S2, src10S2;
        int32x4_t mul0S4;
        uint16x4_t mul0U2;

        int16x8_t coeffS2 = vmovl_s8(vld1_s8(pi1_coeff));

        int16x4_t coeff0S2 = vdup_lane_s16(vget_low_s16(coeffS2), 0);
        int16x4_t coeff1S2 = vdup_lane_s16(vget_low_s16(coeffS2), 1);
        int16x4_t coeff2S2 = vdup_lane_s16(vget_low_s16(coeffS2), 2);
        int16x4_t coeff3S2 = vdup_lane_s16(vget_low_s16(coeffS2), 3);
        int16x4_t coeff4S2 = vdup_lane_s16(vget_high_s16(coeffS2), 0);
        int16x4_t coeff5S2 = vdup_lane_s16(vget_high_s16(coeffS2), 1);
        int16x4_t coeff6S2 = vdup_lane_s16(vget_high_s16(coeffS2), 2);
        int16x4_t coeff7S2 = vdup_lane_s16(vget_high_s16(coeffS2), 3);

        uint16x4_t max_10bit = vdup_n_u16((1 << bit_depth) - 1);

        for (y = 0; y < ht; y += 4)
        {
            for (x = 0; x < wd; x += 4)
            {
                src0S2 = vld1_s16((const int16_t*)(pu2_src + x - src_strd3));
                src1S2 = vld1_s16((const int16_t*)(pu2_src + x - src_strd2));
                src2S2 = vld1_s16((const int16_t*)(pu2_src + x - src_strd));
                src3S2 = vld1_s16((const int16_t*)(pu2_src + x + 0));
                src4S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd));
                src5S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd2));
                src6S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd3));
                src7S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4));
                src8S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4 + src_strd));
                src9S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4 + src_strd2));
                src10S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4 + src_strd3));
                CALCULATE_ROW_WD4x(src0S2, src1S2, src2S2, src3S2,
                                   src4S2, src5S2, src6S2, src7S2, pu2_dst + x)
                CALCULATE_ROW_WD4x(src1S2, src2S2, src3S2, src4S2,
                                   src5S2, src6S2, src7S2, src8S2,
                                   pu2_dst + x + dst_strd)
                CALCULATE_ROW_WD4x(src2S2, src3S2, src4S2, src5S2,
                                   src6S2, src7S2, src8S2, src9S2,
                                   pu2_dst + x + dst_strd2)
                CALCULATE_ROW_WD4x(src3S2, src4S2, src5S2, src6S2,
                                     src7S2, src8S2, src9S2, src10S2,
                                     pu2_dst + x + dst_strd3)
            }
            pu2_src += src_strd4;
            pu2_dst += dst_strd4;
        }
    }
#undef CALCULATE_ROW_WD8x
#undef CALCULATE_ROW_WD4x
}

/**
*******************************************************************************
*
* @brief
*   Interprediction luma function for copy with 16-bit output
*
* @par Description:
*   Copies the luma array of width 'wd' and height 'ht' from the location
*   pointed by 'pu2_src' to the location pointed by 'pi2_dst' with left-shift
*   by (14 - bit_depth).
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients (unused)
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_copy_w16out_neonintr(UWORD16 *pu2_src,
                                                    WORD16 *pi2_dst,
                                                    WORD32 src_strd,
                                                    WORD32 dst_strd,
                                                    WORD8 *pi1_coeff,
                                                    WORD32 ht,
                                                    WORD32 wd,
                                                    UWORD8 bit_depth)
{
    WORD32 row, col;
    uint16x8_t src0_16x8, src1_16x8, src2_16x8, src3_16x8;
    uint16x4_t src0_16x4, src1_16x4, src2_16x4, src3_16x4;
    WORD32 src_strd2 = src_strd * 2;
    WORD32 src_strd3 = src_strd * 3;
    WORD32 dst_strd2 = dst_strd * 2;
    WORD32 dst_strd3 = dst_strd * 3;

    UNUSED(pi1_coeff);

    if (0 == (wd & 7)) /* multiple of 8 case */
    {
        int16x8_t shift_vec = vdupq_n_s16(14 - bit_depth);

        for (row = 0; row < ht; row += 4)
        {
            for (col = 0; col < wd; col += 8)
            {
                /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                src0_16x8 = vld1q_u16(pu2_src);
                src1_16x8 = vld1q_u16(pu2_src + src_strd);
                src2_16x8 = vld1q_u16(pu2_src + src_strd2);
                src3_16x8 = vld1q_u16(pu2_src + src_strd3);

                src0_16x8 = vshlq_u16(src0_16x8, shift_vec);
                src1_16x8 = vshlq_u16(src1_16x8, shift_vec);
                src2_16x8 = vshlq_u16(src2_16x8, shift_vec);
                src3_16x8 = vshlq_u16(src3_16x8, shift_vec);

                /* storing 8 16-bit output values */
                vst1q_s16(pi2_dst, vreinterpretq_s16_u16(src0_16x8));
                vst1q_s16(pi2_dst + dst_strd, vreinterpretq_s16_u16(src1_16x8));
                vst1q_s16(pi2_dst + dst_strd2, vreinterpretq_s16_u16(src2_16x8));
                vst1q_s16(pi2_dst + dst_strd3, vreinterpretq_s16_u16(src3_16x8));

                pu2_src += 8; /* pointer update */
                pi2_dst += 8; /* pointer update */
            } /* inner for loop ends here(8-output values in single iteration) */

            pu2_src += 4 * src_strd - wd; /* pointer update */
            pi2_dst += 4 * dst_strd - wd; /* pointer update */
        }
    }
    else if (0 == (wd & 3)) /* wd = multiple of 4 case */
    {
        int16x4_t shift_vec4 = vdup_n_s16(14 - bit_depth);

        for (row = 0; row < ht; row += 4)
        {
            for (col = 0; col < wd; col += 4)
            {
                /* load 4 pixel values from 3:0 pos. relative to cur. pos. */
                src0_16x4 = vld1_u16(pu2_src);
                src1_16x4 = vld1_u16(pu2_src + src_strd);
                src2_16x4 = vld1_u16(pu2_src + src_strd2);
                src3_16x4 = vld1_u16(pu2_src + src_strd3);

                src0_16x4 = vshl_u16(src0_16x4, shift_vec4);
                src1_16x4 = vshl_u16(src1_16x4, shift_vec4);
                src2_16x4 = vshl_u16(src2_16x4, shift_vec4);
                src3_16x4 = vshl_u16(src3_16x4, shift_vec4);

                /* storing 4 16-bit output values */
                vst1_s16(pi2_dst, vreinterpret_s16_u16(src0_16x4));
                vst1_s16(pi2_dst + dst_strd, vreinterpret_s16_u16(src1_16x4));
                vst1_s16(pi2_dst + dst_strd2, vreinterpret_s16_u16(src2_16x4));
                vst1_s16(pi2_dst + dst_strd3, vreinterpret_s16_u16(src3_16x4));

                pu2_src += 4; /* pointer update */
                pi2_dst += 4; /* pointer update */
            } /* inner for loop ends here(4-output values in single iteration) */

            pu2_src += 4 * src_strd - wd; /* pointer update */
            pi2_dst += 4 * dst_strd - wd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     Interprediction luma filter for horizontal 16bit output
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and writes to the location pointed
*    by 'pi2_dst'. The output is downshifted by (bit_depth - 8) and is used
*    as an input for vertical filtering or weighted prediction.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/


void ihevc_hbd_inter_pred_luma_horz_w16out_neonintr(UWORD16 *pu2_src,
                                                    WORD16 *pi2_dst,
                                                    WORD32  src_strd,
                                                    WORD32  dst_strd,
                                                    WORD8 *pi1_coeff,
                                                    WORD32  ht,
                                                    WORD32  wd,
                                                    UWORD8  bit_depth)
{
    int32x4_t shift_vec = vdupq_n_s32(8 - bit_depth);
    int8x8_t coeffS1 = vld1_s8(pi1_coeff);
    pu2_src -= 3;
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 x, y;

    uint16x8_t aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2, aSrc4U2, aSrc5U2, aSrc6U2, aSrc7U2;
    uint16x8_t bSrc0U2, bSrc1U2, bSrc2U2, bSrc3U2, bSrc4U2, bSrc5U2, bSrc6U2, bSrc7U2;
    uint16x8_t mul0U2, mul1U2, mul2U2, mul3U2, mul4U2, mul5U2;
    int16x4_t  sft0S2, sft1S2, sft2S2, sft3S2;
    uint32x4_t add0U4, add1U4, add2U4, add3U4;
    int32x4_t  sub0U4, sub1U4, sub2U4, sub3U4, mul0S4;
    int16x4_t  src0S2, src1S2, src2S2, src3S2, src4S2, src5S2, src6S2, src7S2;
    int16x4_t  coeff0S2, coeff1S2, coeff2S2, coeff3S2, coeff4S2, coeff5S2, coeff6S2, coeff7S2;
    uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2, coeff4U2, coeff5U2, coeff6U2, coeff7U2;

#define CALCULATE_2ROWS_WD8x(pSrc, pDst)                                    \
        aSrc0U2 = vld1q_u16(pSrc);                                          \
        aSrc1U2 = vld1q_u16(pSrc + 1);                                      \
        bSrc0U2 = vld1q_u16(pSrc + src_strd);                               \
        bSrc1U2 = vld1q_u16(pSrc + src_strd + 1);                           \
        aSrc2U2 = vld1q_u16(pSrc + 2);                                      \
        aSrc3U2 = vld1q_u16(pSrc + 3);                                      \
        bSrc2U2 = vld1q_u16(pSrc + src_strd + 2);                           \
        bSrc3U2 = vld1q_u16(pSrc + src_strd + 3);                           \
        mul0U2 = vmulq_u16(aSrc0U2, coeff0U2);                              \
        mul1U2 = vmulq_u16(aSrc1U2, coeff1U2);                              \
        mul3U2 = vmulq_u16(bSrc0U2, coeff0U2);                              \
        mul4U2 = vmulq_u16(bSrc1U2, coeff1U2);                              \
        mul0U2 = vmlaq_u16(mul0U2, aSrc2U2, coeff2U2);                      \
        mul1U2 = vmlaq_u16(mul1U2, aSrc3U2, coeff3U2);                      \
        mul3U2 = vmlaq_u16(mul3U2, bSrc2U2, coeff2U2);                      \
        mul4U2 = vmlaq_u16(mul4U2, bSrc3U2, coeff3U2);                      \
        aSrc4U2 = vld1q_u16(pSrc + 4);                                      \
        aSrc5U2 = vld1q_u16(pSrc + 5);                                      \
        bSrc4U2 = vld1q_u16(pSrc + src_strd + 4);                           \
        bSrc5U2 = vld1q_u16(pSrc + src_strd + 5);                           \
        aSrc6U2 = vld1q_u16(pSrc + 6);                                      \
        aSrc7U2 = vld1q_u16(pSrc + 7);                                      \
        bSrc6U2 = vld1q_u16(pSrc + src_strd + 6);                           \
        bSrc7U2 = vld1q_u16(pSrc + src_strd + 7);                           \
        mul2U2 = vmulq_u16(aSrc4U2, coeff4U2);                              \
        mul0U2 = vmlaq_u16(mul0U2, aSrc5U2, coeff5U2);                      \
        mul5U2 = vmulq_u16(bSrc4U2, coeff4U2);                              \
        mul3U2 = vmlaq_u16(mul3U2, bSrc5U2, coeff5U2);                      \
        mul2U2 = vmlaq_u16(mul2U2, aSrc6U2, coeff6U2);                      \
        mul0U2 = vmlaq_u16(mul0U2, aSrc7U2, coeff7U2);                      \
        mul5U2 = vmlaq_u16(mul5U2, bSrc6U2, coeff6U2);                      \
        mul3U2 = vmlaq_u16(mul3U2, bSrc7U2, coeff7U2);                      \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        add2U4 = vaddl_u16(vget_low_u16(mul4U2), vget_low_u16(mul5U2));     \
        add3U4 = vaddl_u16(vget_high_u16(mul4U2), vget_high_u16(mul5U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sub2U4 = vsubw_s16(vreinterpretq_s32_u32(add2U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul3U2)));    \
        sub3U4 = vsubw_s16(vreinterpretq_s32_u32(add3U4), vget_high_s16(    \
                                        vreinterpretq_s16_u16(mul3U2)));    \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        sft1S2 = vmovn_s32(vshlq_s32(sub1U4, shift_vec));                   \
        sft2S2 = vmovn_s32(vshlq_s32(sub2U4, shift_vec));                   \
        sft3S2 = vmovn_s32(vshlq_s32(sub3U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);                                             \
        vst1_s16(pDst + 4, sft1S2);                                         \
        vst1_s16(pDst + dst_strd, sft2S2);                                  \
        vst1_s16(pDst + dst_strd + 4, sft3S2);

#define CALCULATE_ROW_WD8x(pSrc, pDst)                                      \
        aSrc0U2 = vld1q_u16(pSrc);                                          \
        aSrc1U2 = vld1q_u16(pSrc + 1);                                      \
        aSrc2U2 = vld1q_u16(pSrc + 2);                                      \
        aSrc3U2 = vld1q_u16(pSrc + 3);                                      \
        mul0U2 = vmulq_u16(aSrc0U2, coeff0U2);                              \
        mul1U2 = vmulq_u16(aSrc1U2, coeff1U2);                              \
        mul0U2 = vmlaq_u16(mul0U2, aSrc2U2, coeff2U2);                      \
        mul1U2 = vmlaq_u16(mul1U2, aSrc3U2, coeff3U2);                      \
        aSrc4U2 = vld1q_u16(pSrc + 4);                                      \
        aSrc5U2 = vld1q_u16(pSrc + 5);                                      \
        aSrc6U2 = vld1q_u16(pSrc + 6);                                      \
        aSrc7U2 = vld1q_u16(pSrc + 7);                                      \
        mul2U2 = vmulq_u16(aSrc4U2, coeff4U2);                              \
        mul0U2 = vmlaq_u16(mul0U2, aSrc5U2, coeff5U2);                      \
        mul2U2 = vmlaq_u16(mul2U2, aSrc6U2, coeff6U2);                      \
        mul0U2 = vmlaq_u16(mul0U2, aSrc7U2, coeff7U2);                      \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        sft1S2 = vmovn_s32(vshlq_s32(sub1U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);                                             \
        vst1_s16(pDst + 4, sft1S2);

#define CALCULATE_ROW_WD4x(pSrc, pDst)                                      \
        src0S2 = vld1_s16((const int16_t*)(pSrc));                          \
        src1S2 = vld1_s16((const int16_t*)(pSrc + 1));                      \
        src2S2 = vld1_s16((const int16_t*)(pSrc + 2));                      \
        src3S2 = vld1_s16((const int16_t*)(pSrc + 3));                      \
        src4S2 = vld1_s16((const int16_t*)(pSrc + 4));                      \
        src5S2 = vld1_s16((const int16_t*)(pSrc + 5));                      \
        src6S2 = vld1_s16((const int16_t*)(pSrc + 6));                      \
        src7S2 = vld1_s16((const int16_t*)(pSrc + 7));                      \
        mul0S4 = vmull_s16(src1S2, coeff1S2);                               \
        mul0S4 = vmlal_s16(mul0S4, src0S2, coeff0S2);                       \
        mul0S4 = vmlal_s16(mul0S4, src3S2, coeff3S2);                       \
        mul0S4 = vmlal_s16(mul0S4, src2S2, coeff2S2);                       \
        mul0S4 = vmlal_s16(mul0S4, src4S2, coeff4S2);                       \
        mul0S4 = vmlal_s16(mul0S4, src5S2, coeff5S2);                       \
        mul0S4 = vmlal_s16(mul0S4, src6S2, coeff6S2);                       \
        mul0S4 = vmlal_s16(mul0S4, src7S2, coeff7S2);                       \
        sft0S2 = vmovn_s32(vshlq_s32(mul0S4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);

    if (!(wd & 7)) {
        uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(vabs_s8(coeffS1)));
        coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
        coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
        coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
        coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);
        coeff4U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 0);
        coeff5U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 1);
        coeff6U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 2);
        coeff7U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 3);

        if (!(wd & 31)) {
            for (y = 0; y < (ht & ~1); y += 2) {
                for (x = 0; x < wd; x += 32) {
                    CALCULATE_2ROWS_WD8x(pu2_src + x, pi2_dst + x)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 8, pi2_dst + x + 8)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 16, pi2_dst + x + 16)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 24, pi2_dst + x + 24)
                }
                pu2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        } else if (!(wd & 15)) {
            for (y = 0; y < (ht & ~1); y += 2) {
                for (x = 0; x < wd; x += 16) {
                    CALCULATE_2ROWS_WD8x(pu2_src + x, pi2_dst + x)
                    CALCULATE_2ROWS_WD8x(pu2_src + x + 8, pi2_dst + x + 8)
                }
                pu2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        } else {
            for (y = 0; y < (ht & ~1); y += 2) {
                for (x = 0; x < wd; x += 8) {
                    CALCULATE_2ROWS_WD8x(pu2_src + x, pi2_dst + x)
                }
                pu2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        }

        if (ht & 1) {
            for (x = 0; x < wd; x += 8) {
                CALCULATE_ROW_WD8x(pu2_src + x, pi2_dst + x)
            }
        }
    } else if (!(wd & 3)) {
        int16x8_t coeffS2 = vmovl_s8(coeffS1);
        coeff0S2 = vdup_lane_s16(vget_low_s16(coeffS2), 0);
        coeff1S2 = vdup_lane_s16(vget_low_s16(coeffS2), 1);
        coeff2S2 = vdup_lane_s16(vget_low_s16(coeffS2), 2);
        coeff3S2 = vdup_lane_s16(vget_low_s16(coeffS2), 3);
        coeff4S2 = vdup_lane_s16(vget_high_s16(coeffS2), 0);
        coeff5S2 = vdup_lane_s16(vget_high_s16(coeffS2), 1);
        coeff6S2 = vdup_lane_s16(vget_high_s16(coeffS2), 2);
        coeff7S2 = vdup_lane_s16(vget_high_s16(coeffS2), 3);

        for (y = 0; y < (ht & ~1); y += 2) {
            for (x = 0; x < wd; x += 4) {
                CALCULATE_ROW_WD4x(pu2_src + x, pi2_dst + x)
                CALCULATE_ROW_WD4x(pu2_src + x + src_strd, pi2_dst + x + dst_strd)
            }
            pu2_src += src_strd2;
            pi2_dst += dst_strd2;
        }

        if (ht & 1) {
            for (x = 0; x < wd; x += 4) {
                CALCULATE_ROW_WD4x(pu2_src + x, pi2_dst + x)
            }
        }
    }
#undef CALCULATE_2ROWS_WD8x
#undef CALCULATE_ROW_WD8x
#undef CALCULATE_ROW_WD4x
}

/**
*******************************************************************************
*
* @brief
*      Interprediction luma filter for vertical 16bit output
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*    the elements pointed by 'pu2_src' and writes to the location pointed by
*    'pi2_dst'. The output is downshifted by (bit_depth - 8) and is used as
*    an input for weighted prediction.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_vert_w16out_neonintr(UWORD16 *pu2_src,
                                                    WORD16 *pi2_dst,
                                                    WORD32 src_strd,
                                                    WORD32 dst_strd,
                                                    WORD8 *pi1_coeff,
                                                    WORD32 ht,
                                                    WORD32 wd,
                                                    UWORD8 bit_depth)
{
    int32x4_t shift_vec = vdupq_n_s32(8 - bit_depth);
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = (src_strd << 2);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 dst_strd3 = dst_strd2 + dst_strd;
    WORD32 dst_strd4 = (dst_strd << 2);
    WORD32 x, y;

    int8x8_t  coeffS1 = vabs_s8(vld1_s8(pi1_coeff));
    uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(coeffS1));

    uint16x8_t src0U2, src1U2, src2U2, src3U2, src4U2, src5U2;
    uint16x8_t src6U2, src7U2, src8U2, src9U2, src10U2;
    uint16x8_t res0U2, res1U2, res2U2;
    int16x4_t  sft0S2, sft1S2;
    uint32x4_t add0U4, add1U4;
    int32x4_t  sub0S4, sub1S4;

#define CALCULATE_ROW_WD8x(src0, src1, src2, src3, src4,                    \
                                             src5, src6, src7, out)         \
            res0U2 = vmulq_u16(src0, coeff0U2);                             \
            res1U2 = vmulq_u16(src1, coeff1U2);                             \
            res0U2 = vmlaq_u16(res0U2, src2, coeff2U2);                     \
            res1U2 = vmlaq_u16(res1U2, src3, coeff3U2);                     \
            res2U2 = vmulq_u16(src4, coeff4U2);                             \
            res0U2 = vmlaq_u16(res0U2, src5, coeff5U2);                     \
            res2U2 = vmlaq_u16(res2U2, src6, coeff6U2);                     \
            res0U2 = vmlaq_u16(res0U2, src7, coeff7U2);                     \
            add0U4 = vaddl_u16(vget_low_u16(res1U2), vget_low_u16(res2U2)); \
            add1U4 = vaddl_u16(vget_high_u16(res1U2), vget_high_u16(res2U2));\
            sub0S4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),               \
                                vreinterpret_s16_u16(vget_low_u16(res0U2)));\
            sub1S4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(\
                                            vreinterpretq_s16_u16(res0U2)));\
            sft0S2 = vmovn_s32(vshlq_s32(sub0S4, shift_vec));               \
            sft1S2 = vmovn_s32(vshlq_s32(sub1S4, shift_vec));               \
            vst1_s16(out, sft0S2);  \
            vst1_s16(out + 4, sft1S2);

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, src4,                    \
                                              src5, src6, src7, out)        \
            mul0S4 = vmull_s16(src1, coeff1S2);                             \
            mul0S4 = vmlal_s16(mul0S4, src0, coeff0S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src3, coeff3S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src2, coeff2S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src4, coeff4S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src5, coeff5S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src6, coeff6S2);                     \
            mul0S4 = vmlal_s16(mul0S4, src7, coeff7S2);                     \
            sft0S2 = vmovn_s32(vshlq_s32(mul0S4, shift_vec));               \
            vst1_s16(out, sft0S2);

    if (!(wd & 7))
    {
        uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2;
        uint16x8_t coeff4U2, coeff5U2, coeff6U2, coeff7U2;
        coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
        coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
        coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
        coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);
        coeff4U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 0);
        coeff5U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 1);
        coeff6U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 2);
        coeff7U2 = vdupq_lane_u16(vget_high_u16(coeffU2), 3);

        if (!(wd & 15))
        {
            for (y = 0; y < ht; y += 4)
            {
                for (x = 0; x < wd; x += 16)
                {
                    src0U2 = vld1q_u16(pu2_src + x - src_strd3);
                    src1U2 = vld1q_u16(pu2_src + x - src_strd2);
                    src2U2 = vld1q_u16(pu2_src + x - src_strd);
                    src3U2 = vld1q_u16(pu2_src + x);
                    src4U2 = vld1q_u16(pu2_src + x + src_strd);
                    src5U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src6U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src7U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd);
                    src9U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd2);
                    src10U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd3);
                    CALCULATE_ROW_WD8x(src0U2, src1U2, src2U2, src3U2,
                                       src4U2, src5U2, src6U2, src7U2, pi2_dst + x)
                    CALCULATE_ROW_WD8x(src1U2, src2U2, src3U2, src4U2,
                                       src5U2, src6U2, src7U2, src8U2,
                                       pi2_dst + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2U2, src3U2, src4U2, src5U2,
                                       src6U2, src7U2, src8U2, src9U2,
                                       pi2_dst + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3U2, src4U2, src5U2, src6U2,
                                         src7U2, src8U2, src9U2, src10U2,
                                         pi2_dst + x + dst_strd3)

                    src0U2 = vld1q_u16(pu2_src + x + 8 - src_strd3);
                    src1U2 = vld1q_u16(pu2_src + x + 8 - src_strd2);
                    src2U2 = vld1q_u16(pu2_src + x + 8 - src_strd);
                    src3U2 = vld1q_u16(pu2_src + x + 8);
                    src4U2 = vld1q_u16(pu2_src + x + 8 + src_strd);
                    src5U2 = vld1q_u16(pu2_src + x + 8 + src_strd2);
                    src6U2 = vld1q_u16(pu2_src + x + 8 + src_strd3);
                    src7U2 = vld1q_u16(pu2_src + x + 8 + src_strd4);
                    src8U2 = vld1q_u16(pu2_src + x + 8 + src_strd4 + src_strd);
                    src9U2 = vld1q_u16(pu2_src + x + 8 + src_strd4 + src_strd2);
                    src10U2 = vld1q_u16(pu2_src + x + 8 + src_strd4 + src_strd3);

                    CALCULATE_ROW_WD8x(src0U2, src1U2, src2U2, src3U2,
                                       src4U2, src5U2, src6U2, src7U2,
                                       pi2_dst + 8 + x)
                    CALCULATE_ROW_WD8x(src1U2, src2U2, src3U2, src4U2,
                                       src5U2, src6U2, src7U2, src8U2,
                                       pi2_dst + 8 + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2U2, src3U2, src4U2, src5U2,
                                       src6U2, src7U2, src8U2, src9U2,
                                       pi2_dst + 8 + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3U2, src4U2, src5U2, src6U2,
                                         src7U2, src8U2, src9U2, src10U2,
                                         pi2_dst + 8 + x + dst_strd3)
                }
                pu2_src += src_strd4;
                pi2_dst += dst_strd4;
            }
        }
        else
        {
            for (y = 0; y < ht; y += 4) {
                for (x = 0; x < wd; x += 8) {
                    src0U2 = vld1q_u16(pu2_src + x - src_strd3);
                    src1U2 = vld1q_u16(pu2_src + x - src_strd2);
                    src2U2 = vld1q_u16(pu2_src + x - src_strd);
                    src3U2 = vld1q_u16(pu2_src + x);
                    src4U2 = vld1q_u16(pu2_src + x + src_strd);
                    src5U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src6U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src7U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd);
                    src9U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd2);
                    src10U2 = vld1q_u16(pu2_src + x + src_strd4 + src_strd3);

                    CALCULATE_ROW_WD8x(src0U2, src1U2, src2U2, src3U2,
                                       src4U2, src5U2, src6U2, src7U2, pi2_dst + x)
                    CALCULATE_ROW_WD8x(src1U2, src2U2, src3U2, src4U2,
                                       src5U2, src6U2, src7U2, src8U2,
                                       pi2_dst + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2U2, src3U2, src4U2, src5U2,
                                       src6U2, src7U2, src8U2, src9U2,
                                       pi2_dst + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3U2, src4U2, src5U2, src6U2,
                                         src7U2, src8U2, src9U2, src10U2,
                                         pi2_dst + x + dst_strd3)
                }
                pu2_src += src_strd4;
                pi2_dst += dst_strd4;
            }
        }
    }
    else if (!(wd & 3)) {
        int16x4_t src0S2, src1S2, src2S2, src3S2, src4S2, src5S2;
        int16x4_t src6S2, src7S2, src8S2, src9S2, src10S2;
        int32x4_t mul0S4;

        int16x4_t coeff0S2, coeff1S2, coeff2S2, coeff3S2;
        int16x4_t coeff4S2, coeff5S2, coeff6S2, coeff7S2;
        int16x8_t coeffS2 = vmovl_s8(vld1_s8(pi1_coeff));

        coeff0S2 = vdup_lane_s16(vget_low_s16(coeffS2), 0);
        coeff1S2 = vdup_lane_s16(vget_low_s16(coeffS2), 1);
        coeff2S2 = vdup_lane_s16(vget_low_s16(coeffS2), 2);
        coeff3S2 = vdup_lane_s16(vget_low_s16(coeffS2), 3);
        coeff4S2 = vdup_lane_s16(vget_high_s16(coeffS2), 0);
        coeff5S2 = vdup_lane_s16(vget_high_s16(coeffS2), 1);
        coeff6S2 = vdup_lane_s16(vget_high_s16(coeffS2), 2);
        coeff7S2 = vdup_lane_s16(vget_high_s16(coeffS2), 3);

        for (y = 0; y < ht; y += 4) {
            for (x = 0; x < wd; x += 4) {
                src0S2 = vld1_s16((const int16_t*)(pu2_src + x - src_strd3));
                src1S2 = vld1_s16((const int16_t*)(pu2_src + x - src_strd2));
                src2S2 = vld1_s16((const int16_t*)(pu2_src + x - src_strd));
                src3S2 = vld1_s16((const int16_t*)(pu2_src + x + 0));
                src4S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd));
                src5S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd2));
                src6S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd3));
                src7S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4));
                src8S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4 + src_strd));
                src9S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4 + src_strd2));
                src10S2 = vld1_s16((const int16_t*)(pu2_src + x + src_strd4 + src_strd3));
                CALCULATE_ROW_WD4x(src0S2, src1S2, src2S2, src3S2,
                                   src4S2, src5S2, src6S2, src7S2, pi2_dst + x)
                CALCULATE_ROW_WD4x(src1S2, src2S2, src3S2, src4S2,
                                   src5S2, src6S2, src7S2, src8S2,
                                   pi2_dst + x + dst_strd)
                CALCULATE_ROW_WD4x(src2S2, src3S2, src4S2, src5S2,
                                   src6S2, src7S2, src8S2, src9S2,
                                   pi2_dst + x + dst_strd2)
                CALCULATE_ROW_WD4x(src3S2, src4S2, src5S2, src6S2,
                                     src7S2, src8S2, src9S2, src10S2,
                                     pi2_dst + x + dst_strd3)
            }
            pu2_src += src_strd4;
            pi2_dst += dst_strd4;
        }
    }
#undef CALCULATE_ROW_WD8x
#undef CALCULATE_ROW_WD4x
}

/**
*******************************************************************************
*
* @brief
*
*        Luma vertical filter for 16bit input.
*
* @par Description:
*   Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*   the elements pointed by 'pi2_src' and writes to the location pointed by
*   'pu2_dst'. Input is 16 bits. The filter output is downshifted by
*   (14 - bit_depth), rounded by 6, and clipped to bit_depth.
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_vert_w16inp_neonintr(WORD16 *pi2_src,
                                                    UWORD16 *pu2_dst,
                                                    WORD32 src_strd,
                                                    WORD32 dst_strd,
                                                    WORD8 *pi1_coeff,
                                                    WORD32 ht,
                                                    WORD32 wd,
                                                    UWORD8 bit_depth)
{
    int32x4_t shift_vec = vdupq_n_s32(bit_depth - 14);
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = (src_strd << 2);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 dst_strd3 = dst_strd2 + dst_strd;
    WORD32 dst_strd4 = (dst_strd << 2);
    WORD32 x, y;

    uint16x4_t max_10bit = vdup_n_u16((1 << bit_depth) - 1);
    int8x8_t  coeffS1 = vld1_s8(pi1_coeff);
    int16x8_t coeffS2 = vmovl_s8(coeffS1);

    uint16x4_t sft2U2, sft3U2, sol0U2, sol1U2;
    int16x8_t src0S2, src1S2, src2S2, src3S2, src4S2, src5S2;
    int16x8_t src6S2, src7S2, src8S2, src9S2, src10S2;
    int32x4_t res0S4, res1S4, mul0S4, mul1S4;
    int32x4_t sft0S4, sft1S4, sub0S4, sub1S4;

#define CALCULATE_ROW_WD8x(src0, src1, src2, src3, src4,                                  \
                                             src5, src6, src7, out)                       \
        res0S4 = vmull_lane_s16(vget_low_s16(src0), vget_low_s16(coeffS2), 0);            \
        mul0S4 = vmull_s16(vget_high_s16(src0), vget_high_s16(coeff0S2));                 \
        res1S4 = vmull_lane_s16(vget_low_s16(src1), vget_low_s16(coeffS2), 1);            \
        mul1S4 = vmull_s16(vget_high_s16(src1), vget_high_s16(coeff1S2));                 \
        res0S4 = vmlal_lane_s16(res0S4, vget_low_s16(src2), vget_low_s16(coeffS2), 2);    \
        mul0S4 = vmlal_s16(mul0S4, vget_high_s16(src2), vget_high_s16(coeff2S2));         \
        res1S4 = vmlal_lane_s16(res1S4, vget_low_s16(src3), vget_low_s16(coeffS2), 3);    \
        mul1S4 = vmlal_s16(mul1S4, vget_high_s16(src3), vget_high_s16(coeff3S2));         \
        res1S4 = vmlal_lane_s16(res1S4, vget_low_s16(src4), vget_high_s16(coeffS2), 0);   \
        mul1S4 = vmlal_s16(mul1S4, vget_high_s16(src4), vget_high_s16(coeff4S2));         \
        res0S4 = vmlal_lane_s16(res0S4, vget_low_s16(src5), vget_high_s16(coeffS2), 1);   \
        mul0S4 = vmlal_s16(mul0S4, vget_high_s16(src5), vget_high_s16(coeff5S2));         \
        res1S4 = vmlal_lane_s16(res1S4, vget_low_s16(src6), vget_high_s16(coeffS2), 2);   \
        mul1S4 = vmlal_s16(mul1S4, vget_high_s16(src6), vget_high_s16(coeff6S2));         \
        res0S4 = vmlal_lane_s16(res0S4, vget_low_s16(src7), vget_high_s16(coeffS2), 3);   \
        mul0S4 = vmlal_s16(mul0S4, vget_high_s16(src7), vget_high_s16(coeff7S2));         \
        sub0S4 = vaddq_s32(res0S4, res1S4);                                               \
        sub1S4 = vaddq_s32(mul0S4, mul1S4);                                               \
        sft0S4 = vshlq_s32(sub0S4, shift_vec);                                            \
        sft1S4 = vshlq_s32(sub1S4, shift_vec);                                            \
        sft2U2 = vqrshrun_n_s32(sft0S4, 6);                                               \
        sft3U2 = vqrshrun_n_s32(sft1S4, 6);                                               \
        sol0U2 = vmin_u16(sft2U2, max_10bit);                                             \
        sol1U2 = vmin_u16(sft3U2, max_10bit);                                             \
        vst1_u16(out, sol0U2);                                                            \
        vst1_u16(out + 4, sol1U2);

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, src4,                                  \
                                              src5, src6, src7, out)                      \
            mul0S4 = vmull_s16(src1, coeff1S2);                                           \
            mul0S4 = vmlal_s16(mul0S4, src0, coeff0S2);                                   \
            mul0S4 = vmlal_s16(mul0S4, src3, coeff3S2);                                   \
            mul0S4 = vmlal_s16(mul0S4, src2, coeff2S2);                                   \
            mul0S4 = vmlal_s16(mul0S4, src4, coeff4S2);                                   \
            mul0S4 = vmlal_s16(mul0S4, src5, coeff5S2);                                   \
            mul0S4 = vmlal_s16(mul0S4, src6, coeff6S2);                                   \
            mul0S4 = vmlal_s16(mul0S4, src7, coeff7S2);                                   \
            sft0S4 = vshlq_s32(mul0S4, shift_vec);                                        \
            sft2U2 = vqrshrun_n_s32(sft0S4, 6);                                           \
            sol0U2 = vmin_u16(sft2U2, max_10bit);                                         \
            vst1_u16(out, sol0U2);

    if (!(wd & 7)) {
        int16x8_t coeff0S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 0);
        int16x8_t coeff1S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 1);
        int16x8_t coeff2S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 2);
        int16x8_t coeff3S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 3);
        int16x8_t coeff4S2 = vdupq_lane_s16(vget_high_s16(coeffS2), 0);
        int16x8_t coeff5S2 = vdupq_lane_s16(vget_high_s16(coeffS2), 1);
        int16x8_t coeff6S2 = vdupq_lane_s16(vget_high_s16(coeffS2), 2);
        int16x8_t coeff7S2 = vdupq_lane_s16(vget_high_s16(coeffS2), 3);

        if (!(wd & 15)) {
            for (y = 0; y < ht; y += 4) {
                for (x = 0; x < wd; x += 16) {
                    src0S2 = vld1q_s16(pi2_src + x - src_strd3);
                    src1S2 = vld1q_s16(pi2_src + x - src_strd2);
                    src2S2 = vld1q_s16(pi2_src + x - src_strd);
                    src3S2 = vld1q_s16(pi2_src + x);
                    src4S2 = vld1q_s16(pi2_src + x + src_strd);
                    src5S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src6S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src7S2 = vld1q_s16(pi2_src + x + src_strd4);
                    src8S2 = vld1q_s16(pi2_src + x + src_strd4 + src_strd);
                    src9S2 = vld1q_s16(pi2_src + x + src_strd4 + src_strd2);
                    src10S2 = vld1q_s16(pi2_src + x + src_strd4 + src_strd3);
                    CALCULATE_ROW_WD8x(src0S2, src1S2, src2S2, src3S2,
                                       src4S2, src5S2, src6S2, src7S2, pu2_dst + x)
                    CALCULATE_ROW_WD8x(src1S2, src2S2, src3S2, src4S2,
                                       src5S2, src6S2, src7S2, src8S2,
                                       pu2_dst + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2S2, src3S2, src4S2, src5S2,
                                       src6S2, src7S2, src8S2, src9S2,
                                       pu2_dst + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3S2, src4S2, src5S2, src6S2,
                                       src7S2, src8S2, src9S2, src10S2,
                                       pu2_dst + x + dst_strd3)

                    src0S2 = vld1q_s16(pi2_src + x + 8 - src_strd3);
                    src1S2 = vld1q_s16(pi2_src + x + 8 - src_strd2);
                    src2S2 = vld1q_s16(pi2_src + x + 8 - src_strd);
                    src3S2 = vld1q_s16(pi2_src + x + 8);
                    src4S2 = vld1q_s16(pi2_src + x + 8 + src_strd);
                    src5S2 = vld1q_s16(pi2_src + x + 8 + src_strd2);
                    src6S2 = vld1q_s16(pi2_src + x + 8 + src_strd3);
                    src7S2 = vld1q_s16(pi2_src + x + 8 + src_strd4);
                    src8S2 = vld1q_s16(pi2_src + x + 8 + src_strd4 + src_strd);
                    src9S2 = vld1q_s16(pi2_src + x + 8 + src_strd4 + src_strd2);
                    src10S2 = vld1q_s16(pi2_src + x + 8 + src_strd4 + src_strd3);
                    CALCULATE_ROW_WD8x(src0S2, src1S2, src2S2, src3S2,
                                       src4S2, src5S2, src6S2, src7S2,
                                       pu2_dst + 8 + x)
                    CALCULATE_ROW_WD8x(src1S2, src2S2, src3S2, src4S2,
                                       src5S2, src6S2, src7S2, src8S2,
                                       pu2_dst + 8 + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2S2, src3S2, src4S2, src5S2,
                                       src6S2, src7S2, src8S2, src9S2,
                                       pu2_dst + 8 + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3S2, src4S2, src5S2, src6S2,
                                       src7S2, src8S2, src9S2, src10S2,
                                       pu2_dst + 8 + x + dst_strd3)
                }
                pi2_src += src_strd4;
                pu2_dst += dst_strd4;
            }
        } else {
            for (y = 0; y < ht; y += 4) {
                for (x = 0; x < wd; x += 8) {
                    src0S2 = vld1q_s16(pi2_src + x - src_strd3);
                    src1S2 = vld1q_s16(pi2_src + x - src_strd2);
                    src2S2 = vld1q_s16(pi2_src + x - src_strd);
                    src3S2 = vld1q_s16(pi2_src + x);
                    src4S2 = vld1q_s16(pi2_src + x + src_strd);
                    src5S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src6S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src7S2 = vld1q_s16(pi2_src + x + src_strd4);
                    src8S2 = vld1q_s16(pi2_src + x + src_strd4 + src_strd);
                    src9S2 = vld1q_s16(pi2_src + x + src_strd4 + src_strd2);
                    src10S2 = vld1q_s16(pi2_src + x + src_strd4 + src_strd3);

                    CALCULATE_ROW_WD8x(src0S2, src1S2, src2S2, src3S2,
                                       src4S2, src5S2, src6S2, src7S2, pu2_dst + x)
                    CALCULATE_ROW_WD8x(src1S2, src2S2, src3S2, src4S2,
                                       src5S2, src6S2, src7S2, src8S2,
                                       pu2_dst + x + dst_strd)
                    CALCULATE_ROW_WD8x(src2S2, src3S2, src4S2, src5S2,
                                       src6S2, src7S2, src8S2, src9S2,
                                       pu2_dst + x + dst_strd2)
                    CALCULATE_ROW_WD8x(src3S2, src4S2, src5S2, src6S2,
                                       src7S2, src8S2, src9S2, src10S2,
                                       pu2_dst + x + dst_strd3)
                }
                pi2_src += src_strd4;
                pu2_dst += dst_strd4;
            }
        }
    }
    else if (!(wd & 3)) {
        int16x4_t src0S2, src1S2, src2S2, src3S2, src4S2, src5S2;
        int16x4_t src6S2, src7S2, src8S2, src9S2, src10S2;
        int32x4_t mul0S4;

        int16x4_t coeff0S2 = vdup_lane_s16(vget_low_s16(coeffS2), 0);
        int16x4_t coeff1S2 = vdup_lane_s16(vget_low_s16(coeffS2), 1);
        int16x4_t coeff2S2 = vdup_lane_s16(vget_low_s16(coeffS2), 2);
        int16x4_t coeff3S2 = vdup_lane_s16(vget_low_s16(coeffS2), 3);
        int16x4_t coeff4S2 = vdup_lane_s16(vget_high_s16(coeffS2), 0);
        int16x4_t coeff5S2 = vdup_lane_s16(vget_high_s16(coeffS2), 1);
        int16x4_t coeff6S2 = vdup_lane_s16(vget_high_s16(coeffS2), 2);
        int16x4_t coeff7S2 = vdup_lane_s16(vget_high_s16(coeffS2), 3);

        for (y = 0; y < ht; y += 4) {
            for (x = 0; x < wd; x += 4) {
                src0S2 = vld1_s16(pi2_src + x - src_strd3);
                src1S2 = vld1_s16(pi2_src + x - src_strd2);
                src2S2 = vld1_s16(pi2_src + x - src_strd);
                src3S2 = vld1_s16(pi2_src + x + 0);
                src4S2 = vld1_s16(pi2_src + x + src_strd);
                src5S2 = vld1_s16(pi2_src + x + src_strd2);
                src6S2 = vld1_s16(pi2_src + x + src_strd3);
                src7S2 = vld1_s16(pi2_src + x + src_strd4);
                src8S2 = vld1_s16(pi2_src + x + src_strd4 + src_strd);
                src9S2 = vld1_s16(pi2_src + x + src_strd4 + src_strd2);
                src10S2 = vld1_s16(pi2_src + x + src_strd4 + src_strd3);
                CALCULATE_ROW_WD4x(src0S2, src1S2, src2S2, src3S2,
                                   src4S2, src5S2, src6S2, src7S2, pu2_dst + x)
                CALCULATE_ROW_WD4x(src1S2, src2S2, src3S2, src4S2,
                                   src5S2, src6S2, src7S2, src8S2,
                                   pu2_dst + x + dst_strd)
                CALCULATE_ROW_WD4x(src2S2, src3S2, src4S2, src5S2,
                                   src6S2, src7S2, src8S2, src9S2,
                                   pu2_dst + x + dst_strd2)
                CALCULATE_ROW_WD4x(src3S2, src4S2, src5S2, src6S2,
                                   src7S2, src8S2, src9S2, src10S2,
                                   pu2_dst + x + dst_strd3)
            }
            pi2_src += src_strd4;
            pu2_dst += dst_strd4;
        }
    }
#undef CALCULATE_ROW_WD8x
#undef CALCULATE_ROW_WD4x
}

/**
*******************************************************************************
*
* @brief
*      Luma prediction filter for vertical 16bit input & output
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*    the elements pointed by 'pi2_src' and writes to the location pointed by
*    'pi2_dst'. Input is 16 bits. The filter output is downshifted by 6 and
*    8192 is subtracted to store it as a 16-bit number. The output is used as
*    an input to weighted prediction.
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples (unused)
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/


void ihevc_hbd_inter_pred_luma_vert_w16inp_w16out_neonintr(WORD16 *pi2_src,
                                                           WORD16 *pi2_dst,
                                                           WORD32 src_strd,
                                                           WORD32 dst_strd,
                                                           WORD8 *pi1_coeff,
                                                           WORD32 ht,
                                                           WORD32 wd,
                                                           UWORD8 bit_depth)
{
    UNUSED(bit_depth);
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = (src_strd << 2);
    WORD32 src_strd8 = (src_strd4 << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 dst_strd3 = dst_strd2 + dst_strd;
    WORD32 dst_strd4 = (dst_strd << 2);
    WORD32 x, y;

    int16x8_t offsetS2 = vdupq_n_s16(OFFSET14);
    int8x8_t  coeffS1 = vld1_s8(pi1_coeff);
    int16x8_t coeffS2 = vmovl_s8(coeffS1);
    int16x4_t coeff_lo = vget_low_s16(coeffS2);
    int16x4_t coeff_hi = vget_high_s16(coeffS2);

#define CALCULATE_ROW_WD8x(src0, src1, src2, src3, src4,                     \
                           src5, src6, src7, out)                            \
    {                                                                        \
        int32x4_t l0, l1, h0, h1;                                            \
        l0 = vmull_lane_s16(vget_low_s16(src0), coeff_lo, 0);                \
        h0 = vmull_lane_s16(vget_high_s16(src0), coeff_lo, 0);               \
        l1 = vmull_lane_s16(vget_low_s16(src1), coeff_lo, 1);                \
        h1 = vmull_lane_s16(vget_high_s16(src1), coeff_lo, 1);               \
        l0 = vmlal_lane_s16(l0, vget_low_s16(src2), coeff_lo, 2);            \
        h0 = vmlal_lane_s16(h0, vget_high_s16(src2), coeff_lo, 2);           \
        l1 = vmlal_lane_s16(l1, vget_low_s16(src3), coeff_lo, 3);            \
        h1 = vmlal_lane_s16(h1, vget_high_s16(src3), coeff_lo, 3);           \
        l1 = vmlal_lane_s16(l1, vget_low_s16(src4), coeff_hi, 0);            \
        h1 = vmlal_lane_s16(h1, vget_high_s16(src4), coeff_hi, 0);           \
        l0 = vmlal_lane_s16(l0, vget_low_s16(src5), coeff_hi, 1);            \
        h0 = vmlal_lane_s16(h0, vget_high_s16(src5), coeff_hi, 1);           \
        l1 = vmlal_lane_s16(l1, vget_low_s16(src6), coeff_hi, 2);            \
        h1 = vmlal_lane_s16(h1, vget_high_s16(src6), coeff_hi, 2);           \
        l0 = vmlal_lane_s16(l0, vget_low_s16(src7), coeff_hi, 3);            \
        h0 = vmlal_lane_s16(h0, vget_high_s16(src7), coeff_hi, 3);           \
        l0 = vaddq_s32(l0, l1);                                              \
        h0 = vaddq_s32(h0, h1);                                              \
        int16x8_t res = vsubq_s16(vcombine_s16(vshrn_n_s32(l0, FILTER_PREC), \
                                               vshrn_n_s32(h0, FILTER_PREC)),\
                                  offsetS2);                                 \
        vst1q_s16((out), res);                                               \
    }

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, src4,                     \
                           src5, src6, src7, out)                            \
    {                                                                        \
        int32x4_t l0, l1;                                                    \
        l0 = vmull_lane_s16(src0, coeff_lo, 0);                              \
        l1 = vmull_lane_s16(src1, coeff_lo, 1);                              \
        l0 = vmlal_lane_s16(l0, src2, coeff_lo, 2);                          \
        l1 = vmlal_lane_s16(l1, src3, coeff_lo, 3);                          \
        l1 = vmlal_lane_s16(l1, src4, coeff_hi, 0);                          \
        l0 = vmlal_lane_s16(l0, src5, coeff_hi, 1);                          \
        l1 = vmlal_lane_s16(l1, src6, coeff_hi, 2);                          \
        l0 = vmlal_lane_s16(l0, src7, coeff_hi, 3);                          \
        l0 = vaddq_s32(l0, l1);                                              \
        int16x4_t res_l = vsub_s16(vshrn_n_s32(l0, FILTER_PREC),             \
                                   vget_low_s16(offsetS2));                  \
        vst1_s16((out), res_l);                                              \
    }

    if (!(wd & 7)) {
        for (y = 0; y < ht; y += 4) {
            for (x = 0; x < wd; x += 8) {
                const WORD16* src = pi2_src + x - src_strd3;
                WORD16* dst = pi2_dst + x;
                int16x8_t s0 = vld1q_s16(src);
                int16x8_t s1 = vld1q_s16(src + src_strd);
                int16x8_t s2 = vld1q_s16(src + src_strd2);
                int16x8_t s3 = vld1q_s16(src + src_strd3);
                int16x8_t s4 = vld1q_s16(src + src_strd4);
                int16x8_t s5 = vld1q_s16(src + src_strd4 + src_strd);
                int16x8_t s6 = vld1q_s16(src + src_strd4 + src_strd2);
                int16x8_t s7 = vld1q_s16(src + src_strd4 + src_strd3);
                int16x8_t s8 = vld1q_s16(src + src_strd8);
                int16x8_t s9 = vld1q_s16(src + src_strd8 + src_strd);
                int16x8_t s10 = vld1q_s16(src + src_strd8 + src_strd2);

                CALCULATE_ROW_WD8x(s0, s1, s2, s3, s4, s5, s6, s7, dst);
                CALCULATE_ROW_WD8x(s1, s2, s3, s4, s5, s6, s7, s8, dst + dst_strd);
                CALCULATE_ROW_WD8x(s2, s3, s4, s5, s6, s7, s8, s9, dst + dst_strd2);
                CALCULATE_ROW_WD8x(s3, s4, s5, s6, s7, s8, s9, s10, dst + dst_strd3);
            }
            pi2_src += src_strd4;
            pi2_dst += dst_strd4;
        }
    }
    else if (!(wd & 3)) {
        for (y = 0; y < ht; y += 4) {
            for (x = 0; x < wd; x += 4) {
                const WORD16* src = pi2_src + x - src_strd3;
                WORD16* dst = pi2_dst + x;
                int16x4_t s0 = vld1_s16(src);
                int16x4_t s1 = vld1_s16(src + src_strd);
                int16x4_t s2 = vld1_s16(src + src_strd2);
                int16x4_t s3 = vld1_s16(src + src_strd3);
                int16x4_t s4 = vld1_s16(src + src_strd4);
                int16x4_t s5 = vld1_s16(src + src_strd4 + src_strd);
                int16x4_t s6 = vld1_s16(src + src_strd4 + src_strd2);
                int16x4_t s7 = vld1_s16(src + src_strd4 + src_strd3);
                int16x4_t s8 = vld1_s16(src + src_strd8);
                int16x4_t s9 = vld1_s16(src + src_strd8 + src_strd);
                int16x4_t s10 = vld1_s16(src + src_strd8 + src_strd2);

                CALCULATE_ROW_WD4x(s0, s1, s2, s3, s4, s5, s6, s7, dst);
                CALCULATE_ROW_WD4x(s1, s2, s3, s4, s5, s6, s7, s8, dst + dst_strd);
                CALCULATE_ROW_WD4x(s2, s3, s4, s5, s6, s7, s8, s9, dst + dst_strd2);
                CALCULATE_ROW_WD4x(s3, s4, s5, s6, s7, s8, s9, s10, dst + dst_strd3);
            }
            pi2_src += src_strd4;
            pi2_dst += dst_strd4;
        }
    }
#undef CALCULATE_ROW_WD8x
#undef CALCULATE_ROW_WD4x
}

/**
*******************************************************************************
*
* @brief
*   Interprediction chroma function for copy
*
* @par Description:
*   Copies the chroma array of width 'wd' and height 'ht' from the location
*   pointed by 'pu2_src' to the location pointed by 'pu2_dst'. Note that each
*   chroma row contains 2 * wd samples.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients (unused)
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples (unused)
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_copy_neonintr(UWORD16 *pu2_src,
                                               UWORD16 *pu2_dst,
                                               WORD32 src_strd,
                                               WORD32 dst_strd,
                                               WORD8 *pi1_coeff,
                                               WORD32 ht,
                                               WORD32 wd,
                                               UWORD8 bit_depth)
{
    UNUSED(pi1_coeff);
    UNUSED(bit_depth);
    WORD32 row, col, wdx2;
    uint16x8_t src0_16x8, src1_16x8, src2_16x8, src3_16x8;
    uint16x4_t src0_16x4, src1_16x4, src2_16x4, src3_16x4;
    WORD32 src_strd2 = src_strd * 2;
    WORD32 src_strd3 = src_strd * 3;
    WORD32 dst_strd2 = dst_strd * 2;
    WORD32 dst_strd3 = dst_strd * 3;

    wdx2 = wd * 2;

    if (0 == (ht & 3)) /* ht multiple of 4 */
    {
        if (0 == (wdx2 & 7)) /* wdx2 multiple of 8 case */
        {
            for (row = 0; row < ht; row += 4)
            {
                for (col = 0; col < wdx2; col += 8)
                {
                    /* load 8 16-bit pixel values per row */
                    src0_16x8 = vld1q_u16(pu2_src);
                    src1_16x8 = vld1q_u16(pu2_src + src_strd);
                    src2_16x8 = vld1q_u16(pu2_src + src_strd2);
                    src3_16x8 = vld1q_u16(pu2_src + src_strd3);

                    /* store 8 16-bit pixel values per row */
                    vst1q_u16(pu2_dst, src0_16x8);
                    vst1q_u16(pu2_dst + dst_strd, src1_16x8);
                    vst1q_u16(pu2_dst + dst_strd2, src2_16x8);
                    vst1q_u16(pu2_dst + dst_strd3, src3_16x8);

                    pu2_src += 8; /* pointer update */
                    pu2_dst += 8; /* pointer update */
                }

                pu2_src += 4 * src_strd - wdx2; /* pointer update */
                pu2_dst += 4 * dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 = multiple of 4 case */
        {
            for (row = 0; row < ht; row += 4)
            {
                for (col = 0; col < wdx2; col += 4)
                {
                    /* load 4 16-bit pixel values per row */
                    src0_16x4 = vld1_u16(pu2_src);
                    src1_16x4 = vld1_u16(pu2_src + src_strd);
                    src2_16x4 = vld1_u16(pu2_src + src_strd2);
                    src3_16x4 = vld1_u16(pu2_src + src_strd3);

                    /* store 4 16-bit pixel values per row */
                    vst1_u16(pu2_dst, src0_16x4);
                    vst1_u16(pu2_dst + dst_strd, src1_16x4);
                    vst1_u16(pu2_dst + dst_strd2, src2_16x4);
                    vst1_u16(pu2_dst + dst_strd3, src3_16x4);

                    pu2_src += 4; /* pointer update */
                    pu2_dst += 4; /* pointer update */
                }

                pu2_src += 4 * src_strd - wdx2; /* pointer update */
                pu2_dst += 4 * dst_strd - wdx2; /* pointer update */
            }
        }
    }
    else if (0 == (ht & 1)) /* ht multiple of 2 */
    {
        if (0 == (wdx2 & 7)) /* wdx2 multiple of 8 case */
        {
            for (row = 0; row < ht; row += 2)
            {
                for (col = 0; col < wdx2; col += 8)
                {
                    /* load 8 16-bit pixel values per row */
                    src0_16x8 = vld1q_u16(pu2_src);
                    src1_16x8 = vld1q_u16(pu2_src + src_strd);

                    /* store 8 16-bit pixel values per row */
                    vst1q_u16(pu2_dst, src0_16x8);
                    vst1q_u16(pu2_dst + dst_strd, src1_16x8);

                    pu2_src += 8; /* pointer update */
                    pu2_dst += 8; /* pointer update */
                }

                pu2_src += 2 * src_strd - wdx2; /* pointer update */
                pu2_dst += 2 * dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 = multiple of 4 case */
        {
            for (row = 0; row < ht; row += 2)
            {
                for (col = 0; col < wdx2; col += 4)
                {
                    /* load 4 16-bit pixel values per row */
                    src0_16x4 = vld1_u16(pu2_src);
                    src1_16x4 = vld1_u16(pu2_src + src_strd);

                    /* store 4 16-bit pixel values per row */
                    vst1_u16(pu2_dst, src0_16x4);
                    vst1_u16(pu2_dst + dst_strd, src1_16x4);

                    pu2_src += 4; /* pointer update */
                    pu2_dst += 4; /* pointer update */
                }

                pu2_src += 2 * src_strd - wdx2; /* pointer update */
                pu2_dst += 2 * dst_strd - wdx2; /* pointer update */
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     Chroma interprediction filter for horizontal input
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and writes to the location pointed
*    by 'pu2_dst'. The output is downshifted by 6 and clipped to bit_depth.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_chroma_horz_neonintr(UWORD16 *pu2_src,
                                               UWORD16 *pu2_dst,
                                               WORD32 src_strd,
                                               WORD32 dst_strd,
                                               WORD8 *pi1_coeff,
                                               WORD32 ht,
                                               WORD32 wd,
                                               UWORD8 bit_depth)
{
    int8x8_t coeffS1 = vld1_s8(pi1_coeff);
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 x, y;

    uint16x8_t src0U2, src1U2, src2U2, src3U2, src4U2, src5U2, src6U2, src7U2;
    uint16x8_t mul0U2, mul1U2, mul2U2, mul3U2, mul4U2, mul5U2;
    uint16x4_t aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2;
    uint16x4_t aMul0U2, aMul1U2, aMul2U2;
    uint16x4_t aCoeff0U2, aCoeff1U2, aCoeff2U2, aCoeff3U2;
    uint16x4_t sft0U2, sft1U2, sft2U2, sft3U2;
    uint16x8_t res0U2, res1U2;
    uint32x4_t add0U4, add1U4, add2U4, add3U4;
    int32x4_t  sub0U4, sub1U4, sub2U4, sub3U4;
    uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2;
    uint16x8_t max_val_16x8 = vdupq_n_u16((1 << bit_depth) - 1);
    uint16x4_t max_val_16x4 = vdup_n_u16((1 << bit_depth) - 1);

    pu2_src -= 2;

#define CALCULATE_ROWS_WD8x(pSrc, pDst)                                     \
        src0U2 = vld1q_u16(pSrc);                                           \
        src1U2 = vld1q_u16(pSrc + 2);                                       \
        src2U2 = vld1q_u16(pSrc + 4);                                       \
        src3U2 = vld1q_u16(pSrc + 6);                                       \
        src4U2 = vld1q_u16(pSrc + 8);                                       \
        src5U2 = vld1q_u16(pSrc + 10);                                      \
        src6U2 = vld1q_u16(pSrc + 12);                                      \
        src7U2 = vld1q_u16(pSrc + 14);                                      \
        mul0U2 = vmulq_u16(src0U2, coeff0U2);                               \
        mul1U2 = vmulq_u16(src1U2, coeff1U2);                               \
        mul2U2 = vmulq_u16(src2U2, coeff2U2);                               \
        mul0U2 = vmlaq_u16(mul0U2, src3U2, coeff3U2);                       \
        mul3U2 = vmulq_u16(src4U2, coeff0U2);                               \
        mul4U2 = vmulq_u16(src5U2, coeff1U2);                               \
        mul5U2 = vmulq_u16(src6U2, coeff2U2);                               \
        mul3U2 = vmlaq_u16(mul3U2, src7U2, coeff3U2);                       \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        add2U4 = vaddl_u16(vget_low_u16(mul4U2), vget_low_u16(mul5U2));     \
        add3U4 = vaddl_u16(vget_high_u16(mul4U2), vget_high_u16(mul5U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(\
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sub2U4 = vsubw_s16(vreinterpretq_s32_u32(add2U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul3U2)));    \
        sub3U4 = vsubw_s16(vreinterpretq_s32_u32(add3U4), vget_high_s16(\
                                         vreinterpretq_s16_u16(mul3U2)));   \
        sft0U2 = vqrshrun_n_s32(sub0U4, 6);                                 \
        sft1U2 = vqrshrun_n_s32(sub1U4, 6);                                 \
        sft2U2 = vqrshrun_n_s32(sub2U4, 6);                                 \
        sft3U2 = vqrshrun_n_s32(sub3U4, 6);                                 \
        res0U2 = vcombine_u16(sft0U2, sft1U2);                              \
        res1U2 = vcombine_u16(sft2U2, sft3U2);                              \
        res0U2 = vminq_u16(res0U2, max_val_16x8);                           \
        res1U2 = vminq_u16(res1U2, max_val_16x8);                           \
        vst1q_u16(pDst, res0U2);                                            \
        vst1q_u16(pDst + 8, res1U2);

#define CALCULATE_ROW_WD4x(pSrc, pDst)                                      \
        src0U2 = vld1q_u16(pSrc);                                           \
        src1U2 = vld1q_u16(pSrc + 2);                                       \
        src2U2 = vld1q_u16(pSrc + 4);                                       \
        src3U2 = vld1q_u16(pSrc + 6);                                       \
        mul0U2 = vmulq_u16(src0U2, coeff0U2);                               \
        mul1U2 = vmulq_u16(src1U2, coeff1U2);                               \
        mul2U2 = vmulq_u16(src2U2, coeff2U2);                               \
        mul0U2 = vmlaq_u16(mul0U2, src3U2, coeff3U2);                       \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sft0U2 = vqrshrun_n_s32(sub0U4, 6);                                 \
        sft1U2 = vqrshrun_n_s32(sub1U4, 6);                                 \
        res0U2 = vcombine_u16(sft0U2, sft1U2);                              \
        res0U2 = vminq_u16(res0U2, max_val_16x8);                           \
        vst1q_u16(pDst, res0U2);

#define CALCULATE_ROW_WD2x(pSrc, pDst)                                      \
        aSrc0U2 = vld1_u16(pSrc);                                           \
        aSrc1U2 = vld1_u16(pSrc + 2);                                       \
        aSrc2U2 = vld1_u16(pSrc + 4);                                       \
        aSrc3U2 = vld1_u16(pSrc + 6);                                       \
        aMul0U2 = vmul_u16(aSrc0U2, aCoeff0U2);                             \
        aMul1U2 = vmul_u16(aSrc1U2, aCoeff1U2);                             \
        aMul2U2 = vmul_u16(aSrc2U2, aCoeff2U2);                             \
        aMul0U2 = vmla_u16(aMul0U2, aSrc3U2, aCoeff3U2);                     \
        add0U4 = vaddl_u16(aMul1U2, aMul2U2);                               \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(aMul0U2));                 \
        sft0U2 = vqrshrun_n_s32(sub0U4, 6);                                 \
        sft0U2 = vmin_u16(sft0U2, max_val_16x4);                            \
        vst1_u16(pDst, sft0U2);

    uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(vabs_s8(coeffS1)));
    coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
    coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
    coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
    coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);
    aCoeff0U2 = vget_low_u16(coeff0U2);
    aCoeff1U2 = vget_low_u16(coeff1U2);
    aCoeff2U2 = vget_low_u16(coeff2U2);
    aCoeff3U2 = vget_low_u16(coeff3U2);

    if (!(wd & 15)) {
        for (y = 0; y < ht; y += 2) {
            for (x = 0; x < (wd << 1); x += 32) {
                CALCULATE_ROWS_WD8x(pu2_src + x, pu2_dst + x)
                CALCULATE_ROWS_WD8x(pu2_src + x + 16, pu2_dst + x + 16)
                UWORD16 *tmp_pu2_src = (pu2_src + src_strd);
                UWORD16 *tmp_pu2_dst = (pu2_dst + dst_strd);
                CALCULATE_ROWS_WD8x(tmp_pu2_src + x, tmp_pu2_dst + x)
                CALCULATE_ROWS_WD8x(tmp_pu2_src + x + 16, tmp_pu2_dst + x + 16)
            }
            pu2_src += src_strd2;
            pu2_dst += dst_strd2;
        }
    }
    else if (!(wd & 7)) {
        for (y = 0; y < ht; y += 2) {
            for (x = 0; x < (wd << 1); x += 16) {
                CALCULATE_ROWS_WD8x(pu2_src + x, pu2_dst + x)
                UWORD16 *tmp_pu2_src = (pu2_src + src_strd);
                UWORD16 *tmp_pu2_dst = (pu2_dst + dst_strd);
                CALCULATE_ROWS_WD8x(tmp_pu2_src + x, tmp_pu2_dst + x)
            }
            pu2_src += src_strd2;
            pu2_dst += dst_strd2;
        }
    }
    else if (!(wd & 3)) {
        for (y = 0; y < ht; y++) {
            for (x = 0; x < (wd << 1); x += 8) {
                CALCULATE_ROW_WD4x(pu2_src + x, pu2_dst + x)
            }
            pu2_src += src_strd;
            pu2_dst += dst_strd;
        }
    }
    else if (!(wd & 1)) {
        for (y = 0; y < ht; y++) {
            for (x = 0; x < (wd << 1); x += 4) {
                CALCULATE_ROW_WD2x(pu2_src + x, pu2_dst + x)
            }
            pu2_src += src_strd;
            pu2_dst += dst_strd;
        }
    }
#undef CALCULATE_ROWS_WD8x
#undef CALCULATE_ROW_WD4x
#undef CALCULATE_ROW_WD2x
}

/**
*******************************************************************************
*
* @brief
*     Chroma interprediction filter for vertical input
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*    the elements pointed by 'pu2_src' and writes to the location pointed by
*    'pu2_dst'. The output is downshifted by 6 and clipped to bit_depth.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_chroma_vert_neonintr(UWORD16 *pu2_src,
                                               UWORD16 *pu2_dst,
                                               WORD32 src_strd,
                                               WORD32 dst_strd,
                                               WORD8 *pi1_coeff,
                                               WORD32 ht,
                                               WORD32 wd,
                                               UWORD8 bit_depth)
{
    int8x8_t coeffS1 = vld1_s8(pi1_coeff);
    uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(vabs_s8(coeffS1)));
    pu2_src -= src_strd;
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = src_strd2 + src_strd2;
    WORD32 x, y;

    uint16x8_t src0U2, src1U2, src2U2, src3U2, src4U2, src5U2, src6U2, src7U2, src8U2, src9U2;
    uint16x8_t mul0U2, mul1U2, mul2U2, mul3U2, mul4U2, mul5U2;
    uint16x4_t sft0U2, sft1U2, sft2U2, sft3U2;
    uint16x8_t res0U2, res1U2;
    uint32x4_t add0U4, add1U4, add2U4, add3U4;
    int32x4_t  sub0U4, sub1U4, sub2U4, sub3U4;
    uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2;
    uint16x8_t max_val_16x8 = vdupq_n_u16((1 << bit_depth) - 1);
    uint16x4_t max_val_16x4 = vdup_n_u16((1 << bit_depth) - 1);

#define CALCULATE_ROWS_WD8x(src0, src1, src2, src3,                         \
                             src4, src5, src6, src7, pDst)                  \
        mul0U2 = vmulq_u16(src0, coeff0U2);                                 \
        mul1U2 = vmulq_u16(src1, coeff1U2);                                 \
        mul2U2 = vmulq_u16(src2, coeff2U2);                                 \
        mul0U2 = vmlaq_u16(mul0U2, src3, coeff3U2);                         \
        mul3U2 = vmulq_u16(src4, coeff0U2);                                 \
        mul4U2 = vmulq_u16(src5, coeff1U2);                                 \
        mul5U2 = vmulq_u16(src6, coeff2U2);                                 \
        mul3U2 = vmlaq_u16(mul3U2, src7, coeff3U2);                         \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        add2U4 = vaddl_u16(vget_low_u16(mul4U2), vget_low_u16(mul5U2));     \
        add3U4 = vaddl_u16(vget_high_u16(mul4U2), vget_high_u16(mul5U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sub2U4 = vsubw_s16(vreinterpretq_s32_u32(add2U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul3U2)));    \
        sub3U4 = vsubw_s16(vreinterpretq_s32_u32(add3U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul3U2)));   \
        sft0U2 = vqrshrun_n_s32(sub0U4, 6);                                 \
        sft1U2 = vqrshrun_n_s32(sub1U4, 6);                                 \
        sft2U2 = vqrshrun_n_s32(sub2U4, 6);                                 \
        sft3U2 = vqrshrun_n_s32(sub3U4, 6);                                 \
        res0U2 = vcombine_u16(sft0U2, sft1U2);                              \
        res1U2 = vcombine_u16(sft2U2, sft3U2);                              \
        res0U2 = vminq_u16(res0U2, max_val_16x8);                           \
        res1U2 = vminq_u16(res1U2, max_val_16x8);                           \
        vst1q_u16(pDst, res0U2);                                            \
        vst1q_u16(pDst + 8, res1U2);

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, pDst)                    \
        mul0U2 = vmulq_u16(src0, coeff0U2);                                 \
        mul1U2 = vmulq_u16(src1, coeff1U2);                                 \
        mul2U2 = vmulq_u16(src2, coeff2U2);                                 \
        mul0U2 = vmlaq_u16(mul0U2, src3, coeff3U2);                         \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sft0U2 = vqrshrun_n_s32(sub0U4, 6);                                 \
        sft1U2 = vqrshrun_n_s32(sub1U4, 6);                                 \
        res0U2 = vcombine_u16(sft0U2, sft1U2);                              \
        res0U2 = vminq_u16(res0U2, max_val_16x8);                           \
        vst1q_u16(pDst, res0U2);

#define CALCULATE_ROW_WD2x(src0, src1, src2, src3, pDst)                    \
        aMul0U2 = vmul_u16(src0, aCoeff0U2);                                \
        aMul1U2 = vmul_u16(src1, aCoeff1U2);                                \
        aMul2U2 = vmul_u16(src2, aCoeff2U2);                                \
        aMul0U2 = vmla_u16(aMul0U2, src3, aCoeff3U2);                       \
        add0U4 = vaddl_u16(aMul1U2, aMul2U2);                               \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(aMul0U2));                 \
        sft0U2 = vqrshrun_n_s32(sub0U4, 6);                                 \
        sft0U2 = vmin_u16(sft0U2, max_val_16x4);                            \
        vst1_u16(pDst, sft0U2);

    if (!(wd & 3)) {
        coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
        coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
        coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
        coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);

        if (!(wd & 15)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 32) {
                    src0U2 = vld1q_u16(pu2_src + x);
                    src1U2 = vld1q_u16(pu2_src + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + x + 8);
                    src5U2 = vld1q_u16(pu2_src + x + 8 + src_strd);
                    src6U2 = vld1q_u16(pu2_src + x + 8 + src_strd2);
                    src7U2 = vld1q_u16(pu2_src + x + 8 + src_strd3);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src9U2 = vld1q_u16(pu2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0U2, src1U2, src2U2, src3U2,
                                         src4U2, src5U2, src6U2, src7U2, pu2_dst + x)
                    CALCULATE_ROWS_WD8x(src1U2, src2U2, src3U2, src8U2,
                                         src5U2, src6U2, src7U2, src9U2,
                                         pu2_dst + dst_strd + x)

                    src0U2 = vld1q_u16(pu2_src + 16 + x);
                    src1U2 = vld1q_u16(pu2_src + 16 + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + 16 + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + 16 + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + 16 + x + 8);
                    src5U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd);
                    src6U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd2);
                    src7U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd3);
                    src8U2 = vld1q_u16(pu2_src + 16 + x + src_strd4);
                    src9U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0U2, src1U2, src2U2, src3U2,
                                         src4U2, src5U2, src6U2, src7U2,
                                         pu2_dst + x + 16)
                    CALCULATE_ROWS_WD8x(src1U2, src2U2, src3U2, src8U2,
                                         src5U2, src6U2, src7U2, src9U2,
                                         pu2_dst + dst_strd + x + 16)
                }
                pu2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
        else if (!(wd & 7)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 16) {
                    src0U2 = vld1q_u16(pu2_src + x);
                    src1U2 = vld1q_u16(pu2_src + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + x + 8);
                    src5U2 = vld1q_u16(pu2_src + x + 8 + src_strd);
                    src6U2 = vld1q_u16(pu2_src + x + 8 + src_strd2);
                    src7U2 = vld1q_u16(pu2_src + x + 8 + src_strd3);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src9U2 = vld1q_u16(pu2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0U2, src1U2, src2U2, src3U2,
                                         src4U2, src5U2, src6U2, src7U2, pu2_dst + x)
                    CALCULATE_ROWS_WD8x(src1U2, src2U2, src3U2, src8U2,
                                         src5U2, src6U2, src7U2, src9U2,
                                         pu2_dst + dst_strd + x)
                }
                pu2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
        else {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 8) {
                    src0U2 = vld1q_u16(pu2_src + x);
                    src1U2 = vld1q_u16(pu2_src + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + x + src_strd4);
                    CALCULATE_ROW_WD4x(src0U2, src1U2, src2U2, src3U2, pu2_dst + x)
                    CALCULATE_ROW_WD4x(src1U2, src2U2, src3U2, src4U2,
                                       pu2_dst + x + dst_strd)
                }
                pu2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
    }
    else if (!(wd & 1)) {
        uint16x4_t aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2, aSrc4U2;
        uint16x4_t aMul0U2, aMul1U2, aMul2U2;
        uint16x4_t aCoeff0U2, aCoeff1U2, aCoeff2U2, aCoeff3U2;
        aCoeff0U2 = vdup_lane_u16(vget_low_u16(coeffU2), 0);
        aCoeff1U2 = vdup_lane_u16(vget_low_u16(coeffU2), 1);
        aCoeff2U2 = vdup_lane_u16(vget_low_u16(coeffU2), 2);
        aCoeff3U2 = vdup_lane_u16(vget_low_u16(coeffU2), 3);

        for (y = 0; y < ht; y += 2) {
            for (x = 0; x < (wd << 1); x += 4) {
                aSrc0U2 = vld1_u16(pu2_src + x);
                aSrc1U2 = vld1_u16(pu2_src + x + src_strd);
                aSrc2U2 = vld1_u16(pu2_src + x + src_strd2);
                aSrc3U2 = vld1_u16(pu2_src + x + src_strd3);
                aSrc4U2 = vld1_u16(pu2_src + x + src_strd4);
                CALCULATE_ROW_WD2x(aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2, pu2_dst + x)
                CALCULATE_ROW_WD2x(aSrc1U2, aSrc2U2, aSrc3U2, aSrc4U2,
                                   pu2_dst + x + dst_strd)
            }
            pu2_src += src_strd2;
            pu2_dst += dst_strd2;
        }
    }
#undef CALCULATE_ROWS_WD8x
#undef CALCULATE_ROW_WD4x
#undef CALCULATE_ROW_WD2x
}

/**
*******************************************************************************
*
* @brief
*   Interprediction chroma function for copy with 16-bit output
*
* @par Description:
*   Copies the chroma array of width 'wd' and height 'ht' from the location
*   pointed by 'pu2_src' to the location pointed by 'pi2_dst' with left shift
*   by (14 - bit_depth). Note that each chroma row contains 2 * wd samples.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients (unused)
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_copy_w16out_neonintr(UWORD16 *pu2_src,
                                                      WORD16 *pi2_dst,
                                                      WORD32 src_strd,
                                                      WORD32 dst_strd,
                                                      WORD8 *pi1_coeff,
                                                      WORD32 ht,
                                                      WORD32 wd,
                                                      UWORD8 bit_depth)
{
    WORD32 row, col, wdx2;
    uint16x8_t src0_16x8, src1_16x8, src2_16x8, src3_16x8;
    uint16x4_t src0_16x4, src1_16x4, src2_16x4, src3_16x4;
    WORD32 src_strd2 = src_strd * 2;
    WORD32 src_strd3 = src_strd * 3;
    WORD32 dst_strd2 = dst_strd * 2;
    WORD32 dst_strd3 = dst_strd * 3;

    UNUSED(pi1_coeff);

    wdx2 = wd * 2;

    if (0 == (ht & 3)) /* ht multiple of 4 case */
    {
        if (0 == (wdx2 & 7)) /* multiple of 8 case */
        {
            int16x8_t shift_vec = vdupq_n_s16(14 - bit_depth);

            for (row = 0; row < ht; row += 4)
            {
                for (col = 0; col < wdx2; col += 8)
                {
                    /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                    src0_16x8 = vld1q_u16(pu2_src);
                    src1_16x8 = vld1q_u16(pu2_src + src_strd);
                    src2_16x8 = vld1q_u16(pu2_src + src_strd2);
                    src3_16x8 = vld1q_u16(pu2_src + src_strd3);

                    src0_16x8 = vshlq_u16(src0_16x8, shift_vec);
                    src1_16x8 = vshlq_u16(src1_16x8, shift_vec);
                    src2_16x8 = vshlq_u16(src2_16x8, shift_vec);
                    src3_16x8 = vshlq_u16(src3_16x8, shift_vec);

                    /* storing 8 16-bit output values */
                    vst1q_s16(pi2_dst, vreinterpretq_s16_u16(src0_16x8));
                    vst1q_s16(pi2_dst + dst_strd, vreinterpretq_s16_u16(src1_16x8));
                    vst1q_s16(pi2_dst + dst_strd2, vreinterpretq_s16_u16(src2_16x8));
                    vst1q_s16(pi2_dst + dst_strd3, vreinterpretq_s16_u16(src3_16x8));

                    pu2_src += 8; /* pointer update */
                    pi2_dst += 8; /* pointer update */
                } /* inner for loop ends here(8-output values in single iteration) */

                pu2_src += 4 * src_strd - wdx2; /* pointer update */
                pi2_dst += 4 * dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 multiple of 4 case */
        {
            int16x4_t shift_vec4 = vdup_n_s16(14 - bit_depth);

            for (row = 0; row < ht; row += 4)
            {
                for (col = 0; col < wdx2; col += 4)
                {
                    /* load 4 pixel values from 3:0 pos. relative to cur. pos. */
                    src0_16x4 = vld1_u16(pu2_src);
                    src1_16x4 = vld1_u16(pu2_src + src_strd);
                    src2_16x4 = vld1_u16(pu2_src + src_strd2);
                    src3_16x4 = vld1_u16(pu2_src + src_strd3);

                    src0_16x4 = vshl_u16(src0_16x4, shift_vec4);
                    src1_16x4 = vshl_u16(src1_16x4, shift_vec4);
                    src2_16x4 = vshl_u16(src2_16x4, shift_vec4);
                    src3_16x4 = vshl_u16(src3_16x4, shift_vec4);

                    /* storing 4 16-bit output values */
                    vst1_s16(pi2_dst, vreinterpret_s16_u16(src0_16x4));
                    vst1_s16(pi2_dst + dst_strd, vreinterpret_s16_u16(src1_16x4));
                    vst1_s16(pi2_dst + dst_strd2, vreinterpret_s16_u16(src2_16x4));
                    vst1_s16(pi2_dst + dst_strd3, vreinterpret_s16_u16(src3_16x4));

                    pu2_src += 4; /* pointer update */
                    pi2_dst += 4; /* pointer update */
                } /* inner for loop ends here(4-output values in single iteration) */

                pu2_src += 4 * src_strd - wdx2; /* pointer update */
                pi2_dst += 4 * dst_strd - wdx2; /* pointer update */
            }
        }
    }
    else if (0 == (ht & 1)) /* ht multiple of 2 case */
    {
        if (0 == (wdx2 & 7)) /* multiple of 8 case */
        {
            int16x8_t shift_vec = vdupq_n_s16(14 - bit_depth);

            for (row = 0; row < ht; row += 2)
            {
                for (col = 0; col < wdx2; col += 8)
                {
                    /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                    src0_16x8 = vld1q_u16(pu2_src);
                    src1_16x8 = vld1q_u16(pu2_src + src_strd);

                    src0_16x8 = vshlq_u16(src0_16x8, shift_vec);
                    src1_16x8 = vshlq_u16(src1_16x8, shift_vec);

                    /* storing 8 16-bit output values */
                    vst1q_s16(pi2_dst, vreinterpretq_s16_u16(src0_16x8));
                    vst1q_s16(pi2_dst + dst_strd, vreinterpretq_s16_u16(src1_16x8));

                    pu2_src += 8; /* pointer update */
                    pi2_dst += 8; /* pointer update */
                } /* inner for loop ends here(8-output values in single iteration) */

                pu2_src += 2 * src_strd - wdx2; /* pointer update */
                pi2_dst += 2 * dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 multiple of 4 case */
        {
            int16x4_t shift_vec4 = vdup_n_s16(14 - bit_depth);

            for (row = 0; row < ht; row += 2)
            {
                for (col = 0; col < wdx2; col += 4)
                {
                    /* load 4 pixel values from 3:0 pos. relative to cur. pos. */
                    src0_16x4 = vld1_u16(pu2_src);
                    src1_16x4 = vld1_u16(pu2_src + src_strd);

                    src0_16x4 = vshl_u16(src0_16x4, shift_vec4);
                    src1_16x4 = vshl_u16(src1_16x4, shift_vec4);

                    /* storing 4 16-bit output values */
                    vst1_s16(pi2_dst, vreinterpret_s16_u16(src0_16x4));
                    vst1_s16(pi2_dst + dst_strd, vreinterpret_s16_u16(src1_16x4));

                    pu2_src += 4; /* pointer update */
                    pi2_dst += 4; /* pointer update */
                } /* inner for loop ends here(4-output values in single iteration) */

                pu2_src += 2 * src_strd - wdx2; /* pointer update */
                pi2_dst += 2 * dst_strd - wdx2; /* pointer update */
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*       chroma interprediction filter to store horizontal 16bit ouput
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and writes to the location pointed
*    by 'pi2_dst'. The output is downshifted by (bit_depth - 8) and is used
*    as an input for vertical filtering or weighted prediction.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_horz_w16out_neonintr(UWORD16 *pu2_src,
                                                      WORD16 *pi2_dst,
                                                      WORD32 src_strd,
                                                      WORD32 dst_strd,
                                                      WORD8 *pi1_coeff,
                                                      WORD32 ht,
                                                      WORD32 wd,
                                                      UWORD8 bit_depth)
{
    int32x4_t shift_vec = vdupq_n_s32(8 - bit_depth);
    int8x8_t coeffS1 = vld1_s8(pi1_coeff);
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 x, y;

    uint16x8_t src0U2, src1U2, src2U2, src3U2, src4U2, src5U2, src6U2, src7U2;
    uint16x8_t mul0U2, mul1U2, mul2U2, mul3U2, mul4U2, mul5U2;
    uint16x4_t aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2;
    uint16x4_t aMul0U2, aMul1U2, aMul2U2;
    uint16x4_t aCoeff0U2, aCoeff1U2, aCoeff2U2, aCoeff3U2;
    int16x4_t  sft0S2, sft1S2, sft2S2, sft3S2;
    uint32x4_t add0U4, add1U4, add2U4, add3U4;
    int32x4_t  sub0U4, sub1U4, sub2U4, sub3U4;
    uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2;

    pu2_src -= 2;

#define CALCULATE_ROWS_WD8x(pSrc, pDst)                                     \
        src0U2 = vld1q_u16(pSrc);                                           \
        src1U2 = vld1q_u16(pSrc + 2);                                       \
        src2U2 = vld1q_u16(pSrc + 4);                                       \
        src3U2 = vld1q_u16(pSrc + 6);                                       \
        src4U2 = vld1q_u16(pSrc + 8);                                       \
        src5U2 = vld1q_u16(pSrc + 10);                                      \
        src6U2 = vld1q_u16(pSrc + 12);                                      \
        src7U2 = vld1q_u16(pSrc + 14);                                      \
        mul0U2 = vmulq_u16(src0U2, coeff0U2);                               \
        mul1U2 = vmulq_u16(src1U2, coeff1U2);                               \
        mul2U2 = vmulq_u16(src2U2, coeff2U2);                               \
        mul0U2 = vmlaq_u16(mul0U2, src3U2, coeff3U2);                       \
        mul3U2 = vmulq_u16(src4U2, coeff0U2);                               \
        mul4U2 = vmulq_u16(src5U2, coeff1U2);                               \
        mul5U2 = vmulq_u16(src6U2, coeff2U2);                               \
        mul3U2 = vmlaq_u16(mul3U2, src7U2, coeff3U2);                       \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        add2U4 = vaddl_u16(vget_low_u16(mul4U2), vget_low_u16(mul5U2));     \
        add3U4 = vaddl_u16(vget_high_u16(mul4U2), vget_high_u16(mul5U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sub2U4 = vsubw_s16(vreinterpretq_s32_u32(add2U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul3U2)));    \
        sub3U4 = vsubw_s16(vreinterpretq_s32_u32(add3U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul3U2)));   \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        sft1S2 = vmovn_s32(vshlq_s32(sub1U4, shift_vec));                   \
        sft2S2 = vmovn_s32(vshlq_s32(sub2U4, shift_vec));                   \
        sft3S2 = vmovn_s32(vshlq_s32(sub3U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);                                             \
        vst1_s16(pDst + 4, sft1S2);                                         \
        vst1_s16(pDst + 8, sft2S2);                                         \
        vst1_s16(pDst + 12, sft3S2);


#define CALCULATE_ROW_WD4x(pSrc, pDst)                                      \
        src0U2 = vld1q_u16(pSrc);                                           \
        src1U2 = vld1q_u16(pSrc + 2);                                       \
        src2U2 = vld1q_u16(pSrc + 4);                                       \
        src3U2 = vld1q_u16(pSrc + 6);                                       \
        mul0U2 = vmulq_u16(src0U2, coeff0U2);                               \
        mul1U2 = vmulq_u16(src1U2, coeff1U2);                               \
        mul2U2 = vmulq_u16(src2U2, coeff2U2);                               \
        mul0U2 = vmlaq_u16(mul0U2, src3U2, coeff3U2);                       \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(\
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        sft1S2 = vmovn_s32(vshlq_s32(sub1U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);                                             \
        vst1_s16(pDst + 4, sft1S2);

#define CALCULATE_ROW_WD2x(pSrc, pDst)                                      \
        aSrc0U2 = vld1_u16(pSrc);                                           \
        aSrc1U2 = vld1_u16(pSrc + 2);                                       \
        aSrc2U2 = vld1_u16(pSrc + 4);                                       \
        aSrc3U2 = vld1_u16(pSrc + 6);                                       \
        aMul0U2 = vmul_u16(aSrc0U2, aCoeff0U2);                             \
        aMul1U2 = vmul_u16(aSrc1U2, aCoeff1U2);                             \
        aMul2U2 = vmul_u16(aSrc2U2, aCoeff2U2);                             \
        aMul0U2 = vmla_u16(aMul0U2, aSrc3U2, aCoeff3U2);                    \
        add0U4 = vaddl_u16(aMul1U2, aMul2U2);                               \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(aMul0U2));                 \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);

    uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(vabs_s8(coeffS1)));
    coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
    coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
    coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
    coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);
    aCoeff0U2 = vget_low_u16(coeff0U2);
    aCoeff1U2 = vget_low_u16(coeff1U2);
    aCoeff2U2 = vget_low_u16(coeff2U2);
    aCoeff3U2 = vget_low_u16(coeff3U2);

    if (!(wd & 15)) {
        for (y = 0; y < (ht & ~1); y += 2) {
            for (x = 0; x < (wd << 1); x += 32) {
                CALCULATE_ROWS_WD8x(pu2_src + x,      pi2_dst + x     )
                CALCULATE_ROWS_WD8x(pu2_src + x + 16, pi2_dst + x + 16)
                UWORD16 *tmp_pu2_src = (pu2_src + src_strd);
                WORD16  *tmp_pi2_dst = (pi2_dst + dst_strd);
                CALCULATE_ROWS_WD8x(tmp_pu2_src + x,      tmp_pi2_dst + x     )
                CALCULATE_ROWS_WD8x(tmp_pu2_src + x + 16, tmp_pi2_dst + x + 16)
            }
            pu2_src += src_strd2;
            pi2_dst += dst_strd2;
        }

        if (ht & 1) {
            for (x = 0; x < (wd << 1); x += 32) {
                CALCULATE_ROWS_WD8x(pu2_src + x,      pi2_dst + x)
                CALCULATE_ROWS_WD8x(pu2_src + x + 16, pi2_dst + x + 16)
            }
        }
    }
    else if (!(wd & 7)) {
        for (y = 0; y < (ht & ~1); y += 2) {
            for (x = 0; x < (wd << 1); x += 16) {
                CALCULATE_ROWS_WD8x(pu2_src     + x,     pi2_dst + x)
                UWORD16 *tmp_pu2_src = (pu2_src + src_strd);
                WORD16  *tmp_pi2_dst = (pi2_dst + dst_strd);
                CALCULATE_ROWS_WD8x(tmp_pu2_src + x, tmp_pi2_dst + x)
            }
            pu2_src += src_strd2;
            pi2_dst += dst_strd2;
        }

        if (ht & 1) {
            for (x = 0; x < (wd << 1); x += 16) {
                CALCULATE_ROWS_WD8x(pu2_src + x, pi2_dst + x)
            }
        }
    }
    else if (!(wd & 3)) {
        for (y = 0; y < ht; y++) {
            for (x = 0; x < (wd << 1); x += 8) {
                CALCULATE_ROW_WD4x(pu2_src + x, pi2_dst + x)
            }
            pu2_src += src_strd;
            pi2_dst += dst_strd;
        }
    }
    else if (!(wd & 1)) {
        for (y = 0; y < ht; y++) {
            for (x = 0; x < (wd << 1); x += 4) {
                CALCULATE_ROW_WD2x(pu2_src + x, pi2_dst + x)
            }
            pu2_src += src_strd;
            pi2_dst += dst_strd;
        }
    }
#undef CALCULATE_ROWS_WD8x
#undef CALCULATE_ROW_WD4x
#undef CALCULATE_ROW_WD2x
}

/**
*******************************************************************************
*
* @brief
*     Interprediction chroma filter to store vertical 16bit ouput
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*    the elements pointed by 'pu2_src' and writes to the location pointed by
*    'pi2_dst'. The output is downshifted by (bit_depth - 8) and is used as
*    an input for weighted prediction.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_vert_w16out_neonintr(UWORD16 *pu2_src,
                                                      WORD16 *pi2_dst,
                                                      WORD32 src_strd,
                                                      WORD32 dst_strd,
                                                      WORD8 *pi1_coeff,
                                                      WORD32 ht,
                                                      WORD32 wd,
                                                      UWORD8 bit_depth)
{
    int32x4_t shift_vec = vdupq_n_s32(8 - bit_depth);
    int8x8_t coeffS1 = vld1_s8(pi1_coeff);
    uint16x8_t coeffU2 = vreinterpretq_u16_s16(vmovl_s8(vabs_s8(coeffS1)));
    pu2_src -= src_strd;
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = src_strd2 + src_strd2;
    WORD32 x, y;

    uint16x8_t src0U2, src1U2, src2U2, src3U2, src4U2, src5U2, src6U2, src7U2, src8U2, src9U2;
    uint16x8_t mul0U2, mul1U2, mul2U2, mul3U2, mul4U2, mul5U2;
    int16x4_t  sft0S2, sft1S2, sft2S2, sft3S2;
    uint32x4_t add0U4, add1U4, add2U4, add3U4;
    int32x4_t  sub0U4, sub1U4, sub2U4, sub3U4;
    uint16x8_t coeff0U2, coeff1U2, coeff2U2, coeff3U2;

#define CALCULATE_ROWS_WD8x(src0, src1, src2, src3,                         \
                             src4, src5, src6, src7, pDst)                  \
        mul0U2 = vmulq_u16(src0, coeff0U2);                                 \
        mul1U2 = vmulq_u16(src1, coeff1U2);                                 \
        mul2U2 = vmulq_u16(src2, coeff2U2);                                 \
        mul0U2 = vmlaq_u16(mul0U2, src3, coeff3U2);                         \
        mul3U2 = vmulq_u16(src4, coeff0U2);                                 \
        mul4U2 = vmulq_u16(src5, coeff1U2);                                 \
        mul5U2 = vmulq_u16(src6, coeff2U2);                                 \
        mul3U2 = vmlaq_u16(mul3U2, src7, coeff3U2);                         \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        add2U4 = vaddl_u16(vget_low_u16(mul4U2), vget_low_u16(mul5U2));     \
        add3U4 = vaddl_u16(vget_high_u16(mul4U2), vget_high_u16(mul5U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sub2U4 = vsubw_s16(vreinterpretq_s32_u32(add2U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul3U2)));    \
        sub3U4 = vsubw_s16(vreinterpretq_s32_u32(add3U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul3U2)));   \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        sft1S2 = vmovn_s32(vshlq_s32(sub1U4, shift_vec));                   \
        sft2S2 = vmovn_s32(vshlq_s32(sub2U4, shift_vec));                   \
        sft3S2 = vmovn_s32(vshlq_s32(sub3U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);                                             \
        vst1_s16(pDst + 4, sft1S2);                                         \
        vst1_s16(pDst + 8, sft2S2);                                         \
        vst1_s16(pDst + 12, sft3S2);

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, pDst)                    \
        mul0U2 = vmulq_u16(src0, coeff0U2);                                 \
        mul1U2 = vmulq_u16(src1, coeff1U2);                                 \
        mul2U2 = vmulq_u16(src2, coeff2U2);                                 \
        mul0U2 = vmlaq_u16(mul0U2, src3, coeff3U2);                         \
        add0U4 = vaddl_u16(vget_low_u16(mul1U2), vget_low_u16(mul2U2));     \
        add1U4 = vaddl_u16(vget_high_u16(mul1U2), vget_high_u16(mul2U2));   \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(vget_low_u16(mul0U2)));    \
        sub1U4 = vsubw_s16(vreinterpretq_s32_u32(add1U4), vget_high_s16(    \
                                         vreinterpretq_s16_u16(mul0U2)));   \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        sft1S2 = vmovn_s32(vshlq_s32(sub1U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);                                             \
        vst1_s16(pDst + 4, sft1S2);

#define CALCULATE_ROW_WD2x(src0, src1, src2, src3, pDst)                    \
        aMul0U2 = vmul_u16(src0, aCoeff0U2);                                \
        aMul1U2 = vmul_u16(src1, aCoeff1U2);                                \
        aMul2U2 = vmul_u16(src2, aCoeff2U2);                                \
        aMul0U2 = vmla_u16(aMul0U2, src3, aCoeff3U2);                       \
        add0U4 = vaddl_u16(aMul1U2, aMul2U2);                               \
        sub0U4 = vsubw_s16(vreinterpretq_s32_u32(add0U4),                   \
                            vreinterpret_s16_u16(aMul0U2));                 \
        sft0S2 = vmovn_s32(vshlq_s32(sub0U4, shift_vec));                   \
        vst1_s16(pDst, sft0S2);                                             \

    if (!(wd & 3)) {
        coeff0U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 0);
        coeff1U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 1);
        coeff2U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 2);
        coeff3U2 = vdupq_lane_u16(vget_low_u16(coeffU2), 3);

        if (!(wd & 15)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 32) {
                    src0U2 = vld1q_u16(pu2_src + x);
                    src1U2 = vld1q_u16(pu2_src + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + x + 8);
                    src5U2 = vld1q_u16(pu2_src + x + 8 + src_strd);
                    src6U2 = vld1q_u16(pu2_src + x + 8 + src_strd2);
                    src7U2 = vld1q_u16(pu2_src + x + 8 + src_strd3);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src9U2 = vld1q_u16(pu2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0U2, src1U2, src2U2, src3U2,
                                         src4U2, src5U2, src6U2, src7U2, pi2_dst + x)
                    CALCULATE_ROWS_WD8x(src1U2, src2U2, src3U2, src8U2,
                                         src5U2, src6U2, src7U2, src9U2,
                                         pi2_dst + dst_strd + x)

                    src0U2 = vld1q_u16(pu2_src + 16 + x);
                    src1U2 = vld1q_u16(pu2_src + 16 + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + 16 + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + 16 + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + 16 + x + 8);
                    src5U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd);
                    src6U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd2);
                    src7U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd3);
                    src8U2 = vld1q_u16(pu2_src + 16 + x + src_strd4);
                    src9U2 = vld1q_u16(pu2_src + 16 + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0U2, src1U2, src2U2, src3U2,
                                         src4U2, src5U2, src6U2, src7U2,
                                         pi2_dst + x + 16)
                    CALCULATE_ROWS_WD8x(src1U2, src2U2, src3U2, src8U2,
                                         src5U2, src6U2, src7U2, src9U2,
                                         pi2_dst + dst_strd + x + 16)
                }
                pu2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        }
        else if (!(wd & 7)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 16) {
                    src0U2 = vld1q_u16(pu2_src + x);
                    src1U2 = vld1q_u16(pu2_src + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + x + 8);
                    src5U2 = vld1q_u16(pu2_src + x + 8 + src_strd);
                    src6U2 = vld1q_u16(pu2_src + x + 8 + src_strd2);
                    src7U2 = vld1q_u16(pu2_src + x + 8 + src_strd3);
                    src8U2 = vld1q_u16(pu2_src + x + src_strd4);
                    src9U2 = vld1q_u16(pu2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0U2, src1U2, src2U2, src3U2,
                                         src4U2, src5U2, src6U2, src7U2, pi2_dst + x)
                    CALCULATE_ROWS_WD8x(src1U2, src2U2, src3U2, src8U2,
                                         src5U2, src6U2, src7U2, src9U2,
                                         pi2_dst + dst_strd + x)
                }
                pu2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        }
        else {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 8) {
                    src0U2 = vld1q_u16(pu2_src + x);
                    src1U2 = vld1q_u16(pu2_src + x + src_strd);
                    src2U2 = vld1q_u16(pu2_src + x + src_strd2);
                    src3U2 = vld1q_u16(pu2_src + x + src_strd3);
                    src4U2 = vld1q_u16(pu2_src + x + src_strd4);
                    CALCULATE_ROW_WD4x(src0U2, src1U2, src2U2, src3U2, pi2_dst + x)
                    CALCULATE_ROW_WD4x(src1U2, src2U2, src3U2, src4U2,
                                       pi2_dst + x + dst_strd)
                }
                pu2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        }
    }
    else if (!(wd & 1)) {
        uint16x4_t aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2, aSrc4U2;
        uint16x4_t aMul0U2, aMul1U2, aMul2U2;
        uint16x4_t aCoeff0U2, aCoeff1U2, aCoeff2U2, aCoeff3U2;
        aCoeff0U2 = vdup_lane_u16(vget_low_u16(coeffU2), 0);
        aCoeff1U2 = vdup_lane_u16(vget_low_u16(coeffU2), 1);
        aCoeff2U2 = vdup_lane_u16(vget_low_u16(coeffU2), 2);
        aCoeff3U2 = vdup_lane_u16(vget_low_u16(coeffU2), 3);

        for (y = 0; y < ht; y += 2) {
            for (x = 0; x < (wd << 1); x += 4) {
                aSrc0U2 = vld1_u16(pu2_src + x);
                aSrc1U2 = vld1_u16(pu2_src + x + src_strd);
                aSrc2U2 = vld1_u16(pu2_src + x + src_strd2);
                aSrc3U2 = vld1_u16(pu2_src + x + src_strd3);
                aSrc4U2 = vld1_u16(pu2_src + x + src_strd4);
                CALCULATE_ROW_WD2x(aSrc0U2, aSrc1U2, aSrc2U2, aSrc3U2, pi2_dst + x)
                CALCULATE_ROW_WD2x(aSrc1U2, aSrc2U2, aSrc3U2, aSrc4U2,
                                   pi2_dst + x + dst_strd)
            }
            pu2_src += src_strd2;
            pi2_dst += dst_strd2;
        }
    }
#undef CALCULATE_ROWS_WD8x
#undef CALCULATE_ROW_WD4x
#undef CALCULATE_ROW_WD2x
}



/**
*******************************************************************************
*
* @brief
*     chroma interprediction filter for vertical 16bit input
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*    the elements pointed by 'pi2_src' and writes to the location pointed by
*    'pu2_dst'. Input is 16 bits. The filter output is downshifted by
*    (14 - bit_depth), rounded by 6, and clipped to bit_depth.
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples
*
* @returns
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_vert_w16inp_neonintr(WORD16 *pi2_src,
                                                      UWORD16 *pu2_dst,
                                                      WORD32 src_strd,
                                                      WORD32 dst_strd,
                                                      WORD8 *pi1_coeff,
                                                      WORD32 ht,
                                                      WORD32 wd,
                                                      UWORD8 bit_depth)
{
    int32x4_t shift_vec = vdupq_n_s32(bit_depth - 14);
    int8x8_t  coeffS1 = vld1_s8(pi1_coeff);
    int16x8_t coeffS2 = vmovl_s8(coeffS1);
    pi2_src -= src_strd;
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = src_strd2 + src_strd2;
    WORD32 x, y;

    int16x8_t src0S2, src1S2, src2S2, src3S2, src4S2, src5S2, src6S2, src7S2, src8S2, src9S2;
    int32x4_t lMul0S4, lMul1S4, hMul0S4, hMul1S4;
    int32x4_t sft0S4, sft1S4, sft2S4, sft3S4;
    uint16x4_t sft0U2, sft1U2, sft2U2, sft3U2;
    uint16x8_t res0U2, res1U2;
    int16x8_t coeff0S2, coeff1S2, coeff2S2, coeff3S2;
    uint16x8_t max_val_16x8 = vdupq_n_u16((1 << bit_depth) - 1);
    uint16x4_t max_val_16x4 = vdup_n_u16((1 << bit_depth) - 1);

#define CALCULATE_ROWS_WD8x(src0, src1, src2, src3,                                      \
                             src4, src5, src6, src7, pDst)                               \
        lMul0S4 = vmull_lane_s16(vget_low_s16(src0), vget_low_s16(coeffS2), 0);          \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src1), vget_low_s16(coeffS2), 1); \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src2), vget_low_s16(coeffS2), 2); \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src3), vget_low_s16(coeffS2), 3); \
        hMul0S4 = vmull_s16(vget_high_s16(src0), vget_high_s16(coeff0S2));               \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src1), vget_high_s16(coeff1S2));      \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src2), vget_high_s16(coeff2S2));      \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src3), vget_high_s16(coeff3S2));      \
        lMul1S4 = vmull_lane_s16(vget_low_s16(src4), vget_low_s16(coeffS2), 0);          \
        lMul1S4 = vmlal_lane_s16(lMul1S4, vget_low_s16(src5), vget_low_s16(coeffS2), 1); \
        lMul1S4 = vmlal_lane_s16(lMul1S4, vget_low_s16(src6), vget_low_s16(coeffS2), 2); \
        lMul1S4 = vmlal_lane_s16(lMul1S4, vget_low_s16(src7), vget_low_s16(coeffS2), 3); \
        hMul1S4 = vmull_s16(vget_high_s16(src4), vget_high_s16(coeff0S2));               \
        hMul1S4 = vmlal_s16(hMul1S4, vget_high_s16(src5), vget_high_s16(coeff1S2));      \
        hMul1S4 = vmlal_s16(hMul1S4, vget_high_s16(src6), vget_high_s16(coeff2S2));      \
        hMul1S4 = vmlal_s16(hMul1S4, vget_high_s16(src7), vget_high_s16(coeff3S2));      \
        sft0S4 = vshlq_s32(lMul0S4, shift_vec);                                          \
        sft1S4 = vshlq_s32(hMul0S4, shift_vec);                                          \
        sft2S4 = vshlq_s32(lMul1S4, shift_vec);                                          \
        sft3S4 = vshlq_s32(hMul1S4, shift_vec);                                          \
        sft0U2 = vqrshrun_n_s32(sft0S4, 6);                                              \
        sft1U2 = vqrshrun_n_s32(sft1S4, 6);                                              \
        sft2U2 = vqrshrun_n_s32(sft2S4, 6);                                              \
        sft3U2 = vqrshrun_n_s32(sft3S4, 6);                                              \
        res0U2 = vcombine_u16(sft0U2, sft1U2);                                           \
        res1U2 = vcombine_u16(sft2U2, sft3U2);                                           \
        res0U2 = vminq_u16(res0U2, max_val_16x8);                                        \
        res1U2 = vminq_u16(res1U2, max_val_16x8);                                        \
        vst1q_u16(pDst, res0U2);                                                         \
        vst1q_u16(pDst + 8, res1U2);

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, pDst)                                 \
        lMul0S4 = vmull_lane_s16(vget_low_s16(src0), vget_low_s16(coeffS2), 0);          \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src1), vget_low_s16(coeffS2), 1); \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src2), vget_low_s16(coeffS2), 2); \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src3), vget_low_s16(coeffS2), 3); \
        hMul0S4 = vmull_s16(vget_high_s16(src0), vget_high_s16(coeff0S2));               \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src1), vget_high_s16(coeff1S2));      \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src2), vget_high_s16(coeff2S2));      \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src3), vget_high_s16(coeff3S2));      \
        sft0S4 = vshlq_s32(lMul0S4, shift_vec);                                          \
        sft1S4 = vshlq_s32(hMul0S4, shift_vec);                                          \
        sft0U2 = vqrshrun_n_s32(sft0S4, 6);                                              \
        sft1U2 = vqrshrun_n_s32(sft1S4, 6);                                              \
        res0U2 = vcombine_u16(sft0U2, sft1U2);                                           \
        res0U2 = vminq_u16(res0U2, max_val_16x8);                                        \
        vst1q_u16(pDst, res0U2);

#define CALCULATE_ROW_WD2x(src0, src1, src2, src3, pDst)                                 \
        lMul0S4 = vmull_lane_s16(src0, vget_low_s16(coeffS2), 0);                        \
        lMul0S4 = vmlal_lane_s16(lMul0S4, src1, vget_low_s16(coeffS2), 1);               \
        lMul0S4 = vmlal_lane_s16(lMul0S4, src2, vget_low_s16(coeffS2), 2);               \
        lMul0S4 = vmlal_lane_s16(lMul0S4, src3, vget_low_s16(coeffS2), 3);               \
        sft0S4 = vshlq_s32(lMul0S4, shift_vec);                                          \
        sft0U2 = vqrshrun_n_s32(sft0S4, 6);                                              \
        sft0U2 = vmin_u16(sft0U2, max_val_16x4);                                         \
        vst1_u16(pDst, sft0U2);

    if (!(wd & 3)) {
        coeff0S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 0);
        coeff1S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 1);
        coeff2S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 2);
        coeff3S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 3);

        if (!(wd & 15)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 32) {
                    src0S2 = vld1q_s16(pi2_src + x);
                    src1S2 = vld1q_s16(pi2_src + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + x + 8);
                    src5S2 = vld1q_s16(pi2_src + x + 8 + src_strd);
                    src6S2 = vld1q_s16(pi2_src + x + 8 + src_strd2);
                    src7S2 = vld1q_s16(pi2_src + x + 8 + src_strd3);
                    src8S2 = vld1q_s16(pi2_src + x + src_strd4);
                    src9S2 = vld1q_s16(pi2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0S2, src1S2, src2S2, src3S2,
                                         src4S2, src5S2, src6S2, src7S2, pu2_dst + x)
                    CALCULATE_ROWS_WD8x(src1S2, src2S2, src3S2, src8S2,
                                         src5S2, src6S2, src7S2, src9S2,
                                         pu2_dst + dst_strd + x)

                    src0S2 = vld1q_s16(pi2_src + 16 + x);
                    src1S2 = vld1q_s16(pi2_src + 16 + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + 16 + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + 16 + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + 16 + x + 8);
                    src5S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd);
                    src6S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd2);
                    src7S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd3);
                    src8S2 = vld1q_s16(pi2_src + 16 + x + src_strd4);
                    src9S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0S2, src1S2, src2S2, src3S2,
                                         src4S2, src5S2, src6S2, src7S2,
                                         pu2_dst + x + 16)
                    CALCULATE_ROWS_WD8x(src1S2, src2S2, src3S2, src8S2,
                                         src5S2, src6S2, src7S2, src9S2,
                                         pu2_dst + dst_strd + x + 16)
                }
                pi2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
        else if (!(wd & 7)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 16) {
                    src0S2 = vld1q_s16(pi2_src + x);
                    src1S2 = vld1q_s16(pi2_src + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + x + 8);
                    src5S2 = vld1q_s16(pi2_src + x + 8 + src_strd);
                    src6S2 = vld1q_s16(pi2_src + x + 8 + src_strd2);
                    src7S2 = vld1q_s16(pi2_src + x + 8 + src_strd3);
                    src8S2 = vld1q_s16(pi2_src + x + src_strd4);
                    src9S2 = vld1q_s16(pi2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0S2, src1S2, src2S2, src3S2,
                                         src4S2, src5S2, src6S2, src7S2, pu2_dst + x)
                    CALCULATE_ROWS_WD8x(src1S2, src2S2, src3S2, src8S2,
                                         src5S2, src6S2, src7S2, src9S2,
                                         pu2_dst + dst_strd + x)
                }
                pi2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
        else {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 8) {
                    src0S2 = vld1q_s16(pi2_src + x);
                    src1S2 = vld1q_s16(pi2_src + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + x + src_strd4);
                    CALCULATE_ROW_WD4x(src0S2, src1S2, src2S2, src3S2, pu2_dst + x)
                    CALCULATE_ROW_WD4x(src1S2, src2S2, src3S2, src4S2,
                                       pu2_dst + x + dst_strd)
                }
                pi2_src += src_strd2;
                pu2_dst += dst_strd2;
            }
        }
    }
    else if (!(wd & 1)) {
        int16x4_t aSrc0S2, aSrc1S2, aSrc2S2, aSrc3S2, aSrc4S2;

        for (y = 0; y < ht; y += 2) {
            for (x = 0; x < (wd << 1); x += 4) {
                aSrc0S2 = vld1_s16(pi2_src + x);
                aSrc1S2 = vld1_s16(pi2_src + x + src_strd);
                aSrc2S2 = vld1_s16(pi2_src + x + src_strd2);
                aSrc3S2 = vld1_s16(pi2_src + x + src_strd3);
                aSrc4S2 = vld1_s16(pi2_src + x + src_strd4);
                CALCULATE_ROW_WD2x(aSrc0S2, aSrc1S2, aSrc2S2, aSrc3S2, pu2_dst + x)
                CALCULATE_ROW_WD2x(aSrc1S2, aSrc2S2, aSrc3S2, aSrc4S2,
                                   pu2_dst + x + dst_strd)
            }
            pi2_src += src_strd2;
            pu2_dst += dst_strd2;
        }
    }
#undef CALCULATE_ROWS_WD8x
#undef CALCULATE_ROW_WD4x
#undef CALCULATE_ROW_WD2x
}

/**
*******************************************************************************
*
* @brief
*
*      Chroma interprediction filter for 16bit vertical input and output.
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to by 'pi1_coeff' to
*    the elements pointed by 'pi2_src' and writes to the location pointed by
*    'pi2_dst'. Input is 16 bits. The filter output is downshifted by 6.
*    The output is used as an input to weighted prediction.
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @param[in] bit_depth
*  bit depth of the samples (unused)
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_vert_w16inp_w16out_neonintr(WORD16 *pi2_src,
                                                             WORD16 *pi2_dst,
                                                             WORD32 src_strd,
                                                             WORD32 dst_strd,
                                                             WORD8 *pi1_coeff,
                                                             WORD32 ht,
                                                             WORD32 wd,
                                                             UWORD8 bit_depth)
{
    UNUSED(bit_depth);
    int8x8_t  coeffS1 = vld1_s8(pi1_coeff);
    int16x8_t coeffS2 = vmovl_s8(coeffS1);
    pi2_src -= src_strd;
    WORD32 src_strd2 = (src_strd << 1);
    WORD32 dst_strd2 = (dst_strd << 1);
    WORD32 src_strd3 = src_strd2 + src_strd;
    WORD32 src_strd4 = src_strd2 + src_strd2;
    WORD32 x, y;

    int16x8_t src0S2, src1S2, src2S2, src3S2, src4S2, src5S2, src6S2, src7S2, src8S2, src9S2;
    int32x4_t lMul0S4, lMul1S4, hMul0S4, hMul1S4;
    int16x4_t sft0S2, sft1S2, sft2S2, sft3S2;
    int16x8_t coeff0S2, coeff1S2, coeff2S2, coeff3S2;

#define CALCULATE_ROWS_WD8x(src0, src1, src2, src3,                                          \
                             src4, src5, src6, src7, pDst)                                   \
        lMul0S4 = vmull_lane_s16(vget_low_s16(src0), vget_low_s16(coeffS2), 0);              \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src1), vget_low_s16(coeffS2), 1);     \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src2), vget_low_s16(coeffS2), 2);     \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src3), vget_low_s16(coeffS2), 3);     \
        hMul0S4 = vmull_s16(vget_high_s16(src0), vget_high_s16(coeff0S2));                   \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src1), vget_high_s16(coeff1S2));          \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src2), vget_high_s16(coeff2S2));          \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src3), vget_high_s16(coeff3S2));          \
        lMul1S4 = vmull_lane_s16(vget_low_s16(src4), vget_low_s16(coeffS2), 0);              \
        lMul1S4 = vmlal_lane_s16(lMul1S4, vget_low_s16(src5), vget_low_s16(coeffS2), 1);     \
        lMul1S4 = vmlal_lane_s16(lMul1S4, vget_low_s16(src6), vget_low_s16(coeffS2), 2);     \
        lMul1S4 = vmlal_lane_s16(lMul1S4, vget_low_s16(src7), vget_low_s16(coeffS2), 3);     \
        hMul1S4 = vmull_s16(vget_high_s16(src4), vget_high_s16(coeff0S2));                   \
        hMul1S4 = vmlal_s16(hMul1S4, vget_high_s16(src5), vget_high_s16(coeff1S2));          \
        hMul1S4 = vmlal_s16(hMul1S4, vget_high_s16(src6), vget_high_s16(coeff2S2));          \
        hMul1S4 = vmlal_s16(hMul1S4, vget_high_s16(src7), vget_high_s16(coeff3S2));          \
        sft0S2 = vshrn_n_s32(lMul0S4, 6);                                                    \
        sft1S2 = vshrn_n_s32(hMul0S4, 6);                                                    \
        sft2S2 = vshrn_n_s32(lMul1S4, 6);                                                    \
        sft3S2 = vshrn_n_s32(hMul1S4, 6);                                                    \
        vst1_s16(pDst,      sft0S2);                                                         \
        vst1_s16(pDst + 4,  sft1S2);                                                         \
        vst1_s16(pDst + 8,  sft2S2);                                                         \
        vst1_s16(pDst + 12, sft3S2);

#define CALCULATE_ROW_WD4x(src0, src1, src2, src3, pDst)                                     \
        lMul0S4 = vmull_lane_s16(vget_low_s16(src0), vget_low_s16(coeffS2), 0);              \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src1), vget_low_s16(coeffS2), 1);     \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src2), vget_low_s16(coeffS2), 2);     \
        lMul0S4 = vmlal_lane_s16(lMul0S4, vget_low_s16(src3), vget_low_s16(coeffS2), 3);     \
        hMul0S4 = vmull_s16(vget_high_s16(src0), vget_high_s16(coeff0S2));                   \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src1), vget_high_s16(coeff1S2));          \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src2), vget_high_s16(coeff2S2));          \
        hMul0S4 = vmlal_s16(hMul0S4, vget_high_s16(src3), vget_high_s16(coeff3S2));          \
        sft0S2  = vshrn_n_s32(lMul0S4, 6);                                                   \
        sft1S2  = vshrn_n_s32(hMul0S4, 6);                                                   \
        vst1_s16(pDst,      sft0S2);                                                         \
        vst1_s16(pDst + 4,  sft1S2);

#define CALCULATE_ROW_WD2x(src0, src1, src2, src3, pDst)                                     \
        lMul0S4 = vmull_lane_s16(src0, vget_low_s16(coeffS2), 0);                            \
        lMul0S4 = vmlal_lane_s16(lMul0S4, src1, vget_low_s16(coeffS2), 1);                   \
        lMul0S4 = vmlal_lane_s16(lMul0S4, src2, vget_low_s16(coeffS2), 2);                   \
        lMul0S4 = vmlal_lane_s16(lMul0S4, src3, vget_low_s16(coeffS2), 3);                   \
        sft0S2  = vshrn_n_s32(lMul0S4, 6);                                                   \
        vst1_s16(pDst, sft0S2);

    if (!(wd & 3)) {
        coeff0S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 0);
        coeff1S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 1);
        coeff2S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 2);
        coeff3S2 = vdupq_lane_s16(vget_low_s16(coeffS2), 3);

        if (!(wd & 15)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 32) {
                    src0S2 = vld1q_s16(pi2_src + x);
                    src1S2 = vld1q_s16(pi2_src + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + x + 8);
                    src5S2 = vld1q_s16(pi2_src + x + 8 + src_strd);
                    src6S2 = vld1q_s16(pi2_src + x + 8 + src_strd2);
                    src7S2 = vld1q_s16(pi2_src + x + 8 + src_strd3);
                    src8S2 = vld1q_s16(pi2_src + x + src_strd4);
                    src9S2 = vld1q_s16(pi2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0S2, src1S2, src2S2, src3S2,
                                         src4S2, src5S2, src6S2, src7S2, pi2_dst + x)
                    CALCULATE_ROWS_WD8x(src1S2, src2S2, src3S2, src8S2,
                                         src5S2, src6S2, src7S2, src9S2,
                                         pi2_dst + dst_strd + x)

                    src0S2 = vld1q_s16(pi2_src + 16 + x);
                    src1S2 = vld1q_s16(pi2_src + 16 + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + 16 + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + 16 + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + 16 + x + 8);
                    src5S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd);
                    src6S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd2);
                    src7S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd3);
                    src8S2 = vld1q_s16(pi2_src + 16 + x + src_strd4);
                    src9S2 = vld1q_s16(pi2_src + 16 + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0S2, src1S2, src2S2, src3S2,
                                         src4S2, src5S2, src6S2, src7S2,
                                         pi2_dst + x + 16)
                    CALCULATE_ROWS_WD8x(src1S2, src2S2, src3S2, src8S2,
                                         src5S2, src6S2, src7S2, src9S2,
                                         pi2_dst + dst_strd + x + 16)
                }
                pi2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        } else if (!(wd & 7)) {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 16) {
                    src0S2 = vld1q_s16(pi2_src + x);
                    src1S2 = vld1q_s16(pi2_src + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + x + 8);
                    src5S2 = vld1q_s16(pi2_src + x + 8 + src_strd);
                    src6S2 = vld1q_s16(pi2_src + x + 8 + src_strd2);
                    src7S2 = vld1q_s16(pi2_src + x + 8 + src_strd3);
                    src8S2 = vld1q_s16(pi2_src + x + src_strd4);
                    src9S2 = vld1q_s16(pi2_src + x + 8 + src_strd4);
                    CALCULATE_ROWS_WD8x(src0S2, src1S2, src2S2, src3S2,
                                         src4S2, src5S2, src6S2, src7S2, pi2_dst + x)
                    CALCULATE_ROWS_WD8x(src1S2, src2S2, src3S2, src8S2,
                                         src5S2, src6S2, src7S2, src9S2,
                                         pi2_dst + dst_strd + x)
                }
                pi2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        } else {
            for (y = 0; y < ht; y += 2) {
                for (x = 0; x < (wd << 1); x += 8) {
                    src0S2 = vld1q_s16(pi2_src + x);
                    src1S2 = vld1q_s16(pi2_src + x + src_strd);
                    src2S2 = vld1q_s16(pi2_src + x + src_strd2);
                    src3S2 = vld1q_s16(pi2_src + x + src_strd3);
                    src4S2 = vld1q_s16(pi2_src + x + src_strd4);
                    CALCULATE_ROW_WD4x(src0S2, src1S2, src2S2, src3S2, pi2_dst + x)
                    CALCULATE_ROW_WD4x(src1S2, src2S2, src3S2, src4S2,
                                       pi2_dst + x + dst_strd)
                }
                pi2_src += src_strd2;
                pi2_dst += dst_strd2;
            }
        }
    } else if (!(wd & 1)) {
        int16x4_t aSrc0S2, aSrc1S2, aSrc2S2, aSrc3S2, aSrc4S2;

        for (y = 0; y < ht; y += 2) {
            for (x = 0; x < (wd << 1); x += 4) {
                aSrc0S2 = vld1_s16(pi2_src + x);
                aSrc1S2 = vld1_s16(pi2_src + x + src_strd);
                aSrc2S2 = vld1_s16(pi2_src + x + src_strd2);
                aSrc3S2 = vld1_s16(pi2_src + x + src_strd3);
                aSrc4S2 = vld1_s16(pi2_src + x + src_strd4);
                CALCULATE_ROW_WD2x(aSrc0S2, aSrc1S2, aSrc2S2, aSrc3S2, pi2_dst + x)
                CALCULATE_ROW_WD2x(aSrc1S2, aSrc2S2, aSrc3S2, aSrc4S2,
                                   pi2_dst + x + dst_strd)
            }
            pi2_src += src_strd2;
            pi2_dst += dst_strd2;
        }
    }
#undef CALCULATE_ROWS_WD8x
#undef CALCULATE_ROW_WD4x
#undef CALCULATE_ROW_WD2x
}
