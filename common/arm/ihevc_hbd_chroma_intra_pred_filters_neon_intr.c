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
*  ihevc_hbd_chroma_intra_pred_filters_neon_intr.c
*
* @brief
*  Contains function definitions for high bit depth (HBD) chroma intra
*  prediction interpolation filters using ARM NEON intrinsics.
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_intra_pred_chroma_ref_filtering_neonintr()
*  - ihevc_hbd_intra_pred_chroma_ref_substitution_neonintr()
*  - ihevc_hbd_intra_pred_chroma_planar_neonintr()
*  - ihevc_hbd_intra_pred_chroma_dc_neonintr()
*  - ihevc_hbd_intra_pred_chroma_horz_neonintr()
*  - ihevc_hbd_intra_pred_chroma_ver_neonintr()
*  - ihevc_hbd_intra_pred_chroma_mode2_neonintr()
*  - ihevc_hbd_intra_pred_chroma_mode_18_34_neonintr()
*  - ihevc_hbd_intra_pred_chroma_mode_3_to_9_neonintr()
*  - ihevc_hbd_intra_pred_chroma_mode_11_to_17_neonintr()
*  - ihevc_hbd_intra_pred_chroma_mode_19_to_25_neonintr()
*  - ihevc_hbd_intra_pred_chroma_mode_27_to_33_neonintr()
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
#include "ihevc_intra_pred.h"
#include "ihevc_chroma_intra_pred.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_common_tables.h"

/*****************************************************************************/
/* Constant Macros                                                           */
/*****************************************************************************/
#define MAX_CU_SIZE 64
#define T16C_4NT 64
#define T8C_4NT 32

static const UWORD16 au2_chroma_planar_factor[66] = {
     0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
     8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
    16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23,
    24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31,
    32, 32
};

/*****************************************************************************/
/* Function Definitions                                                      */
/*****************************************************************************/

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for chroma ref_filtering (4:4:4)
*
* @par Description:
*    Reference DC filtering for neighboring chroma samples dependent on TU size
*    and mode. Refer to section 8.4.4.2.3 in the standard.
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] nt
*  WORD32 Transform Block size
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] mode
*  WORD32 intraprediction mode
*
* @param[in] intra_smoothing_flags
*  WORD32 bit 3 indicates if intra smoothing is enabled/disabled
*         unconditionally (applicable to RExt profiles only);
*         bit 0 indicates strong intra smoothing enabled/disabled
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_ref_filtering_neonintr(UWORD16 *pu2_src,
                                                        WORD32 nt,
                                                        UWORD16 *pu2_dst,
                                                        WORD32 mode,
                                                        WORD32 intra_smoothing_flags)
{
    WORD32 filter_flag;
    WORD32 i = 0;
    WORD32 four_nt = 4 * nt;
    WORD32 two_four_nt = 2 * four_nt;

    UWORD16 src_0nt_u, src_0nt_v;
    UWORD16 src_4nt_u, src_4nt_v;
    UWORD16 *pu2_src_tmp_0 = pu2_src;
    UWORD16 *pu2_src_tmp_1;
    UWORD16 *pu2_src_tmp_2;
    UWORD16 *pu2_dst_tmp_0 = pu2_dst;

    uint16x8_t src_val_0, src_val_1, src_val_2;
    uint16x8_t shift_res;
    uint16x8_t dup_const_2;
    uint16x8_t mul_res, add_res;
    WORD32 intra_smoothing_disabled_flag = (intra_smoothing_flags >> 3) & 0x1;
    WORD32 strong_intra_smoothing_enable_flag = intra_smoothing_flags & 0x1;
    UNUSED(strong_intra_smoothing_enable_flag);

    shift_res = vdupq_n_u16(0);
    filter_flag = intra_smoothing_disabled_flag ?
                    0 : (gau1_intra_pred_ref_filter[mode] & (1 << (CTZ(nt) - 2)));
    if(0 == filter_flag)
    {
        if(pu2_src == pu2_dst)
        {
            return;
        }
        else
        {
            for(i = two_four_nt; i > 0; i -= 8)
            {
                src_val_0 = vld1q_u16(pu2_src_tmp_0);
                pu2_src_tmp_0 += 8;
                vst1q_u16(pu2_dst_tmp_0, src_val_0);
                pu2_dst_tmp_0 += 8;
            }
            pu2_dst[two_four_nt] = pu2_src[two_four_nt];
            pu2_dst[two_four_nt + 1] = pu2_src[two_four_nt + 1];
        }
    }
    else
    {
        src_0nt_u = pu2_src[0];
        src_0nt_v = pu2_src[1];
        src_4nt_u = pu2_src[two_four_nt];
        src_4nt_v = pu2_src[two_four_nt + 1];

        pu2_src_tmp_1 = pu2_src + 2;
        pu2_src_tmp_2 = pu2_src + 4;
        pu2_dst_tmp_0 = pu2_dst + 2;

        dup_const_2 = vdupq_n_u16(2);

        /* Extremities Untouched */
        pu2_dst[0] = src_0nt_u;
        pu2_dst[1] = src_0nt_v;

        /* Perform bilinear filtering of Reference Samples */
        for(i = (two_four_nt - 8); i > 0; i -= 8)
        {
            src_val_0 = vld1q_u16(pu2_src_tmp_0);
            pu2_src_tmp_0 += 8;

            src_val_2 = vld1q_u16(pu2_src_tmp_2);
            pu2_src_tmp_2 += 8;

            src_val_1 = vld1q_u16(pu2_src_tmp_1);
            pu2_src_tmp_1 += 8;

            /* Delay storing the previous iteration's result until after the current
             * iteration's loads so in-place filtering (pu2_src == pu2_dst) does not
             * overwrite overlapping input samples before they are read. */
            if(i < (two_four_nt - 8))
            {
                vst1q_u16(pu2_dst_tmp_0, shift_res);
                pu2_dst_tmp_0 += 8;
            }

            add_res = vaddq_u16(src_val_0, src_val_2);
            mul_res = vmlaq_u16(add_res, src_val_1, dup_const_2);
            shift_res = vrshrq_n_u16(mul_res, 2);
        }

        /* Final 8-element window aligned to end at index (2 * four_nt - 1) */
        src_val_0 = vld1q_u16(pu2_src + two_four_nt - 10);
        src_val_2 = vld1q_u16(pu2_src + two_four_nt - 6);
        src_val_1 = vld1q_u16(pu2_src + two_four_nt - 8);

        vst1q_u16(pu2_dst_tmp_0, shift_res);

        add_res = vaddq_u16(src_val_0, src_val_2);
        mul_res = vmlaq_u16(add_res, src_val_1, dup_const_2);
        shift_res = vrshrq_n_u16(mul_res, 2);

        vst1q_u16(pu2_dst + two_four_nt - 8, shift_res);

        pu2_dst[two_four_nt] = src_4nt_u;
        pu2_dst[two_four_nt + 1] = src_4nt_v;
    }
}

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for chroma ref substitution
*
* @par Description:
*    Reference substitution process for chroma samples unavailable for
*    prediction. Refer to section 8.4.4.2.2 in the standard.
*
* @param[in] pu2_top_left
*  UWORD16 pointer to the top-left
*
* @param[in] pu2_top
*  UWORD16 pointer to the top
*
* @param[in] pu2_left
*  UWORD16 pointer to the left
*
* @param[in] src_strd
*  WORD32 Source stride
*
* @param[in] nt
*  WORD32 Transform Block size
*
* @param[in] nbr_flags
*  WORD32 neighbor availability flags
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 Destination stride
*
* @param[in] chroma_format_idc
*  WORD32 Chroma format indicator
*
* @param[in] bit_depth
*  UWORD8 bit depth of pixel
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_ref_substitution_neonintr(UWORD16 *pu2_top_left,
                                                           UWORD16 *pu2_top,
                                                           UWORD16 *pu2_left,
                                                           WORD32 src_strd,
                                                           WORD32 nt,
                                                           WORD32 nbr_flags,
                                                           UWORD16 *pu2_dst,
                                                           WORD32 dst_strd,
                                                           WORD32 chroma_format_idc,
                                                           UWORD8 bit_depth)
{
    UWORD16 pu2_ref_u, pu2_ref_v;
    WORD32 dc_val, i, j;
    WORD32 get_bits;
    WORD32 next;
    WORD32 bot_left, left, top, tp_right, tp_left;
    WORD32 idx, nbr_id_from_bl, frwd_nbr_flag;
    WORD32 a_nbr_flag[5];
    uint16x8_t dup_uv;
    UNUSED(dst_strd);

    /* Neighbor Flag Structure */
    /* WORD32 nbr_flags MSB-->LSB   TOP LEFT | TOP-RIGHT |  TOP   | LEFT    | BOTTOM LEFT */
    /*                              (1 bit)     (4 bits)  (4 bits) (4 bits)  (4 bits)  */
    /* If no neighbor flags are present, fill the neighbor samples with DC value */
    dc_val = 1 << (bit_depth - 1);
    if(nbr_flags == 0)
    {
        uint16x8_t dc_val_u16 = vdupq_n_u16((UWORD16)dc_val);
        for(i = 0; i < (8 * nt); i += 8)
        {
            vst1q_u16(&pu2_dst[i], dc_val_u16);
        }
        pu2_dst[8 * nt] = dc_val;
        pu2_dst[(8 * nt) + 1] = dc_val;
    }
    else
    {
        /* Else fill the corresponding samples */

        /* Check for the neighbors availability */
        tp_left     = (nbr_flags & 0x10000);
        tp_right    = (nbr_flags & 0x0f000);
        top         = (nbr_flags & 0x00f00);
        left        = (nbr_flags & 0x000f0);
        bot_left    = (nbr_flags & 0x0000f);

        /* Fill nbrs depending on availability */
        /* Top - Left nbrs */
        if(0 != tp_left)
        {
            pu2_dst[(4 * nt)] = *pu2_top_left;
            pu2_dst[(4 * nt) + 1] = *(pu2_top_left + 1);
        }
        /* Left nbrs */
        if(0 != left)
        {
            for(i = 0, j = 0; i < (2 * nt); i += 2)
            {
                pu2_dst[(4 * nt) - 2 - i] = pu2_left[j * src_strd];
                pu2_dst[(4 * nt) - 1 - i] = pu2_left[(j * src_strd) + 1];
                j++;
            }
        }
        /* Bottom - Left nbrs */
        if(0 != bot_left)
        {
            for(i = (2 * nt), j = nt; i < (4 * nt); i += 2)
            {
                pu2_dst[(4 * nt) - 2 - i] = pu2_left[j * src_strd];
                pu2_dst[(4 * nt) - 1 - i] = pu2_left[(j * src_strd) + 1];
                j++;
            }
        }
        /* Top nbrs */
        if(0 != top)
        {
            for(i = 0; i < (2 * nt); i += 8)
            {
                vst1q_u16(&pu2_dst[(4 * nt) + 2 + i], vld1q_u16(&pu2_top[i]));
            }
        }
        /* Top - Right nbrs */
        if(0 != tp_right)
        {
            for(i = (2 * nt); i < (4 * nt); i += 8)
            {
                vst1q_u16(&pu2_dst[(4 * nt) + 2 + i], vld1q_u16(&pu2_top[i]));
            }
        }

        if(nt == 4 || (nt == 8 && chroma_format_idc == CHROMA_FMT_IDC_YUV444))
        {
            /* 1 bit extraction for all the neighboring blocks */
            tp_left = (nbr_flags & 0x10000) >> 16;
            bot_left = (nbr_flags & 0x8) >> 3;
            left = (nbr_flags & 0x80) >> 7;
            top = (nbr_flags & 0x100) >> 8;
            tp_right = (nbr_flags & 0x1000) >> 12;

            next = 1;
            a_nbr_flag[0] = bot_left;
            a_nbr_flag[1] = left;
            a_nbr_flag[2] = tp_left;
            a_nbr_flag[3] = top;
            a_nbr_flag[4] = tp_right;

            /* If bottom-left is not available, reverse substitution process */
            if(bot_left == 0)
            {
                /* Check for the 1st available sample from bottom-left */
                while(!a_nbr_flag[next])
                    next++;

                /* If Left, top-left are available */
                if(next <= 2)
                {
                    idx = (nt * next);
                }
                else /* If top, top-right are available */
                {
                    /* Idx is changed to copy 1 pixel value for top-left, if top-left is not available */
                    idx = (nt * (next - 1)) + 1;
                }

                pu2_ref_u = pu2_dst[2 * idx];
                pu2_ref_v = pu2_dst[(2 * idx) + 1];
                dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                for(i = 0; i + 8 <= (2 * idx); i += 8)
                {
                    vst1q_u16(&pu2_dst[i], dup_uv);
                }
                if(i < (2 * idx))
                {
                    pu2_dst[i] = pu2_ref_u;
                    pu2_dst[i + 1] = pu2_ref_v;
                }
            }

            /* Forward Substitution Process */
            if(left == 0)
            {
                pu2_ref_u = pu2_dst[(2 * nt) - 2];
                pu2_ref_v = pu2_dst[(2 * nt) - 1];
                dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                if(0 == (nt & 7))
                {
                    vst1q_u16(pu2_dst + (2 * nt), dup_uv);
                    vst1q_u16(pu2_dst + (2 * nt) + 8, dup_uv);
                }
                else
                {
                    vst1q_u16(pu2_dst + (2 * nt), dup_uv);
                }
            }
            if(tp_left == 0)
            {
                pu2_dst[4 * nt] = pu2_dst[(4 * nt) - 2];
                pu2_dst[(4 * nt) + 1] = pu2_dst[(4 * nt) - 1];
            }
            if(top == 0)
            {
                pu2_ref_u = pu2_dst[(4 * nt)];
                pu2_ref_v = pu2_dst[(4 * nt) + 1];
                dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                if(0 == (nt & 7))
                {
                    vst1q_u16(pu2_dst + (4 * nt) + 2, dup_uv);
                    vst1q_u16(pu2_dst + (4 * nt) + 10, dup_uv);
                }
                else
                {
                    vst1q_u16(pu2_dst + (4 * nt) + 2, dup_uv);
                }
            }
            if(tp_right == 0)
            {
                pu2_ref_u = pu2_dst[(6 * nt)];
                pu2_ref_v = pu2_dst[(6 * nt) + 1];
                dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                if(0 == (nt & 7))
                {
                    vst1q_u16(pu2_dst + (6 * nt) + 2, dup_uv);
                    vst1q_u16(pu2_dst + (6 * nt) + 10, dup_uv);
                }
                else
                {
                    vst1q_u16(pu2_dst + (6 * nt) + 2, dup_uv);
                }
            }
        }
        else if(nt == 8 || (nt == 16 && chroma_format_idc == CHROMA_FMT_IDC_YUV444))
        {
            WORD32 nbr_flags_temp = 0;
            WORD32 sub_sample = chroma_format_idc == CHROMA_FMT_IDC_YUV444 ? 2 : 1;
            nbr_flags_temp = ((nbr_flags & 0xC) >> 2) + ((nbr_flags & 0xC0) >> 4)
                            + ((nbr_flags & 0x300) >> 4)
                            + ((nbr_flags & 0x3000) >> 6)
                            + ((nbr_flags & 0x10000) >> 8);

            /* compute trailing zeros based on nbr_flag for substitution process of below left */
            /* as each bit in nbr flags corresponds to 4*sub_sample pels for bot_left, left, top and topright but 1 pel for topleft */
            {
                nbr_id_from_bl = look_up_trailing_zeros(nbr_flags_temp & 0XF) * (4 * sub_sample);
                if(nbr_id_from_bl == (32 * sub_sample))
                {
                    nbr_id_from_bl = (16 * sub_sample);
                }
                if(nbr_id_from_bl == (16 * sub_sample))
                {
                    /* for top left : 1 pel per nbr bit */
                    if(!((nbr_flags_temp >> 8) & 0x1))
                    {
                        nbr_id_from_bl++;
                        nbr_id_from_bl += look_up_trailing_zeros((nbr_flags_temp >> 4) & 0xF) * (4 * sub_sample);
                    }
                }
                /* Reverse Substitution Process */
                if(nbr_id_from_bl)
                {
                    /* Replicate the bottom-left and subsequent unavailable pixels with the 1st available pixel above */
                    pu2_ref_u = pu2_dst[2 * nbr_id_from_bl];
                    pu2_ref_v = pu2_dst[(2 * nbr_id_from_bl) + 1];
                    dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                    for(i = 0; i + 8 <= (2 * nbr_id_from_bl); i += 8)
                    {
                        vst1q_u16(&pu2_dst[i], dup_uv);
                    }
                    if(i < (2 * nbr_id_from_bl))
                    {
                        pu2_dst[i] = pu2_ref_u;
                        pu2_dst[i + 1] = pu2_ref_v;
                    }
                }
            }

            /* for the loop of 4*Nt+1 pixels (excluding pixels computed from reverse substitution) */
            while(nbr_id_from_bl < ((T8C_4NT * sub_sample) + 1))
            {
                /* To Obtain the next unavailable idx flag after reverse neighbor substitution */
                frwd_nbr_flag = (nbr_id_from_bl >> (chroma_format_idc == CHROMA_FMT_IDC_YUV444 ? 3 : 2));

                /* The Top-left flag is at the last bit location of nbr_flags */
                if(nbr_id_from_bl == (T8C_4NT * sub_sample / 2))
                {
                    get_bits = GET_BIT(nbr_flags_temp, 8);

                    /* only pel substitution for TL */
                    if(!get_bits)
                    {
                        pu2_dst[2 * nbr_id_from_bl] = pu2_dst[(2 * nbr_id_from_bl) - 2];
                        pu2_dst[(2 * nbr_id_from_bl) + 1] = pu2_dst[(2 * nbr_id_from_bl) - 1];
                    }
                }
                else
                {
                    get_bits = GET_BIT(nbr_flags_temp, frwd_nbr_flag);
                    if(!get_bits)
                    {
                        /* 4*sub_sample pel substitution (other than TL) */
                        pu2_ref_u = pu2_dst[(2 * nbr_id_from_bl) - 2];
                        pu2_ref_v = pu2_dst[(2 * nbr_id_from_bl) - 1];
                        dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                        for(i = 0; i < (8 * sub_sample); i += 8)
                        {
                            vst1q_u16(&pu2_dst[(2 * nbr_id_from_bl) + i], dup_uv);
                        }
                    }
                }
                nbr_id_from_bl += (nbr_id_from_bl == (T8C_4NT * sub_sample / 2)) ? 1 : 4 * sub_sample;
            }
        }
        else if(nt == 16 || (nt == 32 && chroma_format_idc == CHROMA_FMT_IDC_YUV444))
        {
            WORD32 sub_sample = chroma_format_idc == CHROMA_FMT_IDC_YUV444 ? 2 : 1;
            /* compute trailing zeros based on nbr_flag for substitution process of below left */
            /* as each bit in nbr flags corresponds to 4*sub_sample pels for bot_left, left, top and topright but 1 pel for topleft */
            {
                nbr_id_from_bl = look_up_trailing_zeros((nbr_flags & 0XFF)) * 4 * sub_sample;

                if(nbr_id_from_bl == 32 * sub_sample)
                {
                    /* for top left : 1 pel per nbr bit */
                    if(!((nbr_flags >> 16) & 0x1))
                    {
                        /* top left not available */
                        nbr_id_from_bl++;
                        /* top and top right; 4*sub_sample pels per nbr bit */
                        nbr_id_from_bl += look_up_trailing_zeros((nbr_flags >> 8) & 0xFF) * 4 * sub_sample;
                    }
                }
                /* Reverse Substitution Process */
                if(nbr_id_from_bl)
                {
                    /* Replicate the bottom-left and subsequent unavailable pixels with the 1st available pixel above */
                    pu2_ref_u = pu2_dst[2 * nbr_id_from_bl];
                    pu2_ref_v = pu2_dst[2 * nbr_id_from_bl + 1];
                    dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                    for(i = 0; i + 8 <= (2 * nbr_id_from_bl); i += 8)
                    {
                        vst1q_u16(&pu2_dst[i], dup_uv);
                    }
                    if(i < (2 * nbr_id_from_bl))
                    {
                        pu2_dst[i] = pu2_ref_u;
                        pu2_dst[i + 1] = pu2_ref_v;
                    }
                }
            }

            /* for the loop of 4*Nt+1 pixels (excluding pixels computed from reverse substitution) */
            while(nbr_id_from_bl < ((T16C_4NT * sub_sample) + 1))
            {
                /* To Obtain the next unavailable idx flag after reverse neighbor substitution */
                frwd_nbr_flag = (nbr_id_from_bl >> (chroma_format_idc == CHROMA_FMT_IDC_YUV444 ? 3 : 2));

                /* The Top-left flag is at the last bit location of nbr_flags */
                if(nbr_id_from_bl == (T16C_4NT * sub_sample / 2))
                {
                    get_bits = GET_BIT(nbr_flags, 16);
                    /* only pel substitution for TL */
                    if(!get_bits)
                    {
                        pu2_dst[2 * nbr_id_from_bl] = pu2_dst[(2 * nbr_id_from_bl) - 2];
                        pu2_dst[(2 * nbr_id_from_bl) + 1] = pu2_dst[(2 * nbr_id_from_bl) - 1];
                    }
                }
                else
                {
                    get_bits = GET_BIT(nbr_flags, frwd_nbr_flag);
                    if(!get_bits)
                    {
                        /* 4*sub_sample pel substitution (other than TL) */
                        pu2_ref_u = pu2_dst[(2 * nbr_id_from_bl) - 2];
                        pu2_ref_v = pu2_dst[(2 * nbr_id_from_bl) - 1];
                        dup_uv = vreinterpretq_u16_u32(vdupq_n_u32(((UWORD32)pu2_ref_v << 16) | pu2_ref_u));
                        for(i = 0; i < (8 * sub_sample); i += 8)
                        {
                            vst1q_u16(&pu2_dst[(2 * nbr_id_from_bl) + i], dup_uv);
                        }
                    }
                }
                nbr_id_from_bl += (nbr_id_from_bl == (T16C_4NT * sub_sample / 2)) ? 1 : 4 * sub_sample;
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for chroma planar
*
* @par Description:
*   Planar intraprediction with reference neighboring samples location
*   pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'.
*   Refer to section 8.4.4.2.4 in the standard.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the source
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size
*
* @param[in] mode
*  WORD32 intraprediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_planar_neonintr(UWORD16 *pu2_ref,
                                                 WORD32 src_strd,
                                                 UWORD16 *pu2_dst,
                                                 WORD32 dst_strd,
                                                 WORD32 nt,
                                                 WORD32 mode)
{
    WORD32 row, col = 0;
    WORD32 log2nt_plus1 = 32 - CLZ(nt);
    WORD32 two_nt, three_nt;
    UWORD16 *pu2_ref_two_nt_1;
    UWORD16 *pu2_ref_left;
    UWORD16 *pu2_dst_r0, *pu2_dst_r1, *pu2_dst_r2, *pu2_dst_r3;
    const UWORD16 *pu2_col_1;
    uint16x8_t const_nt_1_col_t, const_col_1_t;
    uint16x8_t const_nt_1_r0_dup, const_nt_1_r1_dup;
    uint16x8_t const_nt_1_r2_dup, const_nt_1_r3_dup;
    uint16_t const_nt = nt;
    uint16x8_t const_nt_dup;
    UWORD32 u4_ref_nt_1;
    uint16x8_t pu2_ref_nt_1_dup;
    UWORD32 u4_ref_three_nt_1;
    uint16x8_t ref_left_r0_dup, ref_left_r1_dup, ref_left_r2_dup, ref_left_r3_dup;
    uint16x8_t pu2_ref_two_nt_1_t;
    uint16x8_t pu2_ref_three_nt_1_dup;
    uint16x8_t row_base_r0, row_base_r1, row_base_r2, row_base_r3;
    uint16x8_t prod_col_tr;
    uint16x8_t acc_r0, acc_r1, acc_r2, acc_r3;
    int16x8_t log2nt_dup;
    UNUSED(src_strd);
    UNUSED(mode);

    two_nt = 2 * nt;
    three_nt = 3 * nt;
    pu2_ref_left = pu2_ref + 4 * nt - 8;

    u4_ref_nt_1 = ((UWORD32)pu2_ref[2 * (nt - 1) + 1] << 16) |
                  pu2_ref[2 * (nt - 1)];
    pu2_ref_nt_1_dup = vreinterpretq_u16_u32(vdupq_n_u32(u4_ref_nt_1));

    u4_ref_three_nt_1 = ((UWORD32)pu2_ref[2 * (three_nt + 1) + 1] << 16) |
                        pu2_ref[2 * (three_nt + 1)];
    pu2_ref_three_nt_1_dup = vreinterpretq_u16_u32(vdupq_n_u32(u4_ref_three_nt_1));

    const_nt_dup = vdupq_n_u16(const_nt);

    log2nt_dup = vdupq_n_s16(log2nt_plus1);
    log2nt_dup = vnegq_s16(log2nt_dup);

    for(row = 0; row < nt; row += 4)
    {
        uint32x4_t ref_left_v = vreinterpretq_u32_u16(vld1q_u16(pu2_ref_left));
        pu2_ref_left -= 8;

        ref_left_r0_dup = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_left_v), 1));
        ref_left_r1_dup = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_left_v), 0));
        ref_left_r2_dup = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_left_v), 1));
        ref_left_r3_dup = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_left_v), 0));

        const_nt_1_r0_dup = vdupq_n_u16(nt - 1 - row);
        const_nt_1_r1_dup = vdupq_n_u16(nt - 2 - row);
        const_nt_1_r2_dup = vdupq_n_u16(nt - 3 - row);
        const_nt_1_r3_dup = vdupq_n_u16(nt - 4 - row);

        row_base_r0 = vmlaq_u16(const_nt_dup, vdupq_n_u16(row + 1), pu2_ref_nt_1_dup);
        row_base_r1 = vaddq_u16(row_base_r0, pu2_ref_nt_1_dup);
        row_base_r2 = vaddq_u16(row_base_r1, pu2_ref_nt_1_dup);
        row_base_r3 = vaddq_u16(row_base_r2, pu2_ref_nt_1_dup);

        pu2_col_1 = au2_chroma_planar_factor + 2;
        pu2_ref_two_nt_1 = pu2_ref + 2 * (two_nt + 1);

        pu2_dst_r0 = pu2_dst + row * dst_strd;
        pu2_dst_r1 = pu2_dst_r0 + dst_strd;
        pu2_dst_r2 = pu2_dst_r1 + dst_strd;
        pu2_dst_r3 = pu2_dst_r2 + dst_strd;

        for(col = two_nt; col > 0; col -= 8)
        {
            const_col_1_t = vld1q_u16(pu2_col_1);
            pu2_col_1 += 8;
            const_nt_1_col_t = vsubq_u16(const_nt_dup, const_col_1_t);

            pu2_ref_two_nt_1_t = vld1q_u16(pu2_ref_two_nt_1);
            pu2_ref_two_nt_1 += 8;

            prod_col_tr = vmulq_u16(const_col_1_t, pu2_ref_three_nt_1_dup);

            acc_r0 = vaddq_u16(row_base_r0, prod_col_tr);
            acc_r1 = vaddq_u16(row_base_r1, prod_col_tr);
            acc_r2 = vaddq_u16(row_base_r2, prod_col_tr);
            acc_r3 = vaddq_u16(row_base_r3, prod_col_tr);

            acc_r0 = vmlaq_u16(acc_r0, const_nt_1_col_t, ref_left_r0_dup);
            acc_r1 = vmlaq_u16(acc_r1, const_nt_1_col_t, ref_left_r1_dup);
            acc_r2 = vmlaq_u16(acc_r2, const_nt_1_col_t, ref_left_r2_dup);
            acc_r3 = vmlaq_u16(acc_r3, const_nt_1_col_t, ref_left_r3_dup);

            acc_r0 = vmlaq_u16(acc_r0, const_nt_1_r0_dup, pu2_ref_two_nt_1_t);
            acc_r1 = vmlaq_u16(acc_r1, const_nt_1_r1_dup, pu2_ref_two_nt_1_t);
            acc_r2 = vmlaq_u16(acc_r2, const_nt_1_r2_dup, pu2_ref_two_nt_1_t);
            acc_r3 = vmlaq_u16(acc_r3, const_nt_1_r3_dup, pu2_ref_two_nt_1_t);

            vst1q_u16(pu2_dst_r0, vshlq_u16(acc_r0, log2nt_dup));
            vst1q_u16(pu2_dst_r1, vshlq_u16(acc_r1, log2nt_dup));
            vst1q_u16(pu2_dst_r2, vshlq_u16(acc_r2, log2nt_dup));
            vst1q_u16(pu2_dst_r3, vshlq_u16(acc_r3, log2nt_dup));

            pu2_dst_r0 += 8;
            pu2_dst_r1 += 8;
            pu2_dst_r2 += 8;
            pu2_dst_r3 += 8;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Intra prediction interpolation filter for HBD chroma DC using NEON
*  intrinsics.
*
* @par Description:
*  Intraprediction for DC mode with reference neighboring samples location
*  pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'.
*  Refer to section 8.4.4.2.5 in the HEVC standard.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the source reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intraprediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_dc_neonintr(UWORD16 *pu2_ref,
                                             WORD32 src_strd,
                                             UWORD16 *pu2_dst,
                                             WORD32 dst_strd,
                                             WORD32 nt,
                                             WORD32 mode)
{
    WORD32 i;
    WORD32 row;
    WORD32 log2nt_plus1;
    WORD32 two_nt;
    UWORD16 *pu2_ref_tmp;
    UWORD16 *pu2_ref_tmp1;
    UWORD16 *pu2_dst_tmp;
    uint16x8_t ref_load1, ref_load2;
    uint32x4_t acc_dc_uv0, acc_dc_uv1;
    uint32x2_t sum_uv;
    uint16x4_t dc_uv_u16;
    uint16x8_t dc_val_t;
    UNUSED(src_strd);
    UNUSED(mode);

    pu2_dst_tmp = pu2_dst;

    log2nt_plus1 = 32 - CLZ(nt);
    two_nt = 2 * nt;

    pu2_ref_tmp = pu2_ref + two_nt;
    pu2_ref_tmp1 = pu2_ref + (4 * nt) + 2;
    acc_dc_uv0 = vdupq_n_u32(0);
    acc_dc_uv1 = vdupq_n_u32(0);

    for(i = two_nt; i > 0; i -= 8)
    {
        ref_load1 = vld1q_u16(pu2_ref_tmp);
        pu2_ref_tmp += 8;

        ref_load2 = vld1q_u16(pu2_ref_tmp1);
        pu2_ref_tmp1 += 8;

        acc_dc_uv0 = vaddq_u32(acc_dc_uv0,
                               vaddl_u16(vget_low_u16(ref_load1),
                                         vget_low_u16(ref_load2)));
        acc_dc_uv1 = vaddq_u32(acc_dc_uv1,
                               vaddl_u16(vget_high_u16(ref_load1),
                                         vget_high_u16(ref_load2)));
    }

    acc_dc_uv0 = vaddq_u32(acc_dc_uv0, acc_dc_uv1);
    sum_uv = vadd_u32(vget_low_u32(acc_dc_uv0), vget_high_u32(acc_dc_uv0));
    sum_uv = vrshl_u32(sum_uv, vdup_n_s32(-log2nt_plus1));
    dc_uv_u16 = vmovn_u32(vcombine_u32(sum_uv, sum_uv));
    dc_val_t = vreinterpretq_u16_u32(
        vdupq_lane_u32(vreinterpret_u32_u16(dc_uv_u16), 0));

    if(nt == 4)
    {
        vst1q_u16(pu2_dst_tmp, dc_val_t);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, dc_val_t);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, dc_val_t);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, dc_val_t);
    }
    else if(nt == 8)
    {
        for(row = 0; row < 8; row += 4)
        {
            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            pu2_dst_tmp += dst_strd;
        }
    }
    else if(nt == 16)
    {
        for(row = 0; row < 16; row += 4)
        {
            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            pu2_dst_tmp += dst_strd;
        }
    }
    else
    {
        for(row = 0; row < 32; row += 4)
        {
            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 32, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 40, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 48, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 56, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 32, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 40, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 48, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 56, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 32, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 40, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 48, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 56, dc_val_t);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 8, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 16, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 24, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 32, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 40, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 48, dc_val_t);
            vst1q_u16(pu2_dst_tmp + 56, dc_val_t);
            pu2_dst_tmp += dst_strd;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Intra prediction interpolation filter for HBD chroma mode 2 using NEON
*  intrinsics.
*
* @par Description:
*  Intraprediction for mode 2 (sw angle) with reference neighboring samples
*  location pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'.
*  Refer to section 8.4.4.2.6 in the HEVC standard.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the source reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intraprediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_mode2_neonintr(UWORD16 *pu2_ref,
                                                WORD32 src_strd,
                                                UWORD16 *pu2_dst,
                                                WORD32 dst_strd,
                                                WORD32 nt,
                                                WORD32 mode)
{
    WORD32 row;
    UWORD16 *pu2_dst_tmp = pu2_dst;
    uint16x8_t t0, t1, t2, t3;
    uint16x8_t s0, s1, s2, s3;
    UNUSED(src_strd);
    UNUSED(mode);

    if(nt == 4)
    {
        t0 = vreinterpretq_u16_u32(
            vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref + 8))));
        s0 = vextq_u16(t0, t0, 4);

        t1 = vreinterpretq_u16_u32(
            vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref))));
        s1 = vextq_u16(t1, t1, 4);

        vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, s1);
    }
    else
    {
        UWORD16 *pu2_ref_ptr = pu2_ref + (4 * nt) - 8;

        if(nt == 8)
        {
            t0 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr))));
            s0 = vextq_u16(t0, t0, 4);
            t1 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 8))));
            s1 = vextq_u16(t1, t1, 4);
            t2 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 16))));
            s2 = vextq_u16(t2, t2, 4);
            t3 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 24))));
            s3 = vextq_u16(t3, t3, 4);

            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
            vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
            vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
            vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, s1);
            vst1q_u16(pu2_dst_tmp + 8, s2);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 2));
            vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 2));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 4));
            vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 4));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 6));
            vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 6));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, s2);
            vst1q_u16(pu2_dst_tmp + 8, s3);
        }
        else if(nt == 16)
        {
            t0 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr))));
            s0 = vextq_u16(t0, t0, 4);
            t1 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 8))));
            s1 = vextq_u16(t1, t1, 4);
            t2 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 16))));
            s2 = vextq_u16(t2, t2, 4);
            t3 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 24))));
            s3 = vextq_u16(t3, t3, 4);
            pu2_ref_ptr -= 32;

            for(row = 16; row > 0; row -= 4)
            {
                uint16x8_t t4 = vreinterpretq_u16_u32(
                    vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr))));
                uint16x8_t s4 = vextq_u16(t4, t4, 4);
                pu2_ref_ptr -= 8;

                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, s1);
                vst1q_u16(pu2_dst_tmp + 8, s2);
                vst1q_u16(pu2_dst_tmp + 16, s3);
                vst1q_u16(pu2_dst_tmp + 24, s4);
                pu2_dst_tmp += dst_strd;

                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = s4;
            }
        }
        else
        {
            uint16x8_t t4, t5, t6, t7;
            uint16x8_t s4, s5, s6, s7;

            t0 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr))));
            s0 = vextq_u16(t0, t0, 4);
            t1 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 8))));
            s1 = vextq_u16(t1, t1, 4);
            t2 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 16))));
            s2 = vextq_u16(t2, t2, 4);
            t3 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 24))));
            s3 = vextq_u16(t3, t3, 4);
            t4 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 32))));
            s4 = vextq_u16(t4, t4, 4);
            t5 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 40))));
            s5 = vextq_u16(t5, t5, 4);
            t6 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 48))));
            s6 = vextq_u16(t6, t6, 4);
            t7 = vreinterpretq_u16_u32(
                vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr - 56))));
            s7 = vextq_u16(t7, t7, 4);
            pu2_ref_ptr -= 64;

            for(row = 32; row > 0; row -= 4)
            {
                uint16x8_t t8 = vreinterpretq_u16_u32(
                    vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_ptr))));
                uint16x8_t s8 = vextq_u16(t8, t8, 4);
                pu2_ref_ptr -= 8;

                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 2));
                vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 2));
                vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 2));
                vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 2));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 4));
                vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 4));
                vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 4));
                vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 4));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 6));
                vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 6));
                vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 6));
                vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 6));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, s1);
                vst1q_u16(pu2_dst_tmp + 8, s2);
                vst1q_u16(pu2_dst_tmp + 16, s3);
                vst1q_u16(pu2_dst_tmp + 24, s4);
                vst1q_u16(pu2_dst_tmp + 32, s5);
                vst1q_u16(pu2_dst_tmp + 40, s6);
                vst1q_u16(pu2_dst_tmp + 48, s7);
                vst1q_u16(pu2_dst_tmp + 56, s8);
                pu2_dst_tmp += dst_strd;

                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = s4;
                s4 = s5;
                s5 = s6;
                s6 = s7;
                s7 = s8;
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for chroma mode 3 to mode 9
*
* @par Description:
*    Intraprediction for mode 3 to 9 (positive angle, horizontal mode) with
*    reference neighboring samples location pointed by 'pu2_ref' to the TU
*    block location pointed by 'pu2_dst'
*
* @param[in] pu2_ref
*  UWORD16 pointer to the source
*
* @param[in] src_strd
*  integer source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_mode_3_to_9_neonintr(UWORD16 *pu2_ref,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_dst,
                                                      WORD32 dst_strd,
                                                      WORD32 nt,
                                                      WORD32 mode)
{
    WORD32 row, col;
    WORD32 intra_pred_ang;
    WORD32 pos, fract;
    UNUSED(src_strd);

    /* Intra Pred Angle according to the mode */
    intra_pred_ang = gai4_ihevc_ang_table[mode];

    if(0 == (nt & 7))
    {
        WORD32 two_nt = 2 * nt;
        WORD32 four_nt = 4 * nt;

        for(col = 0; col < two_nt; col += 16)
        {
            WORD32 idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
            uint16x8_t f0, f1, f2, f3, f4, f5, f6, f7;
            uint16x8_t inv_f0, inv_f1, inv_f2, inv_f3, inv_f4, inv_f5, inv_f6, inv_f7;
            UWORD16 *pu2_ref_row = pu2_ref + four_nt - 8;
            UWORD16 *pu2_dst_col = pu2_dst + col;

            pos = ((col >> 1) + 1) * intra_pred_ang;
            idx0 = -2 * (pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx1 = -2 * (pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx2 = -2 * (pos >> 5);
            fract = pos & 31;
            f2 = vdupq_n_u16((uint16_t)fract);
            inv_f2 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx3 = -2 * (pos >> 5);
            fract = pos & 31;
            f3 = vdupq_n_u16((uint16_t)fract);
            inv_f3 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx4 = -2 * (pos >> 5);
            fract = pos & 31;
            f4 = vdupq_n_u16((uint16_t)fract);
            inv_f4 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx5 = -2 * (pos >> 5);
            fract = pos & 31;
            f5 = vdupq_n_u16((uint16_t)fract);
            inv_f5 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx6 = -2 * (pos >> 5);
            fract = pos & 31;
            f6 = vdupq_n_u16((uint16_t)fract);
            inv_f6 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx7 = -2 * (pos >> 5);
            fract = pos & 31;
            f7 = vdupq_n_u16((uint16_t)fract);
            inv_f7 = vdupq_n_u16((uint16_t)(32 - fract));

            for(row = nt; row > 0; row -= 4)
            {
                UWORD16 *p;
                uint16x8_t c0, c1, c2, c3, c4, c5, c6, c7;
                uint32x4x2_t t01, t23, t45, t67;

                p = pu2_ref_row + idx0;
                c0 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f0), vld1q_u16(p - 2), f0), 5);
                p = pu2_ref_row + idx1;
                c1 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f1), vld1q_u16(p - 2), f1), 5);
                p = pu2_ref_row + idx2;
                c2 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f2), vld1q_u16(p - 2), f2), 5);
                p = pu2_ref_row + idx3;
                c3 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f3), vld1q_u16(p - 2), f3), 5);
                p = pu2_ref_row + idx4;
                c4 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f4), vld1q_u16(p - 2), f4), 5);
                p = pu2_ref_row + idx5;
                c5 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f5), vld1q_u16(p - 2), f5), 5);
                p = pu2_ref_row + idx6;
                c6 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f6), vld1q_u16(p - 2), f6), 5);
                p = pu2_ref_row + idx7;
                c7 = vrshrq_n_u16(
                    vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f7), vld1q_u16(p - 2), f7), 5);

                t01 = vtrnq_u32(vreinterpretq_u32_u16(c0), vreinterpretq_u32_u16(c1));
                t23 = vtrnq_u32(vreinterpretq_u32_u16(c2), vreinterpretq_u32_u16(c3));
                t45 = vtrnq_u32(vreinterpretq_u32_u16(c4), vreinterpretq_u32_u16(c5));
                t67 = vtrnq_u32(vreinterpretq_u32_u16(c6), vreinterpretq_u32_u16(c7));

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(t01.val[1]), vget_high_u32(t23.val[1]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(t45.val[1]), vget_high_u32(t67.val[1]))));
                pu2_dst_col += dst_strd;

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(t01.val[0]), vget_high_u32(t23.val[0]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(t45.val[0]), vget_high_u32(t67.val[0]))));
                pu2_dst_col += dst_strd;

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(t01.val[1]), vget_low_u32(t23.val[1]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(t45.val[1]), vget_low_u32(t67.val[1]))));
                pu2_dst_col += dst_strd;

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(t01.val[0]), vget_low_u32(t23.val[0]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(t45.val[0]), vget_low_u32(t67.val[0]))));
                pu2_dst_col += dst_strd;

                pu2_ref_row -= 8;
            }
        }
    }
    else
    {
        UWORD16 *pu2_ref_base = pu2_ref + 8;
        UWORD16 *p;
        uint16x8_t c0, c1, c2, c3;
        uint32x4x2_t t01, t23;

        pos = intra_pred_ang;
        p = pu2_ref_base - 2 * (pos >> 5);
        fract = pos & 31;
        c0 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p - 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        pos += intra_pred_ang;
        p = pu2_ref_base - 2 * (pos >> 5);
        fract = pos & 31;
        c1 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p - 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        pos += intra_pred_ang;
        p = pu2_ref_base - 2 * (pos >> 5);
        fract = pos & 31;
        c2 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p - 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        pos += intra_pred_ang;
        p = pu2_ref_base - 2 * (pos >> 5);
        fract = pos & 31;
        c3 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p - 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        t01 = vtrnq_u32(vreinterpretq_u32_u16(c0), vreinterpretq_u32_u16(c1));
        t23 = vtrnq_u32(vreinterpretq_u32_u16(c2), vreinterpretq_u32_u16(c3));

        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_high_u32(t01.val[1]), vget_high_u32(t23.val[1]))));
        pu2_dst += dst_strd;
        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_high_u32(t01.val[0]), vget_high_u32(t23.val[0]))));
        pu2_dst += dst_strd;
        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_low_u32(t01.val[1]), vget_low_u32(t23.val[1]))));
        pu2_dst += dst_strd;
        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_low_u32(t01.val[0]), vget_low_u32(t23.val[0]))));
    }
}

/**
*******************************************************************************
*
* @brief
*  Horizontal intraprediction (mode 10) with reference samples location
*  pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'
*  using ARM NEON intrinsics.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intra prediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_horz_neonintr(UWORD16 *pu2_ref,
                                               WORD32 src_strd,
                                               UWORD16 *pu2_dst,
                                               WORD32 dst_strd,
                                               WORD32 nt,
                                               WORD32 mode)
{
    WORD32 row;
    WORD32 four_nt = 4 * nt;
    UWORD16 *pu2_ref_tmp = pu2_ref + four_nt - 8;
    UWORD16 *pu2_dst_tmp = pu2_dst;

    UNUSED(src_strd);
    UNUSED(mode);

    if(nt >= 16)
    {
        WORD32 col;
        WORD32 two_nt = 2 * nt;

        for(row = 0; row < nt; row += 4)
        {
            uint32x4_t ref_v = vreinterpretq_u32_u16(vld1q_u16(pu2_ref_tmp));
            uint16x8_t d0 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v), 1));
            uint16x8_t d1 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v), 0));
            uint16x8_t d2 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v), 1));
            uint16x8_t d3 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v), 0));
            UWORD16 *pu2_dst_r0 = pu2_dst_tmp;
            UWORD16 *pu2_dst_r1 = pu2_dst_r0 + dst_strd;
            UWORD16 *pu2_dst_r2 = pu2_dst_r1 + dst_strd;
            UWORD16 *pu2_dst_r3 = pu2_dst_r2 + dst_strd;
            pu2_ref_tmp -= 8;

            for(col = 0; col < two_nt; col += 32)
            {
                vst1q_u16(pu2_dst_r0 + col, d0);
                vst1q_u16(pu2_dst_r0 + col + 8, d0);
                vst1q_u16(pu2_dst_r0 + col + 16, d0);
                vst1q_u16(pu2_dst_r0 + col + 24, d0);

                vst1q_u16(pu2_dst_r1 + col, d1);
                vst1q_u16(pu2_dst_r1 + col + 8, d1);
                vst1q_u16(pu2_dst_r1 + col + 16, d1);
                vst1q_u16(pu2_dst_r1 + col + 24, d1);

                vst1q_u16(pu2_dst_r2 + col, d2);
                vst1q_u16(pu2_dst_r2 + col + 8, d2);
                vst1q_u16(pu2_dst_r2 + col + 16, d2);
                vst1q_u16(pu2_dst_r2 + col + 24, d2);

                vst1q_u16(pu2_dst_r3 + col, d3);
                vst1q_u16(pu2_dst_r3 + col + 8, d3);
                vst1q_u16(pu2_dst_r3 + col + 16, d3);
                vst1q_u16(pu2_dst_r3 + col + 24, d3);
            }
            pu2_dst_tmp = pu2_dst_r3 + dst_strd;
        }
    }
    else if(nt == 8)
    {
        uint32x4_t ref_v0 = vreinterpretq_u32_u16(vld1q_u16(pu2_ref_tmp));
        uint32x4_t ref_v1 = vreinterpretq_u32_u16(vld1q_u16(pu2_ref_tmp - 8));
        uint16x8_t d0 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v0), 1));
        uint16x8_t d1 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v0), 0));
        uint16x8_t d2 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v0), 1));
        uint16x8_t d3 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v0), 0));
        uint16x8_t d4 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v1), 1));
        uint16x8_t d5 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v1), 0));
        uint16x8_t d6 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v1), 1));
        uint16x8_t d7 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v1), 0));

        vst1q_u16(pu2_dst_tmp, d0);
        vst1q_u16(pu2_dst_tmp + 8, d0);
        pu2_dst_tmp += dst_strd;

        vst1q_u16(pu2_dst_tmp, d1);
        vst1q_u16(pu2_dst_tmp + 8, d1);
        pu2_dst_tmp += dst_strd;

        vst1q_u16(pu2_dst_tmp, d2);
        vst1q_u16(pu2_dst_tmp + 8, d2);
        pu2_dst_tmp += dst_strd;

        vst1q_u16(pu2_dst_tmp, d3);
        vst1q_u16(pu2_dst_tmp + 8, d3);
        pu2_dst_tmp += dst_strd;

        vst1q_u16(pu2_dst_tmp, d4);
        vst1q_u16(pu2_dst_tmp + 8, d4);
        pu2_dst_tmp += dst_strd;

        vst1q_u16(pu2_dst_tmp, d5);
        vst1q_u16(pu2_dst_tmp + 8, d5);
        pu2_dst_tmp += dst_strd;

        vst1q_u16(pu2_dst_tmp, d6);
        vst1q_u16(pu2_dst_tmp + 8, d6);
        pu2_dst_tmp += dst_strd;

        vst1q_u16(pu2_dst_tmp, d7);
        vst1q_u16(pu2_dst_tmp + 8, d7);
    }
    else
    {
        uint32x4_t ref_v = vreinterpretq_u32_u16(vld1q_u16(pu2_ref_tmp));
        uint16x8_t d0 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v), 1));
        uint16x8_t d1 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_high_u32(ref_v), 0));
        uint16x8_t d2 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v), 1));
        uint16x8_t d3 = vreinterpretq_u16_u32(vdupq_lane_u32(vget_low_u32(ref_v), 0));

        vst1q_u16(pu2_dst_tmp, d0);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, d1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, d2);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, d3);
    }
}

/**
*******************************************************************************
*
* @brief
*  Vertical intraprediction (mode 26) with reference samples location
*  pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'
*  using ARM NEON intrinsics.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intra prediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_ver_neonintr(UWORD16 *pu2_ref,
                                              WORD32 src_strd,
                                              UWORD16 *pu2_dst,
                                              WORD32 dst_strd,
                                              WORD32 nt,
                                              WORD32 mode)
{
    WORD32 row;
    UWORD16 *pu2_ref_tmp = pu2_ref + (4 * nt) + 2;
    UWORD16 *pu2_dst_tmp = pu2_dst;

    UNUSED(src_strd);
    UNUSED(mode);

    if(nt == 32)
    {
        uint16x8_t r0 = vld1q_u16(pu2_ref_tmp);
        uint16x8_t r1 = vld1q_u16(pu2_ref_tmp + 8);
        uint16x8_t r2 = vld1q_u16(pu2_ref_tmp + 16);
        uint16x8_t r3 = vld1q_u16(pu2_ref_tmp + 24);
        uint16x8_t r4 = vld1q_u16(pu2_ref_tmp + 32);
        uint16x8_t r5 = vld1q_u16(pu2_ref_tmp + 40);
        uint16x8_t r6 = vld1q_u16(pu2_ref_tmp + 48);
        uint16x8_t r7 = vld1q_u16(pu2_ref_tmp + 56);

        for(row = 0; row < 32; row += 4)
        {
            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            vst1q_u16(pu2_dst_tmp + 32, r4);
            vst1q_u16(pu2_dst_tmp + 40, r5);
            vst1q_u16(pu2_dst_tmp + 48, r6);
            vst1q_u16(pu2_dst_tmp + 56, r7);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            vst1q_u16(pu2_dst_tmp + 32, r4);
            vst1q_u16(pu2_dst_tmp + 40, r5);
            vst1q_u16(pu2_dst_tmp + 48, r6);
            vst1q_u16(pu2_dst_tmp + 56, r7);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            vst1q_u16(pu2_dst_tmp + 32, r4);
            vst1q_u16(pu2_dst_tmp + 40, r5);
            vst1q_u16(pu2_dst_tmp + 48, r6);
            vst1q_u16(pu2_dst_tmp + 56, r7);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            vst1q_u16(pu2_dst_tmp + 32, r4);
            vst1q_u16(pu2_dst_tmp + 40, r5);
            vst1q_u16(pu2_dst_tmp + 48, r6);
            vst1q_u16(pu2_dst_tmp + 56, r7);
            pu2_dst_tmp += dst_strd;
        }
    }
    else if(nt == 16)
    {
        uint16x8_t r0 = vld1q_u16(pu2_ref_tmp);
        uint16x8_t r1 = vld1q_u16(pu2_ref_tmp + 8);
        uint16x8_t r2 = vld1q_u16(pu2_ref_tmp + 16);
        uint16x8_t r3 = vld1q_u16(pu2_ref_tmp + 24);

        for(row = 0; row < 16; row += 4)
        {
            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, r0);
            vst1q_u16(pu2_dst_tmp + 8, r1);
            vst1q_u16(pu2_dst_tmp + 16, r2);
            vst1q_u16(pu2_dst_tmp + 24, r3);
            pu2_dst_tmp += dst_strd;
        }
    }
    else if(nt == 8)
    {
        uint16x8_t r0 = vld1q_u16(pu2_ref_tmp);
        uint16x8_t r1 = vld1q_u16(pu2_ref_tmp + 8);

        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        vst1q_u16(pu2_dst_tmp + 8, r1);
    }
    else
    {
        uint16x8_t r0 = vld1q_u16(pu2_ref_tmp);

        vst1q_u16(pu2_dst_tmp, r0);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
        pu2_dst_tmp += dst_strd;
        vst1q_u16(pu2_dst_tmp, r0);
    }
}

/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 34 (ne angle) and mode 18 (nw angle) with
*  reference neighboring samples location pointed by 'pu2_ref' to the TU
*  block location pointed by 'pu2_dst' using ARM NEON intrinsics.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intra prediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_mode_18_34_neonintr(UWORD16 *pu2_ref,
                                                     WORD32 src_strd,
                                                     UWORD16 *pu2_dst,
                                                     WORD32 dst_strd,
                                                     WORD32 nt,
                                                     WORD32 mode)
{
    WORD32 row;
    WORD32 four_nt = 4 * nt;
    UWORD16 *pu2_ref_tmp = pu2_ref;
    UWORD16 *pu2_dst_tmp = pu2_dst;

    UNUSED(src_strd);

    if(0 == (nt & 7))
    {
        if(mode == 34)
        {
            pu2_ref_tmp += (four_nt + 4);
            if(nt == 8)
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 24);

                vst1q_u16(pu2_dst_tmp, s0);
                vst1q_u16(pu2_dst_tmp + 8, s1);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, s1);
                vst1q_u16(pu2_dst_tmp + 8, s2);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 6));
            }
            else if(nt == 16)
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 24);
                pu2_ref_tmp += 32;

                for(row = 16; row > 0; row -= 4)
                {
                    uint16x8_t s4 = vld1q_u16(pu2_ref_tmp);
                    pu2_ref_tmp += 8;

                    vst1q_u16(pu2_dst_tmp, s0);
                    vst1q_u16(pu2_dst_tmp + 8, s1);
                    vst1q_u16(pu2_dst_tmp + 16, s2);
                    vst1q_u16(pu2_dst_tmp + 24, s3);
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                    pu2_dst_tmp += dst_strd;

                    s0 = s1;
                    s1 = s2;
                    s2 = s3;
                    s3 = s4;
                }
            }
            else
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 24);
                uint16x8_t s4 = vld1q_u16(pu2_ref_tmp + 32);
                uint16x8_t s5 = vld1q_u16(pu2_ref_tmp + 40);
                uint16x8_t s6 = vld1q_u16(pu2_ref_tmp + 48);
                uint16x8_t s7 = vld1q_u16(pu2_ref_tmp + 56);
                pu2_ref_tmp += 64;

                for(row = 32; row > 0; row -= 4)
                {
                    uint16x8_t s8 = vld1q_u16(pu2_ref_tmp);
                    pu2_ref_tmp += 8;

                    vst1q_u16(pu2_dst_tmp, s0);
                    vst1q_u16(pu2_dst_tmp + 8, s1);
                    vst1q_u16(pu2_dst_tmp + 16, s2);
                    vst1q_u16(pu2_dst_tmp + 24, s3);
                    vst1q_u16(pu2_dst_tmp + 32, s4);
                    vst1q_u16(pu2_dst_tmp + 40, s5);
                    vst1q_u16(pu2_dst_tmp + 48, s6);
                    vst1q_u16(pu2_dst_tmp + 56, s7);
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                    vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 2));
                    vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 2));
                    vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 2));
                    vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 2));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                    vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 4));
                    vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 4));
                    vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 4));
                    vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 4));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                    vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 6));
                    vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 6));
                    vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 6));
                    vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 6));
                    pu2_dst_tmp += dst_strd;

                    s0 = s1;
                    s1 = s2;
                    s2 = s3;
                    s3 = s4;
                    s4 = s5;
                    s5 = s6;
                    s6 = s7;
                    s7 = s8;
                }
            }
        }
        else /* Loop for mode 18 */
        {
            pu2_ref_tmp += four_nt;
            if(nt == 8)
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp - 16);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp - 8);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 8);

                vst1q_u16(pu2_dst_tmp, s2);
                vst1q_u16(pu2_dst_tmp + 8, s3);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 2));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, s1);
                vst1q_u16(pu2_dst_tmp + 8, s2);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
            }
            else if(nt == 16)
            {
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s4 = vld1q_u16(pu2_ref_tmp + 24);
                pu2_ref_tmp -= 8;

                for(row = 16; row > 0; row -= 4)
                {
                    uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                    pu2_ref_tmp -= 8;

                    vst1q_u16(pu2_dst_tmp, s1);
                    vst1q_u16(pu2_dst_tmp + 8, s2);
                    vst1q_u16(pu2_dst_tmp + 16, s3);
                    vst1q_u16(pu2_dst_tmp + 24, s4);
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                    pu2_dst_tmp += dst_strd;

                    s4 = s3;
                    s3 = s2;
                    s2 = s1;
                    s1 = s0;
                }
            }
            else
            {
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s4 = vld1q_u16(pu2_ref_tmp + 24);
                uint16x8_t s5 = vld1q_u16(pu2_ref_tmp + 32);
                uint16x8_t s6 = vld1q_u16(pu2_ref_tmp + 40);
                uint16x8_t s7 = vld1q_u16(pu2_ref_tmp + 48);
                uint16x8_t s8 = vld1q_u16(pu2_ref_tmp + 56);
                pu2_ref_tmp -= 8;

                for(row = 32; row > 0; row -= 4)
                {
                    uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                    pu2_ref_tmp -= 8;

                    vst1q_u16(pu2_dst_tmp, s1);
                    vst1q_u16(pu2_dst_tmp + 8, s2);
                    vst1q_u16(pu2_dst_tmp + 16, s3);
                    vst1q_u16(pu2_dst_tmp + 24, s4);
                    vst1q_u16(pu2_dst_tmp + 32, s5);
                    vst1q_u16(pu2_dst_tmp + 40, s6);
                    vst1q_u16(pu2_dst_tmp + 48, s7);
                    vst1q_u16(pu2_dst_tmp + 56, s8);
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                    vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 6));
                    vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 6));
                    vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 6));
                    vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 6));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                    vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 4));
                    vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 4));
                    vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 4));
                    vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 4));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                    vst1q_u16(pu2_dst_tmp + 32, vextq_u16(s4, s5, 2));
                    vst1q_u16(pu2_dst_tmp + 40, vextq_u16(s5, s6, 2));
                    vst1q_u16(pu2_dst_tmp + 48, vextq_u16(s6, s7, 2));
                    vst1q_u16(pu2_dst_tmp + 56, vextq_u16(s7, s8, 2));
                    pu2_dst_tmp += dst_strd;

                    s8 = s7;
                    s7 = s6;
                    s6 = s5;
                    s5 = s4;
                    s4 = s3;
                    s3 = s2;
                    s2 = s1;
                    s1 = s0;
                }
            }
        }
    }
    else /* nt == 4 */
    {
        if(mode == 34)
        {
            uint16x8_t s0 = vld1q_u16(pu2_ref + four_nt + 4);
            uint16x8_t s1 = vld1q_u16(pu2_ref + four_nt + 12);

            vst1q_u16(pu2_dst_tmp, s0);
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
        }
        else
        {
            uint16x8_t s0 = vld1q_u16(pu2_ref + four_nt - 8);
            uint16x8_t s1 = vld1q_u16(pu2_ref + four_nt);

            vst1q_u16(pu2_dst_tmp, s1);
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
            pu2_dst_tmp += dst_strd;
            vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 11 to 17 (negative angle, horizontal mode)
*  with reference neighboring samples location pointed by 'pu2_ref' to the
*  TU block location pointed by 'pu2_dst' using ARM NEON intrinsics.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intra prediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_mode_11_to_17_neonintr(UWORD16 *pu2_ref,
                                                        WORD32 src_strd,
                                                        UWORD16 *pu2_dst,
                                                        WORD32 dst_strd,
                                                        WORD32 nt,
                                                        WORD32 mode)
{
    WORD32 row, col, k;
    WORD32 four_nt = 4 * nt;
    WORD32 intra_pred_ang, inv_ang, inv_ang_sum;
    WORD32 pos, fract;
    WORD32 ref_idx;

    UWORD16 ref_temp[2 * MAX_CU_SIZE + 2];
    UWORD16 *ref_main = ref_temp + 2 * (nt - 1);
    UWORD16 *ref_main_tmp = ref_main;
    UWORD16 *pu2_ref_tmp2 = pu2_ref + four_nt - 6;

    UNUSED(src_strd);

    inv_ang_sum = 128;
    intra_pred_ang = gai4_ihevc_ang_table[mode];
    inv_ang = gai4_ihevc_inv_ang_table[mode - 11];

    for(k = nt; k > 0; k -= 4)
    {
        uint32x4_t ref_left_t = vrev64q_u32(vreinterpretq_u32_u16(vld1q_u16(pu2_ref_tmp2)));
        ref_left_t = vextq_u32(ref_left_t, ref_left_t, 2);
        vst1q_u16(ref_main_tmp, vreinterpretq_u16_u32(ref_left_t));
        ref_main_tmp += 8;
        pu2_ref_tmp2 -= 8;
    }

    ref_main[2 * nt] = pu2_ref[2 * nt];
    ref_main[2 * nt + 1] = pu2_ref[2 * nt + 1];

    ref_idx = (nt * intra_pred_ang) >> 5;

    for(k = -2; k > (2 * ref_idx); k -= 2)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[four_nt + ((inv_ang_sum >> 8) << 1)];
        ref_main[k + 1] = pu2_ref[(four_nt + 1) + ((inv_ang_sum >> 8) << 1)];
    }

    if(0 == (nt & 7))
    {
        for(col = 0; col < 2 * nt; col += 16)
        {
            WORD32 idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
            uint16x8_t f0, f1, f2, f3, f4, f5, f6, f7;
            uint16x8_t inv_f0, inv_f1, inv_f2, inv_f3, inv_f4, inv_f5, inv_f6, inv_f7;
            UWORD16 *ref_row = ref_main;
            UWORD16 *pu2_dst_col = pu2_dst + col;

            pos = ((col >> 1) + 1) * intra_pred_ang;
            idx0 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx1 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx2 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f2 = vdupq_n_u16((uint16_t)fract);
            inv_f2 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx3 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f3 = vdupq_n_u16((uint16_t)fract);
            inv_f3 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx4 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f4 = vdupq_n_u16((uint16_t)fract);
            inv_f4 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx5 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f5 = vdupq_n_u16((uint16_t)fract);
            inv_f5 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx6 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f6 = vdupq_n_u16((uint16_t)fract);
            inv_f6 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx7 = ((pos >> 5) + 1) << 1;
            fract = pos & 31;
            f7 = vdupq_n_u16((uint16_t)fract);
            inv_f7 = vdupq_n_u16((uint16_t)(32 - fract));

            for(row = nt; row > 0; row -= 4)
            {
                UWORD16 *p;
                uint16x8_t c0, c1, c2, c3, c4, c5, c6, c7;
                uint32x4x2_t s0, s1, s2, s3;

                p = ref_row + idx0;
                c0 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f0), vld1q_u16(p + 2), f0), 5);
                p = ref_row + idx1;
                c1 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f1), vld1q_u16(p + 2), f1), 5);
                p = ref_row + idx2;
                c2 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f2), vld1q_u16(p + 2), f2), 5);
                p = ref_row + idx3;
                c3 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f3), vld1q_u16(p + 2), f3), 5);
                p = ref_row + idx4;
                c4 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f4), vld1q_u16(p + 2), f4), 5);
                p = ref_row + idx5;
                c5 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f5), vld1q_u16(p + 2), f5), 5);
                p = ref_row + idx6;
                c6 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f6), vld1q_u16(p + 2), f6), 5);
                p = ref_row + idx7;
                c7 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f7), vld1q_u16(p + 2), f7), 5);

                s0 = vtrnq_u32(vreinterpretq_u32_u16(c0), vreinterpretq_u32_u16(c1));
                s1 = vtrnq_u32(vreinterpretq_u32_u16(c2), vreinterpretq_u32_u16(c3));
                s2 = vtrnq_u32(vreinterpretq_u32_u16(c4), vreinterpretq_u32_u16(c5));
                s3 = vtrnq_u32(vreinterpretq_u32_u16(c6), vreinterpretq_u32_u16(c7));

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(s0.val[0]), vget_low_u32(s1.val[0]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(s2.val[0]), vget_low_u32(s3.val[0]))));
                pu2_dst_col += dst_strd;

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(s0.val[1]), vget_low_u32(s1.val[1]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_low_u32(s2.val[1]), vget_low_u32(s3.val[1]))));
                pu2_dst_col += dst_strd;

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(s0.val[0]), vget_high_u32(s1.val[0]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(s2.val[0]), vget_high_u32(s3.val[0]))));
                pu2_dst_col += dst_strd;

                vst1q_u16(pu2_dst_col,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(s0.val[1]), vget_high_u32(s1.val[1]))));
                vst1q_u16(pu2_dst_col + 8,
                          vreinterpretq_u16_u32(
                              vcombine_u32(vget_high_u32(s2.val[1]), vget_high_u32(s3.val[1]))));
                pu2_dst_col += dst_strd;

                ref_row += 8;
            }
        }
    }
    else
    {
        UWORD16 *p;
        uint16x8_t c0, c1, c2, c3;
        uint32x4x2_t t01, t23;

        pos = intra_pred_ang;
        p = ref_main + (((pos >> 5) + 1) << 1);
        fract = pos & 31;
        c0 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p + 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        pos += intra_pred_ang;
        p = ref_main + (((pos >> 5) + 1) << 1);
        fract = pos & 31;
        c1 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p + 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        pos += intra_pred_ang;
        p = ref_main + (((pos >> 5) + 1) << 1);
        fract = pos & 31;
        c2 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p + 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        pos += intra_pred_ang;
        p = ref_main + (((pos >> 5) + 1) << 1);
        fract = pos & 31;
        c3 = vrshrq_n_u16(
            vmlaq_u16(vmulq_u16(vld1q_u16(p), vdupq_n_u16((uint16_t)(32 - fract))),
                      vld1q_u16(p + 2),
                      vdupq_n_u16((uint16_t)fract)),
            5);

        t01 = vtrnq_u32(vreinterpretq_u32_u16(c0), vreinterpretq_u32_u16(c1));
        t23 = vtrnq_u32(vreinterpretq_u32_u16(c2), vreinterpretq_u32_u16(c3));

        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_low_u32(t01.val[0]), vget_low_u32(t23.val[0]))));
        pu2_dst += dst_strd;
        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_low_u32(t01.val[1]), vget_low_u32(t23.val[1]))));
        pu2_dst += dst_strd;
        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_high_u32(t01.val[0]), vget_high_u32(t23.val[0]))));
        pu2_dst += dst_strd;
        vst1q_u16(pu2_dst,
                  vreinterpretq_u16_u32(
                      vcombine_u32(vget_high_u32(t01.val[1]), vget_high_u32(t23.val[1]))));
    }
}

/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 19 to 25 (negative angle, vertical mode) with
*  reference neighboring samples location pointed by 'pu2_ref' to the TU
*  block location pointed by 'pu2_dst' using ARM NEON intrinsics.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intra prediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_mode_19_to_25_neonintr(UWORD16 *pu2_ref,
                                                        WORD32 src_strd,
                                                        UWORD16 *pu2_dst,
                                                        WORD32 dst_strd,
                                                        WORD32 nt,
                                                        WORD32 mode)
{
    WORD32 row, k;
    WORD32 four_nt = 4 * nt;
    WORD32 intra_pred_ang, inv_ang, inv_ang_sum, pos, fract;
    WORD32 ref_idx;
    UWORD16 ref_temp[(2 * MAX_CU_SIZE) + 2];
    UWORD16 *ref_main = ref_temp + 2 * (nt - 1);
    UWORD16 *pu2_ref_tmp1 = pu2_ref + four_nt;
    UWORD16 *pu2_dst_tmp1 = pu2_dst;

    UNUSED(src_strd);

    intra_pred_ang = gai4_ihevc_ang_table_chroma[mode];
    inv_ang = gai4_ihevc_inv_ang_table_chroma[mode - 12];

    ref_idx = (nt * intra_pred_ang) >> 5;
    inv_ang_sum = 128;

    for(k = -2; k > (2 * ref_idx); k -= 2)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[four_nt - ((inv_ang_sum >> 8) << 1)];
        ref_main[k + 1] = pu2_ref[(four_nt + 1) - ((inv_ang_sum >> 8) << 1)];
    }

    UWORD16 *ref_base = ref_main + 2;

    if(nt == 8)
    {
        uint16x8_t c0 = vld1q_u16(pu2_ref_tmp1);
        uint16x8_t c1 = vld1q_u16(pu2_ref_tmp1 + 8);
        vst1q_u16(ref_main, c0);
        vst1q_u16(ref_main + 8, c1);
        ref_main[16] = pu2_ref_tmp1[16];
        ref_main[17] = pu2_ref_tmp1[17];

        pos = 0;
        for(row = 0; row < 8; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a00, a01, b00, b01, a10, a11, b10, b11;
            uint16x8_t r00, r01, r10, r11;

            pos += intra_pred_ang;
            p0 = ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            b00 = vextq_u16(a00, a01, 2);
            b01 = vld1q_u16(p0 + 10);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            b10 = vextq_u16(a10, a11, 2);
            b11 = vld1q_u16(p1 + 10);

            r00 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a00, inv_f0), b00, f0), 5);
            r01 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a01, inv_f0), b01, f0), 5);
            r10 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a10, inv_f1), b10, f1), 5);
            r11 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a11, inv_f1), b11, f1), 5);

            vst1q_u16(pu2_dst_tmp1, r00);
            vst1q_u16(pu2_dst_tmp1 + 8, r01);
            pu2_dst_tmp1 += dst_strd;
            vst1q_u16(pu2_dst_tmp1, r10);
            vst1q_u16(pu2_dst_tmp1 + 8, r11);
            pu2_dst_tmp1 += dst_strd;
        }
    }
    else if(nt == 16)
    {
        uint16x8_t c0 = vld1q_u16(pu2_ref_tmp1);
        uint16x8_t c1 = vld1q_u16(pu2_ref_tmp1 + 8);
        uint16x8_t c2 = vld1q_u16(pu2_ref_tmp1 + 16);
        uint16x8_t c3 = vld1q_u16(pu2_ref_tmp1 + 24);
        vst1q_u16(ref_main, c0);
        vst1q_u16(ref_main + 8, c1);
        vst1q_u16(ref_main + 16, c2);
        vst1q_u16(ref_main + 24, c3);
        ref_main[32] = pu2_ref_tmp1[32];
        ref_main[33] = pu2_ref_tmp1[33];

        pos = 0;
        for(row = 0; row < 16; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a00, a01, a02, a03, b00, b01, b02, b03;
            uint16x8_t a10, a11, a12, a13, b10, b11, b12, b13;
            uint16x8_t r00, r01, r02, r03, r10, r11, r12, r13;

            pos += intra_pred_ang;
            p0 = ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            a02 = vld1q_u16(p0 + 16);
            a03 = vld1q_u16(p0 + 24);
            b00 = vextq_u16(a00, a01, 2);
            b01 = vextq_u16(a01, a02, 2);
            b02 = vextq_u16(a02, a03, 2);
            b03 = vld1q_u16(p0 + 26);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            a12 = vld1q_u16(p1 + 16);
            a13 = vld1q_u16(p1 + 24);
            b10 = vextq_u16(a10, a11, 2);
            b11 = vextq_u16(a11, a12, 2);
            b12 = vextq_u16(a12, a13, 2);
            b13 = vld1q_u16(p1 + 26);

            r00 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a00, inv_f0), b00, f0), 5);
            r01 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a01, inv_f0), b01, f0), 5);
            r02 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a02, inv_f0), b02, f0), 5);
            r03 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a03, inv_f0), b03, f0), 5);

            r10 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a10, inv_f1), b10, f1), 5);
            r11 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a11, inv_f1), b11, f1), 5);
            r12 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a12, inv_f1), b12, f1), 5);
            r13 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a13, inv_f1), b13, f1), 5);

            vst1q_u16(pu2_dst_tmp1, r00);
            vst1q_u16(pu2_dst_tmp1 + 8, r01);
            vst1q_u16(pu2_dst_tmp1 + 16, r02);
            vst1q_u16(pu2_dst_tmp1 + 24, r03);
            pu2_dst_tmp1 += dst_strd;

            vst1q_u16(pu2_dst_tmp1, r10);
            vst1q_u16(pu2_dst_tmp1 + 8, r11);
            vst1q_u16(pu2_dst_tmp1 + 16, r12);
            vst1q_u16(pu2_dst_tmp1 + 24, r13);
            pu2_dst_tmp1 += dst_strd;
        }
    }
    else if(nt == 32)
    {
        uint16x8_t c0 = vld1q_u16(pu2_ref_tmp1);
        uint16x8_t c1 = vld1q_u16(pu2_ref_tmp1 + 8);
        uint16x8_t c2 = vld1q_u16(pu2_ref_tmp1 + 16);
        uint16x8_t c3 = vld1q_u16(pu2_ref_tmp1 + 24);
        uint16x8_t c4 = vld1q_u16(pu2_ref_tmp1 + 32);
        uint16x8_t c5 = vld1q_u16(pu2_ref_tmp1 + 40);
        uint16x8_t c6 = vld1q_u16(pu2_ref_tmp1 + 48);
        uint16x8_t c7 = vld1q_u16(pu2_ref_tmp1 + 56);
        vst1q_u16(ref_main, c0);
        vst1q_u16(ref_main + 8, c1);
        vst1q_u16(ref_main + 16, c2);
        vst1q_u16(ref_main + 24, c3);
        vst1q_u16(ref_main + 32, c4);
        vst1q_u16(ref_main + 40, c5);
        vst1q_u16(ref_main + 48, c6);
        vst1q_u16(ref_main + 56, c7);
        ref_main[64] = pu2_ref_tmp1[64];
        ref_main[65] = pu2_ref_tmp1[65];

        pos = 0;
        for(row = 0; row < 32; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a0, a1, a2, a3, a4, a5, a6, a7;
            uint16x8_t b0, b1, b2, b3, b4, b5, b6, b7;

            pos += intra_pred_ang;
            p0 = ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a0 = vld1q_u16(p0);
            a1 = vld1q_u16(p0 + 8);
            a2 = vld1q_u16(p0 + 16);
            a3 = vld1q_u16(p0 + 24);
            a4 = vld1q_u16(p0 + 32);
            a5 = vld1q_u16(p0 + 40);
            a6 = vld1q_u16(p0 + 48);
            a7 = vld1q_u16(p0 + 56);

            b0 = vextq_u16(a0, a1, 2);
            b1 = vextq_u16(a1, a2, 2);
            b2 = vextq_u16(a2, a3, 2);
            b3 = vextq_u16(a3, a4, 2);
            b4 = vextq_u16(a4, a5, 2);
            b5 = vextq_u16(a5, a6, 2);
            b6 = vextq_u16(a6, a7, 2);
            b7 = vld1q_u16(p0 + 58);

            vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f0), b0, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 8, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f0), b1, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 16, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a2, inv_f0), b2, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 24, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a3, inv_f0), b3, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 32, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a4, inv_f0), b4, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 40, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a5, inv_f0), b5, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 48, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a6, inv_f0), b6, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 56, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a7, inv_f0), b7, f0), 5));
            pu2_dst_tmp1 += dst_strd;

            a0 = vld1q_u16(p1);
            a1 = vld1q_u16(p1 + 8);
            a2 = vld1q_u16(p1 + 16);
            a3 = vld1q_u16(p1 + 24);
            a4 = vld1q_u16(p1 + 32);
            a5 = vld1q_u16(p1 + 40);
            a6 = vld1q_u16(p1 + 48);
            a7 = vld1q_u16(p1 + 56);

            b0 = vextq_u16(a0, a1, 2);
            b1 = vextq_u16(a1, a2, 2);
            b2 = vextq_u16(a2, a3, 2);
            b3 = vextq_u16(a3, a4, 2);
            b4 = vextq_u16(a4, a5, 2);
            b5 = vextq_u16(a5, a6, 2);
            b6 = vextq_u16(a6, a7, 2);
            b7 = vld1q_u16(p1 + 58);

            vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f1), b0, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 8, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f1), b1, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 16, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a2, inv_f1), b2, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 24, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a3, inv_f1), b3, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 32, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a4, inv_f1), b4, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 40, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a5, inv_f1), b5, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 48, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a6, inv_f1), b6, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 56, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a7, inv_f1), b7, f1), 5));
            pu2_dst_tmp1 += dst_strd;
        }
    }
    else
    {
        UWORD16 *p0, *p1, *p2, *p3;
        uint16x8_t f0, inv_f0, f1, inv_f1, f2, inv_f2, f3, inv_f3;
        uint16x8_t a0, b0, a1, b1, a2, b2, a3, b3;

        vst1q_u16(ref_main, vld1q_u16(pu2_ref_tmp1));
        ref_main[8] = pu2_ref_tmp1[8];
        ref_main[9] = pu2_ref_tmp1[9];

        pos = intra_pred_ang;
        p0 = ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f0 = vdupq_n_u16((uint16_t)fract);
        inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p1 = ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f1 = vdupq_n_u16((uint16_t)fract);
        inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p2 = ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f2 = vdupq_n_u16((uint16_t)fract);
        inv_f2 = vdupq_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p3 = ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f3 = vdupq_n_u16((uint16_t)fract);
        inv_f3 = vdupq_n_u16((uint16_t)(32 - fract));

        a0 = vld1q_u16(p0);
        b0 = vld1q_u16(p0 + 2);
        a1 = vld1q_u16(p1);
        b1 = vld1q_u16(p1 + 2);
        a2 = vld1q_u16(p2);
        b2 = vld1q_u16(p2 + 2);
        a3 = vld1q_u16(p3);
        b3 = vld1q_u16(p3 + 2);

        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f0), b0, f0), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f1), b1, f1), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a2, inv_f2), b2, f2), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a3, inv_f3), b3, f3), 5));
    }
}

/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 27 to 33 (positive angle, vertical mode) with
*  reference neighboring samples location pointed by 'pu2_ref' to the TU
*  block location pointed by 'pu2_dst' using ARM NEON intrinsics.
*
* @param[in] pu2_ref
*  UWORD16 pointer to the reference samples
*
* @param[in] src_strd
*  WORD32 source stride
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] dst_strd
*  WORD32 destination stride
*
* @param[in] nt
*  WORD32 Transform Block size (Chroma)
*
* @param[in] mode
*  WORD32 intra prediction mode
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_mode_27_to_33_neonintr(UWORD16 *pu2_ref,
                                                        WORD32 src_strd,
                                                        UWORD16 *pu2_dst,
                                                        WORD32 dst_strd,
                                                        WORD32 nt,
                                                        WORD32 mode)
{
    WORD32 row;
    WORD32 intra_pred_ang;
    WORD32 pos, fract;
    UWORD16 *pu2_ref_base = pu2_ref + (4 * nt) + 2;
    UWORD16 *pu2_dst_tmp1 = pu2_dst;

    UNUSED(src_strd);

    intra_pred_ang = gai4_ihevc_ang_table_chroma[mode];

    if(nt == 8)
    {
        pos = 0;
        for(row = 0; row < 8; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a00, a01, b00, b01, a10, a11, b10, b11;
            uint16x8_t r00, r01, r10, r11;

            pos += intra_pred_ang;
            p0 = pu2_ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = pu2_ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            b00 = vextq_u16(a00, a01, 2);
            b01 = vld1q_u16(p0 + 10);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            b10 = vextq_u16(a10, a11, 2);
            b11 = vld1q_u16(p1 + 10);

            r00 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a00, inv_f0), b00, f0), 5);
            r01 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a01, inv_f0), b01, f0), 5);
            r10 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a10, inv_f1), b10, f1), 5);
            r11 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a11, inv_f1), b11, f1), 5);

            vst1q_u16(pu2_dst_tmp1, r00);
            vst1q_u16(pu2_dst_tmp1 + 8, r01);
            pu2_dst_tmp1 += dst_strd;
            vst1q_u16(pu2_dst_tmp1, r10);
            vst1q_u16(pu2_dst_tmp1 + 8, r11);
            pu2_dst_tmp1 += dst_strd;
        }
    }
    else if(nt == 16)
    {
        pos = 0;
        for(row = 0; row < 16; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a00, a01, a02, a03, b00, b01, b02, b03;
            uint16x8_t a10, a11, a12, a13, b10, b11, b12, b13;
            uint16x8_t r00, r01, r02, r03, r10, r11, r12, r13;

            pos += intra_pred_ang;
            p0 = pu2_ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = pu2_ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            a02 = vld1q_u16(p0 + 16);
            a03 = vld1q_u16(p0 + 24);
            b00 = vextq_u16(a00, a01, 2);
            b01 = vextq_u16(a01, a02, 2);
            b02 = vextq_u16(a02, a03, 2);
            b03 = vld1q_u16(p0 + 26);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            a12 = vld1q_u16(p1 + 16);
            a13 = vld1q_u16(p1 + 24);
            b10 = vextq_u16(a10, a11, 2);
            b11 = vextq_u16(a11, a12, 2);
            b12 = vextq_u16(a12, a13, 2);
            b13 = vld1q_u16(p1 + 26);

            r00 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a00, inv_f0), b00, f0), 5);
            r01 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a01, inv_f0), b01, f0), 5);
            r02 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a02, inv_f0), b02, f0), 5);
            r03 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a03, inv_f0), b03, f0), 5);

            r10 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a10, inv_f1), b10, f1), 5);
            r11 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a11, inv_f1), b11, f1), 5);
            r12 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a12, inv_f1), b12, f1), 5);
            r13 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a13, inv_f1), b13, f1), 5);

            vst1q_u16(pu2_dst_tmp1, r00);
            vst1q_u16(pu2_dst_tmp1 + 8, r01);
            vst1q_u16(pu2_dst_tmp1 + 16, r02);
            vst1q_u16(pu2_dst_tmp1 + 24, r03);
            pu2_dst_tmp1 += dst_strd;

            vst1q_u16(pu2_dst_tmp1, r10);
            vst1q_u16(pu2_dst_tmp1 + 8, r11);
            vst1q_u16(pu2_dst_tmp1 + 16, r12);
            vst1q_u16(pu2_dst_tmp1 + 24, r13);
            pu2_dst_tmp1 += dst_strd;
        }
    }
    else if(nt == 32)
    {
        pos = 0;
        for(row = 0; row < 32; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a0, a1, a2, a3, a4, a5, a6, a7;
            uint16x8_t b0, b1, b2, b3, b4, b5, b6, b7;

            pos += intra_pred_ang;
            p0 = pu2_ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = pu2_ref_base + ((pos >> 5) << 1);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a0 = vld1q_u16(p0);
            a1 = vld1q_u16(p0 + 8);
            a2 = vld1q_u16(p0 + 16);
            a3 = vld1q_u16(p0 + 24);
            a4 = vld1q_u16(p0 + 32);
            a5 = vld1q_u16(p0 + 40);
            a6 = vld1q_u16(p0 + 48);
            a7 = vld1q_u16(p0 + 56);

            b0 = vextq_u16(a0, a1, 2);
            b1 = vextq_u16(a1, a2, 2);
            b2 = vextq_u16(a2, a3, 2);
            b3 = vextq_u16(a3, a4, 2);
            b4 = vextq_u16(a4, a5, 2);
            b5 = vextq_u16(a5, a6, 2);
            b6 = vextq_u16(a6, a7, 2);
            b7 = vld1q_u16(p0 + 58);

            vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f0), b0, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 8, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f0), b1, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 16, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a2, inv_f0), b2, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 24, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a3, inv_f0), b3, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 32, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a4, inv_f0), b4, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 40, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a5, inv_f0), b5, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 48, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a6, inv_f0), b6, f0), 5));
            vst1q_u16(pu2_dst_tmp1 + 56, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a7, inv_f0), b7, f0), 5));
            pu2_dst_tmp1 += dst_strd;

            a0 = vld1q_u16(p1);
            a1 = vld1q_u16(p1 + 8);
            a2 = vld1q_u16(p1 + 16);
            a3 = vld1q_u16(p1 + 24);
            a4 = vld1q_u16(p1 + 32);
            a5 = vld1q_u16(p1 + 40);
            a6 = vld1q_u16(p1 + 48);
            a7 = vld1q_u16(p1 + 56);

            b0 = vextq_u16(a0, a1, 2);
            b1 = vextq_u16(a1, a2, 2);
            b2 = vextq_u16(a2, a3, 2);
            b3 = vextq_u16(a3, a4, 2);
            b4 = vextq_u16(a4, a5, 2);
            b5 = vextq_u16(a5, a6, 2);
            b6 = vextq_u16(a6, a7, 2);
            b7 = vld1q_u16(p1 + 58);

            vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f1), b0, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 8, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f1), b1, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 16, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a2, inv_f1), b2, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 24, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a3, inv_f1), b3, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 32, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a4, inv_f1), b4, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 40, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a5, inv_f1), b5, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 48, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a6, inv_f1), b6, f1), 5));
            vst1q_u16(pu2_dst_tmp1 + 56, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a7, inv_f1), b7, f1), 5));
            pu2_dst_tmp1 += dst_strd;
        }
    }
    else
    {
        UWORD16 *p0, *p1, *p2, *p3;
        uint16x8_t f0, inv_f0, f1, inv_f1, f2, inv_f2, f3, inv_f3;
        uint16x8_t a0, b0, a1, b1, a2, b2, a3, b3;

        pos = intra_pred_ang;
        p0 = pu2_ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f0 = vdupq_n_u16((uint16_t)fract);
        inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p1 = pu2_ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f1 = vdupq_n_u16((uint16_t)fract);
        inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p2 = pu2_ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f2 = vdupq_n_u16((uint16_t)fract);
        inv_f2 = vdupq_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p3 = pu2_ref_base + ((pos >> 5) << 1);
        fract = pos & 31;
        f3 = vdupq_n_u16((uint16_t)fract);
        inv_f3 = vdupq_n_u16((uint16_t)(32 - fract));

        a0 = vld1q_u16(p0);
        b0 = vld1q_u16(p0 + 2);
        a1 = vld1q_u16(p1);
        b1 = vld1q_u16(p1 + 2);
        a2 = vld1q_u16(p2);
        b2 = vld1q_u16(p2 + 2);
        a3 = vld1q_u16(p3);
        b3 = vld1q_u16(p3 + 2);

        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f0), b0, f0), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f1), b1, f1), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a2, inv_f2), b2, f2), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1q_u16(pu2_dst_tmp1, vrshrq_n_u16(vmlaq_u16(vmulq_u16(a3, inv_f3), b3, f3), 5));
    }
}

