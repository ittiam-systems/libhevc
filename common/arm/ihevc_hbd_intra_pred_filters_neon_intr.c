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
*  ihevc_hbd_intra_pred_filters_neon_intr.c
*
* @brief
*  Contains function definitions for high bit depth (HBD) luma intra prediction
*  interpolation filters using ARM NEON intrinsics.
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_intra_pred_luma_ref_substitution_neonintr()
*  - ihevc_hbd_intra_pred_ref_filtering_neonintr()
*  - ihevc_hbd_intra_pred_luma_planar_neonintr()
*  - ihevc_hbd_intra_pred_luma_dc_neonintr()
*  - ihevc_hbd_intra_pred_luma_horz_neonintr()
*  - ihevc_hbd_intra_pred_luma_ver_neonintr()
*  - ihevc_hbd_intra_pred_luma_mode2_neonintr()
*  - ihevc_hbd_intra_pred_luma_mode_18_34_neonintr()
*  - ihevc_hbd_intra_pred_luma_mode_3_to_9_neonintr()
*  - ihevc_hbd_intra_pred_luma_mode_11_to_17_neonintr()
*  - ihevc_hbd_intra_pred_luma_mode_19_to_25_neonintr()
*  - ihevc_hbd_intra_pred_luma_mode_27_to_33_neonintr()
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
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_common_tables.h"

/*****************************************************************************/
/* Constant Macros                                                           */
/*****************************************************************************/
#define MAX_CU_SIZE 64
#define T32_4NT 128
#define T16_4NT 64
#define BIT_DEPTH_MAX 12

/*****************************************************************************/
/* Function Definitions                                                      */
/*****************************************************************************/

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for pu2_ref substitution
*
*
* @par Description:
*    Reference substitution process for samples unavailable for prediction
*    Refer to section 8.4.4.2.2
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
*  WORD32 transform Block size
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
void ihevc_hbd_intra_pred_luma_ref_substitution_neonintr(UWORD16 *pu2_top_left,
                                                         UWORD16 *pu2_top,
                                                         UWORD16 *pu2_left,
                                                         WORD32 src_strd,
                                                         WORD32 nt,
                                                         WORD32 nbr_flags,
                                                         UWORD16 *pu2_dst,
                                                         WORD32 dst_strd,
                                                         UWORD8 bit_depth)
{
    UWORD16 pu2_ref;
    WORD32 dc_val, i;
    WORD32 total_samples = (4 * nt) + 1;
    WORD32 two_nt = 2 * nt;
    WORD32 three_nt = 3 * nt;
    WORD32 get_bits;
    WORD32 next;
    WORD32 bot_left, left, top, tp_right, tp_left;
    WORD32 idx, nbr_id_from_bl, frwd_nbr_flag;
    UNUSED(dst_strd);
    dc_val = 1 << (bit_depth - 1);

    /* Neighbor Flag Structure*/
    /*    Top-Left | Top-Right | Top | Left | Bottom-Left
              1         4         4     4         4
     */

    /* If no neighbor flags are present, fill the neighbor samples with DC value */
    if(nbr_flags == 0)
    {
        for(i = 0; i < total_samples; i++)
        {
            pu2_dst[i] = dc_val;
        }
    }
    else
    {
        if(nt <= 8)
        {
            /* 1 bit extraction for all the neighboring blocks */
            tp_left = (nbr_flags & 0x10000) >> 16;
            bot_left = (nbr_flags & 0x8) >> 3;
            left = (nbr_flags & 0x80) >> 7;
            top = (nbr_flags & 0x100) >> 8;
            tp_right = (nbr_flags & 0x1000) >> 12;

            /* Else fill the corresponding samples */
            if(tp_left)
                pu2_dst[two_nt] = *pu2_top_left;
            else
                pu2_dst[two_nt] = 0;

            if(left)
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt - 1 - i] = pu2_left[i * src_strd];
            }
            else
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt - 1 - i] = 0;
            }

            if(bot_left)
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt - 1 - i] = pu2_left[i * src_strd];
            }
            else
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt - 1 - i] = 0;
            }

            if(top)
            {
                if(0 == (nt & 7))
                    vst1q_u16(&pu2_dst[two_nt + 1], vld1q_u16(pu2_top));
                else
                    vst1_u16(&pu2_dst[two_nt + 1], vld1_u16(pu2_top));
            }
            else
            {
                if(0 == (nt & 7))
                    vst1q_u16(&pu2_dst[two_nt + 1], vdupq_n_u16(0));
                else
                    vst1_u16(&pu2_dst[two_nt + 1], vdup_n_u16(0));
            }

            if(tp_right)
            {
                if(0 == (nt & 7))
                    vst1q_u16(&pu2_dst[two_nt + 1 + nt], vld1q_u16(pu2_top + nt));
                else
                    vst1_u16(&pu2_dst[two_nt + 1 + nt], vld1_u16(pu2_top + nt));
            }
            else
            {
                if(0 == (nt & 7))
                    vst1q_u16(&pu2_dst[two_nt + 1 + nt], vdupq_n_u16(0));
                else
                    vst1_u16(&pu2_dst[two_nt + 1 + nt], vdup_n_u16(0));
            }

            next = 1;

            /* If bottom -left is not available, reverse substitution process*/
            if(bot_left == 0)
            {
                WORD32 a_nbr_flag[5] = { bot_left, left, tp_left, top, tp_right };

                /* Check for the 1st available sample from bottom-left*/
                while(!a_nbr_flag[next])
                    next++;

                /* If Left, top-left are available*/
                if(next <= 2)
                {
                    idx = nt * next;
                    pu2_ref = pu2_dst[idx];
                    for(i = 0; i < idx; i++)
                        pu2_dst[i] = pu2_ref;
                }
                else /* If top, top-right are available */
                {
                    /* Idx is changed to copy 1 pixel value for top-left ,if top-left is not available*/
                    idx = (nt * (next - 1)) + 1;
                    pu2_ref = pu2_dst[idx];
                    for(i = 0; i < idx; i++)
                        pu2_dst[i] = pu2_ref;
                }
            }

            /* Forward Substitution Process */
            /* If left is Unavailable, copy the last bottom-left value */

            if(left == 0)
            {
                if(0 == (nt & 7))
                {
                    vst1q_u16(pu2_dst + nt, vdupq_n_u16(pu2_dst[nt - 1]));
                }
                else
                {
                    vst1_u16(pu2_dst + nt, vdup_n_u16(pu2_dst[nt - 1]));
                }
            }
            if(tp_left == 0)
                pu2_dst[two_nt] = pu2_dst[two_nt - 1];
            if(top == 0)
            {
                if(0 == (nt & 7))
                {
                    vst1q_u16(pu2_dst + two_nt + 1, vdupq_n_u16(pu2_dst[two_nt]));
                }
                else
                {
                    vst1_u16(pu2_dst + two_nt + 1, vdup_n_u16(pu2_dst[two_nt]));
                }
            }
            if(tp_right == 0)
            {
                if(0 == (nt & 7))
                {
                    vst1q_u16(pu2_dst + three_nt + 1, vdupq_n_u16(pu2_dst[three_nt]));
                }
                else
                {
                    vst1_u16(pu2_dst + three_nt + 1, vdup_n_u16(pu2_dst[three_nt]));
                }
            }
        }
        if(nt == 16)
        {
            WORD32 nbr_flags_temp = 0;
            nbr_flags_temp = ((nbr_flags & 0xC) >> 2) + ((nbr_flags & 0xC0) >> 4)
                            + ((nbr_flags & 0x300) >> 4)
                            + ((nbr_flags & 0x3000) >> 6)
                            + ((nbr_flags & 0x10000) >> 8);

            /* Else fill the corresponding samples */
            if(nbr_flags & 0x10000)
                pu2_dst[two_nt] = *pu2_top_left;
            else
                pu2_dst[two_nt] = 0;

            if(nbr_flags & 0xC0)
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt - 1 - i] = pu2_left[i * src_strd];
            }
            else
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt - 1 - i] = 0;
            }

            if(nbr_flags & 0xC)
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt - 1 - i] = pu2_left[i * src_strd];
            }
            else
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt - 1 - i] = 0;
            }

            if(nbr_flags & 0x300)
            {
                for(i = 0; i < nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], vld1q_u16(&pu2_top[i]));
            }
            else
            {
                uint16x8_t zero_u16 = vdupq_n_u16(0);
                for(i = 0; i < nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], zero_u16);
            }

            if(nbr_flags & 0x3000)
            {
                for(i = nt; i < two_nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], vld1q_u16(&pu2_top[i]));
            }
            else
            {
                uint16x8_t zero_u16 = vdupq_n_u16(0);
                for(i = nt; i < two_nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], zero_u16);
            }

            /* compute trailing zeors based on nbr_flag for substitution process of below left see section .*/
            /* as each bit in nbr flags corresponds to 8 pels for bot_left, left, top and topright but 1 pel for topleft */
            {
                nbr_id_from_bl = look_up_trailing_zeros(nbr_flags_temp & 0XF) * 8; /* for below left and left */

                if(nbr_id_from_bl == 64)
                    nbr_id_from_bl = 32;

                if(nbr_id_from_bl == 32)
                {
                    /* for top left : 1 pel per nbr bit */
                    if(!((nbr_flags_temp >> 8) & 0x1))
                    {
                        nbr_id_from_bl++;
                        nbr_id_from_bl += look_up_trailing_zeros((nbr_flags_temp >> 4) & 0xF) * 8; /* top and top right;  8 pels per nbr bit */
                    }
                }
                /* Reverse Substitution Process*/
                if(nbr_id_from_bl)
                {
                    /* Replicate the bottom-left and subsequent unavailable pixels with the 1st available pixel above */
                    pu2_ref = pu2_dst[nbr_id_from_bl];
                    for(i = (nbr_id_from_bl - 1); i >= 0; i--)
                    {
                        pu2_dst[i] = pu2_ref;
                    }
                }
            }

            /* for the loop of 4*Nt+1 pixels (excluding pixels computed from reverse substitution) */
            while(nbr_id_from_bl < ((T16_4NT) + 1))
            {
                /* To Obtain the next unavailable idx flag after reverse neighbor substitution  */
                /* Devide by 8 to obtain the original index */
                frwd_nbr_flag = (nbr_id_from_bl >> 3); /*+ (nbr_id_from_bl & 0x1);*/

                /* The Top-left flag is at the last bit location of nbr_flags*/
                if(nbr_id_from_bl == (T16_4NT / 2))
                {
                    get_bits = GET_BIT(nbr_flags_temp, 8);

                    /* only pel substitution for TL */
                    if(!get_bits)
                        pu2_dst[nbr_id_from_bl] = pu2_dst[nbr_id_from_bl - 1];
                }
                else
                {
                    get_bits = GET_BIT(nbr_flags_temp, frwd_nbr_flag);
                    if(!get_bits)
                    {
                        /* 8 pel substitution (other than TL) */
                        pu2_ref = pu2_dst[nbr_id_from_bl - 1];
                        for(i = 0; i < 8; i++)
                            pu2_dst[nbr_id_from_bl + i] = pu2_ref;
                    }
                }
                nbr_id_from_bl += (nbr_id_from_bl == (T16_4NT / 2)) ? 1 : 8;
            }
        }

        if(nt == 32)
        {
            /* Else fill the corresponding samples */
            if(nbr_flags & 0x10000)
                pu2_dst[two_nt] = *pu2_top_left;
            else
                pu2_dst[two_nt] = 0;

            if(nbr_flags & 0xF0)
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt - 1 - i] = pu2_left[i * src_strd];
            }
            else
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt - 1 - i] = 0;
            }

            if(nbr_flags & 0xF)
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt - 1 - i] = pu2_left[i * src_strd];
            }
            else
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt - 1 - i] = 0;
            }

            if(nbr_flags & 0xF00)
            {
                for(i = 0; i < nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], vld1q_u16(&pu2_top[i]));
            }
            else
            {
                uint16x8_t zero_u16 = vdupq_n_u16(0);
                for(i = 0; i < nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], zero_u16);
            }

            if(nbr_flags & 0xF000)
            {
                for(i = nt; i < two_nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], vld1q_u16(&pu2_top[i]));
            }
            else
            {
                uint16x8_t zero_u16 = vdupq_n_u16(0);
                for(i = nt; i < two_nt; i += 8)
                    vst1q_u16(&pu2_dst[two_nt + 1 + i], zero_u16);
            }

            /* compute trailing ones based on mbr_flag for substitution process of below left see section .*/
            /* as each bit in nbr flags corresponds to 8 pels for bot_left, left, top and topright but 1 pel for topleft */
            {
                nbr_id_from_bl = look_up_trailing_zeros((nbr_flags & 0XFF)) * 8; /* for below left and left */

                if(nbr_id_from_bl == 64)
                {
                    /* for top left : 1 pel per nbr bit */
                    if(!((nbr_flags >> 16) & 0x1))
                    {
                        /* top left not available */
                        nbr_id_from_bl++;
                        /* top and top right;  8 pels per nbr bit */
                        nbr_id_from_bl += look_up_trailing_zeros((nbr_flags >> 8) & 0xFF) * 8;
                    }
                }
                /* Reverse Substitution Process*/
                if(nbr_id_from_bl)
                {
                    /* Replicate the bottom-left and subsequent unavailable pixels with the 1st available pixel above */
                    pu2_ref = pu2_dst[nbr_id_from_bl];
                    for(i = (nbr_id_from_bl - 1); i >= 0; i--)
                        pu2_dst[i] = pu2_ref;
                }
            }

            /* for the loop of 4*Nt+1 pixels (excluding pixels computed from reverse substitution) */
            while(nbr_id_from_bl < ((T32_4NT) + 1))
            {
                /* To Obtain the next unavailable idx flag after reverse neighbor substitution  */
                /* Devide by 8 to obtain the original index */
                frwd_nbr_flag = (nbr_id_from_bl >> 3); /*+ (nbr_id_from_bl & 0x1);*/

                /* The Top-left flag is at the last bit location of nbr_flags*/
                if(nbr_id_from_bl == (T32_4NT / 2))
                {
                    get_bits = GET_BIT(nbr_flags, 16);
                    /* only pel substitution for TL */
                    if(!get_bits)
                        pu2_dst[nbr_id_from_bl] = pu2_dst[nbr_id_from_bl - 1];
                }
                else
                {
                    get_bits = GET_BIT(nbr_flags, frwd_nbr_flag);
                    if(!get_bits)
                    {
                        /* 8 pel substitution (other than TL) */
                        pu2_ref = pu2_dst[nbr_id_from_bl - 1];
                        for(i = 0; i < 8; i++)
                            pu2_dst[nbr_id_from_bl + i] = pu2_ref;
                    }
                }
                nbr_id_from_bl += (nbr_id_from_bl == (T32_4NT / 2)) ? 1 : 8;
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for ref_filtering
*
*
* @par Description:
*    Reference DC filtering for neighboring samples dependent on TU size and
*    mode Refer to section 8.4.4.2.3 in the standard
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] nt
*  integer Transform Block size
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] mode
*  integer intraprediction mode
*
* @param[in] intra_smoothing_flags
*  integer bit 3 indicates if intra smoothing is enabled/disabled
*          unconditionally. this is applicable to frext profiles only
*          bit 0 indicates strong intra smoothing enabled/disabled
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
void ihevc_hbd_intra_pred_ref_filtering_neonintr(UWORD16 *pu2_src,
                                                 WORD32 nt,
                                                 UWORD16 *pu2_dst,
                                                 WORD32 mode,
                                                 WORD32 intra_smoothing_flags,
                                                 UWORD8 bit_depth)
{
    WORD32 filter_flag;
    WORD32 i = 0;
    WORD32 four_nt = 4 * nt;

    WORD32 src_4nt;
    WORD32 src_0nt;
    /* Naming has been made as per the functionlity it has, For eg. pu2_src_tmp_1 is denoting pu2_src + 1   */
    /* src_val_1 to load value from pointer pu2_src_tmp_1, add_res has the result of adding 2 values        */
    UWORD16 *pu2_src_tmp_0 = pu2_src;
    UWORD16 *pu2_src_tmp_1;
    UWORD16 *pu2_src_tmp_2;
    UWORD16 *pu2_dst_tmp_0 = pu2_dst;
    UWORD16 *pu2_dst_tmp_1;

    uint16x8_t src_val_0, src_val_2;
    uint16x8_t src_val_1, shift_res;
    uint16x8_t dup_const_2;
    uint16x8_t mul_res, add_res;
    WORD32 bi_linear_int_flag = 0;
    WORD32 abs_cond_left_flag = 0;
    WORD32 abs_cond_top_flag = 0;
    WORD32 dc_val = 1 << (bit_depth - 5);
    WORD32 intra_smoothing_disabled = (intra_smoothing_flags >> 3);
    WORD32 strong_intra_smoothing_enable_flag = intra_smoothing_flags & 1;

    shift_res = vdupq_n_u16(0);
    filter_flag = intra_smoothing_disabled ?
                    0 : (gau1_intra_pred_ref_filter[mode] & (1 << (CTZ(nt) - 2)));
    if(0 == filter_flag)
    {
        if(pu2_src == pu2_dst)
        {
            return;
        }
        else
        {
            for(i = four_nt; i > 0; i -= 8)
            {
                src_val_0 = vld1q_u16(pu2_src_tmp_0);
                pu2_src_tmp_0 += 8;
                vst1q_u16(pu2_dst_tmp_0, src_val_0);
                pu2_dst_tmp_0 += 8;
            }
            pu2_dst[four_nt] = pu2_src[four_nt];
        }
    }
    else
    {
        /* If strong intra smoothin is enabled and transform size is 32 */
        if((1 == strong_intra_smoothing_enable_flag) && (32 == nt))
        {
            /*Strong Intra Filtering*/
            abs_cond_top_flag = (ABS(pu2_src[2 * nt] + pu2_src[4 * nt]
                            - (2 * pu2_src[3 * nt]))) < dc_val;
            abs_cond_left_flag = (ABS(pu2_src[2 * nt] + pu2_src[0]
                            - (2 * pu2_src[nt]))) < dc_val;

            bi_linear_int_flag = ((1 == abs_cond_left_flag)
                            && (1 == abs_cond_top_flag));
        }

        src_4nt = pu2_src[4 * nt];
        src_0nt = pu2_src[0];
        /* Strong filtering of reference samples */
        if(1 == bi_linear_int_flag)
        {
            WORD32 two_nt = four_nt >> 1;

            WORD32 pu2_src_0_val = pu2_src[0];
            WORD32 pu2_src_2_nt_val = pu2_src[2 * nt];
            WORD32 pu2_src_4_nt_val = pu2_src[4 * nt];

            WORD32 prod_two_nt_src_0_val = two_nt * pu2_src_0_val;
            uint32x4_t prod_two_nt_src_0_val_t = vdupq_n_u32(prod_two_nt_src_0_val);

            WORD32 prod_two_nt_src_2_nt_val = two_nt * pu2_src_2_nt_val;
            uint32x4_t prod_two_nt_src_2_nt_val_t = vdupq_n_u32(prod_two_nt_src_2_nt_val);

            const UWORD8 *const_col_i;
            uint16x8_t const_col_i_val;
            uint32x4_t prod_val_1_lo, prod_val_1_hi;
            uint32x4_t prod_val_2_lo, prod_val_2_hi;
            uint32x4_t prod_val_3_lo, prod_val_3_hi;
            uint32x4_t prod_val_4_lo, prod_val_4_hi;
            uint16x8_t res_val_1;
            uint16x8_t res_val_2;
            uint16x4_t pu2_src_0_val_t = vdup_n_u16(pu2_src_0_val);
            uint16x4_t pu2_src_2_nt_val_t = vdup_n_u16(pu2_src_2_nt_val);
            uint16x4_t pu2_src_4_nt_val_t = vdup_n_u16(pu2_src_4_nt_val);
            pu2_dst_tmp_0 = pu2_dst + 1;
            pu2_dst_tmp_1 = pu2_dst + two_nt + 1;

            const_col_i = gau1_ihevc_planar_factor + 1;

            for(i = two_nt; i > 0; i -= 8)
            {
                const_col_i_val = vmovl_u8(vld1_u8(const_col_i));
                const_col_i += 8;

                prod_val_1_lo = vmlsl_u16(prod_two_nt_src_0_val_t, vget_low_u16(const_col_i_val), pu2_src_0_val_t);
                prod_val_2_lo = vmlal_u16(prod_val_1_lo, vget_low_u16(const_col_i_val), pu2_src_2_nt_val_t);
                prod_val_1_hi = vmlsl_u16(prod_two_nt_src_0_val_t, vget_high_u16(const_col_i_val), pu2_src_0_val_t);
                prod_val_2_hi = vmlal_u16(prod_val_1_hi, vget_high_u16(const_col_i_val), pu2_src_2_nt_val_t);

                res_val_1 = vcombine_u16(vrshrn_n_u32(prod_val_2_lo, 6), vrshrn_n_u32(prod_val_2_hi, 6));
                prod_val_3_lo = vmlsl_u16(prod_two_nt_src_2_nt_val_t, vget_low_u16(const_col_i_val), pu2_src_2_nt_val_t);
                prod_val_3_hi = vmlsl_u16(prod_two_nt_src_2_nt_val_t, vget_high_u16(const_col_i_val), pu2_src_2_nt_val_t);

                vst1q_u16(pu2_dst_tmp_0, res_val_1);
                pu2_dst_tmp_0 += 8;
                prod_val_4_lo = vmlal_u16(prod_val_3_lo, vget_low_u16(const_col_i_val), pu2_src_4_nt_val_t);
                prod_val_4_hi = vmlal_u16(prod_val_3_hi, vget_high_u16(const_col_i_val), pu2_src_4_nt_val_t);

                res_val_2 = vcombine_u16(vrshrn_n_u32(prod_val_4_lo, 6), vrshrn_n_u32(prod_val_4_hi, 6));
                vst1q_u16(pu2_dst_tmp_1, res_val_2);
                pu2_dst_tmp_1 += 8;
            }
            pu2_dst[2 * nt] = pu2_src[2 * nt];
        }
        else
        {
            pu2_src_tmp_1 = pu2_src + 1;
            pu2_src_tmp_2 = pu2_src + 2;
            pu2_dst_tmp_0 += 1;

            dup_const_2 = vdupq_n_u16(2);

            /* Extremities Untouched*/
            pu2_dst[0] = pu2_src[0];

            /* To avoid the issue when the dest and src has the same pointer this load has been done
             * outside and the 2nd consecutive load is done before the store of the 1st */

            /* Perform bilinear filtering of Reference Samples */
            for(i = (four_nt - 1); i > 0; i -= 8)
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
                if(i < four_nt - 1)
                {
                    vst1q_u16(pu2_dst_tmp_0, shift_res);
                    pu2_dst_tmp_0 += 8;
                }

                add_res = vaddq_u16(src_val_0, src_val_2);

                mul_res = vmlaq_u16(add_res, src_val_1, dup_const_2);
                shift_res = vrshrq_n_u16(mul_res, 2);
            }
            vst1q_u16(pu2_dst_tmp_0, shift_res);
            pu2_dst_tmp_0 += 8;
        }
        pu2_dst[4 * nt] = src_4nt;
        pu2_dst[0] = src_0nt;
    }
}

/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for luma planar
*
* @par Description:
*      Planar Intraprediction with reference neighboring samples location
*      pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'
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
*  integer intra prediction mode
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
void ihevc_hbd_intra_pred_luma_planar_neonintr(UWORD16 *pu2_ref,
                                               WORD32 src_strd,
                                               UWORD16 *pu2_dst,
                                               WORD32 dst_strd,
                                               WORD32 nt,
                                               WORD32 mode,
                                               UWORD8 bit_depth)
{
    /* named it in the way (nt - 1 - col) --> const_nt_1_col(const denotes g_ihevc_planar_factor)   */
    /* load const_nt_1_col values into a d register                                                 */
    /* named it in the way pu2_ref[nt - 1] --> pu2_ref_nt_1                                         */
    /* the value of pu2_ref_nt_1 is duplicated to d register hence pu2_ref_nt_1_dup                 */
    /* log2nt + 1 is taken care while assigning the values itself                                   */
    /* In width multiple of 4 case the row also has been unrolled by 2 and store has been taken care*/

    WORD32 row, col = 0;
    WORD32 log2nt_plus1 = 6;
    WORD32 two_nt, three_nt;
    UWORD16 *pu2_ref_two_nt_1;
    UWORD16 *pu2_dst_tmp;
    const UWORD8 *const_nt_1_col;
    uint16x8_t const_nt_1_col_t;
    const UWORD8 *const_col_1;
    uint16x8_t const_col_1_t;
    uint16_t const_nt_1_row;
    uint16x8_t const_nt_1_row_dup;
    uint16_t const_row_1;
    uint16x8_t const_row_1_dup;
    uint16_t const_nt = nt;
    uint16x8_t const_nt_dup;
    uint16_t pu2_ref_nt_1 = pu2_ref[nt - 1];
    uint16x8_t pu2_ref_nt_1_dup;
    uint16_t pu2_ref_two_nt_1_row;
    uint16_t pu2_ref_three_nt_1;
    uint16x8_t pu2_ref_two_nt_1_row_dup;
    uint16x8_t pu2_ref_two_nt_1_t;
    uint16x8_t pu2_ref_three_nt_1_dup;
    uint16x8_t prod_t1;
    uint16x8_t prod_t2;
    uint16x8_t sto_res;
    int16x8_t log2nt_dup;
    UNUSED(src_strd);
    UNUSED(mode);
    UNUSED(bit_depth);

    log2nt_plus1 = 32 - CLZ(nt);
    two_nt = 2 * nt;
    three_nt = 3 * nt;
    /* loops have been unrolld considering the fact width is multiple of 8  */
    if(0 == (nt & 7))
    {
        pu2_dst_tmp = pu2_dst;

        pu2_ref_three_nt_1 = pu2_ref[three_nt + 1];

        pu2_ref_nt_1_dup = vdupq_n_u16(pu2_ref_nt_1);
        const_nt_dup = vdupq_n_u16(const_nt);

        log2nt_dup = vdupq_n_s16(log2nt_plus1);
        log2nt_dup = vnegq_s16(log2nt_dup);

        pu2_ref_three_nt_1_dup = vdupq_n_u16(pu2_ref_three_nt_1);

        for(row = 0; row < nt; row++)
        {
            pu2_ref_two_nt_1_row = pu2_ref[two_nt - 1 - row];
            pu2_ref_two_nt_1_row_dup = vdupq_n_u16(pu2_ref_two_nt_1_row);

            const_nt_1_row = nt - 1 - row;
            const_nt_1_row_dup = vdupq_n_u16(const_nt_1_row);

            const_row_1 = row + 1;
            const_row_1_dup = vdupq_n_u16(const_row_1);

            const_nt_1_col = gau1_ihevc_planar_factor + nt - 8;

            const_col_1 = gau1_ihevc_planar_factor + 1;
            pu2_ref_two_nt_1 = pu2_ref + two_nt + 1;

            for(col = nt; col > 0; col -= 8)
            {
                const_nt_1_col_t = vmovl_u8(vrev64_u8(vld1_u8(const_nt_1_col)));
                const_nt_1_col -= 8;

                const_col_1_t = vmovl_u8(vld1_u8(const_col_1));
                const_col_1 += 8;
                prod_t1 = vmulq_u16(const_nt_1_col_t, pu2_ref_two_nt_1_row_dup);

                pu2_ref_two_nt_1_t = vld1q_u16(pu2_ref_two_nt_1);
                pu2_ref_two_nt_1 += 8;
                prod_t2 = vmulq_u16(const_col_1_t, pu2_ref_three_nt_1_dup);

                prod_t1 = vmlaq_u16(prod_t1, const_nt_1_row_dup, pu2_ref_two_nt_1_t);
                prod_t2 = vmlaq_u16(prod_t2, const_row_1_dup, pu2_ref_nt_1_dup);
                prod_t1 = vaddq_u16(prod_t1, const_nt_dup);
                prod_t1 = vaddq_u16(prod_t1, prod_t2);

                sto_res = vshlq_u16(prod_t1, log2nt_dup);
                vst1q_u16(pu2_dst_tmp, sto_res);
                pu2_dst_tmp += 8;
            }
            pu2_dst_tmp += dst_strd - nt;
        }
    }
    /* loops have been unrolld considering the fact width is multiple of 4  */
    /* If column is multiple of 4 then height should be multiple of 2       */
    else
    {
        uint16x8_t const_row_1_dup1;
        uint16x4_t pu2_ref_two_nt_1_low;
        uint8x8_t const_nt_1_col_u8;
        uint8x8_t const_nt_1_col_u8_1;
        uint8x8_t const_col_1_u8;
        uint8x8_t const_col_1_u8_1;
        uint16x8_t pu2_ref_two_nt_1_row_dup1;
        uint16x8_t const_nt_1_row_dup1;

        pu2_ref_three_nt_1 = pu2_ref[three_nt + 1];

        pu2_ref_nt_1_dup = vdupq_n_u16(pu2_ref_nt_1);
        const_nt_dup = vdupq_n_u16(const_nt);

        log2nt_dup = vdupq_n_s16(log2nt_plus1);
        log2nt_dup = vnegq_s16(log2nt_dup);

        pu2_ref_three_nt_1_dup = vdupq_n_u16(pu2_ref_three_nt_1);

        for(row = 0; row < nt; row += 2)
        {
            pu2_ref_two_nt_1_row = pu2_ref[two_nt - 1 - row];
            pu2_ref_two_nt_1_row_dup = vdupq_n_u16(pu2_ref_two_nt_1_row);
            pu2_ref_two_nt_1_row = pu2_ref[two_nt - 2 - row];
            pu2_ref_two_nt_1_row_dup1 = vdupq_n_u16(pu2_ref_two_nt_1_row);
            pu2_ref_two_nt_1_row_dup = vextq_u16(pu2_ref_two_nt_1_row_dup, pu2_ref_two_nt_1_row_dup1, 4);

            const_nt_1_row = nt - 1 - row;
            const_nt_1_row_dup = vdupq_n_u16(const_nt_1_row);
            const_nt_1_row = nt - 2 - row;
            const_nt_1_row_dup1 = vdupq_n_u16(const_nt_1_row);
            const_nt_1_row_dup = vextq_u16(const_nt_1_row_dup, const_nt_1_row_dup1, 4);

            const_row_1 = row + 1;
            const_row_1_dup = vdupq_n_u16(const_row_1);
            const_row_1 = row + 2;
            const_row_1_dup1 = vdupq_n_u16(const_row_1);
            const_row_1_dup = vextq_u16(const_row_1_dup, const_row_1_dup1, 4);

            const_nt_1_col = gau1_ihevc_planar_factor + nt - 4;

            const_col_1 = gau1_ihevc_planar_factor + 1;

            pu2_ref_two_nt_1 = pu2_ref + two_nt + 1;

            const_nt_1_col_u8 = vld1_u8(const_nt_1_col);
            const_nt_1_col_u8 = vrev64_u8(const_nt_1_col_u8);

            const_col_1_u8 = vld1_u8(const_col_1);
            const_nt_1_col_u8_1 = vreinterpret_u8_u64(vshr_n_u64(vreinterpret_u64_u8(const_nt_1_col_u8), 32));

            pu2_dst_tmp = pu2_dst;
            const_nt_1_col_u8 = vext_u8(const_nt_1_col_u8, const_nt_1_col_u8_1, 4);
            const_nt_1_col_t = vmovl_u8(const_nt_1_col_u8);

            const_col_1_u8_1 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(const_col_1_u8), 32));
            prod_t1 = vmulq_u16(const_nt_1_col_t, pu2_ref_two_nt_1_row_dup);

            pu2_ref_two_nt_1_low = vld1_u16(pu2_ref_two_nt_1);
            const_col_1_u8 = vext_u8(const_col_1_u8_1, const_col_1_u8, 4);
            const_col_1_t = vmovl_u8(const_col_1_u8);

            pu2_ref_two_nt_1_t = vcombine_u16(pu2_ref_two_nt_1_low, pu2_ref_two_nt_1_low);
            prod_t2 = vmulq_u16(const_col_1_t, pu2_ref_three_nt_1_dup);

            prod_t2 = vmlaq_u16(prod_t2, const_row_1_dup, pu2_ref_nt_1_dup);

            prod_t1 = vmlaq_u16(prod_t1, const_nt_1_row_dup, pu2_ref_two_nt_1_t);
            prod_t1 = vaddq_u16(prod_t1, const_nt_dup);
            prod_t1 = vaddq_u16(prod_t1, prod_t2);

            sto_res = vshlq_u16(prod_t1, log2nt_dup);

            vst1_u16(pu2_dst, vget_low_u16(sto_res));
            pu2_dst_tmp += dst_strd;

            vst1_u16(pu2_dst_tmp, vget_high_u16(sto_res));

            pu2_dst += 2 * dst_strd;
        }
    }
}
/* INTRA_PRED_LUMA_PLANAR */

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma dc
*
* @par Description:
*    Intraprediction for DC mode with reference neighboring samples location
*    pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'
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
*  integer intra prediction mode
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
void ihevc_hbd_intra_pred_luma_dc_neonintr(UWORD16 *pu2_ref,
                                           WORD32 src_strd,
                                           UWORD16 *pu2_dst,
                                           WORD32 dst_strd,
                                           WORD32 nt,
                                           WORD32 mode,
                                           UWORD8 bit_depth)
{
    WORD32 dc_val = 0, two_dc_val = 0, three_dc_val = 0;
    WORD32 i = 0;
    WORD32 row = 0, col = 0, col_count;
    WORD32 log2nt_plus1 = 6;
    WORD32 two_nt = 0;
    uint16x8_t ref_load_q;
    uint16x8_t three_dc_val_t;
    uint16x8_t sto_res_tmp;
    uint16x8_t sto_res_tmp1;
    uint16x8_t sto_res_tmp2;
    uint16x8_t sto_res_tmp3;
    uint16x8_t sto_res_tmp4;
    uint16x8_t dc_val_t;

    UWORD16 *pu2_ref_tmp;
    UWORD16 *pu2_ref_tmp1;
    UWORD16 *pu2_dst_tmp;
    UWORD16 *pu2_dst_tmp1;
    UWORD16 *pu2_dst_tmp2;
    UNUSED(src_strd);
    UNUSED(mode);
    UNUSED(bit_depth);

    /* log2nt + 1 is taken care while assigning the values itself.          */
    log2nt_plus1 = 32 - CLZ(nt);

    /* loops have been unrolld considering the fact width is multiple of 8  */
    if(0 == (nt & 7))
    {
        uint16x8_t ref_load1;
        uint16x8_t ref_load2;
        uint32x4_t acc_dc_pair1;
        uint64x2_t acc_dc_pair2;
        uint64x1_t acc_dc = vdup_n_u64(col);

        two_nt = 2 * nt;
        pu2_ref_tmp = pu2_ref + nt;
        pu2_ref_tmp1 = pu2_ref + two_nt + 1;

        for(i = two_nt; i > nt; i -= 8)
        {
            ref_load1 = vld1q_u16(pu2_ref_tmp);
            pu2_ref_tmp += 8;
            acc_dc_pair1 = vpaddlq_u16(ref_load1);

            ref_load2 = vld1q_u16(pu2_ref_tmp1);
            pu2_ref_tmp1 += 8;

            acc_dc_pair1 = vpadalq_u16(acc_dc_pair1, ref_load2);
            acc_dc_pair2 = vpaddlq_u32(acc_dc_pair1);
            acc_dc = vadd_u64(acc_dc, vadd_u64(vget_low_u64(acc_dc_pair2), vget_high_u64(acc_dc_pair2)));
        }

        dc_val = (vget_lane_u32(vreinterpret_u32_u64(acc_dc), 0) + nt) >> (log2nt_plus1);
        dc_val_t = vdupq_n_u16(dc_val);
        two_dc_val = 2 * dc_val;
        three_dc_val = 3 * dc_val;
        three_dc_val += 2;

        three_dc_val_t = vdupq_n_u16((UWORD16)three_dc_val);
        pu2_ref_tmp = pu2_ref + two_nt + 1 + 0;
        pu2_dst_tmp = pu2_dst;

        if(nt == 32)
        {
            for(row = 0; row < 32; row += 4)
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
            for(col = nt; col > 0; col -= 8)
            {
                ref_load1 = vld1q_u16(pu2_ref_tmp);
                pu2_ref_tmp += 8;
                ref_load_q = vaddq_u16(ref_load1, three_dc_val_t);
                sto_res_tmp = vshrq_n_u16(ref_load_q, 2);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp);
                pu2_dst_tmp += 8;
            }

            pu2_ref_tmp = pu2_ref + two_nt - 9;
            pu2_dst_tmp = pu2_dst + dst_strd;
            col_count = nt - 8;

            /* Except the first row the remaining rows are done here                            */
            /* Both column and row has been unrolled by 8                                       */
            /* Store has been taken care for the unrolling                                      */
            /* Except the 1st column of the remaining rows(other than 1st row), the values are  */
            /* constant hence it is extracted with an constant value and stored                 */
            /* If the column is greater than 8, then the remaining values are constant which is */
            /* taken care in the inner for loop                                                 */

            for(row = nt; row > 0; row -= 8)
            {
                pu2_dst_tmp1 = pu2_dst_tmp + 8;
                ref_load1 = vld1q_u16(pu2_ref_tmp);
                pu2_ref_tmp -= 8;
                ref_load_q = vaddq_u16(ref_load1, three_dc_val_t);
                sto_res_tmp = vshrq_n_u16(ref_load_q, 2);

                sto_res_tmp1 = vextq_u16(sto_res_tmp, dc_val_t, 7);

                sto_res_tmp = vextq_u16(dc_val_t, sto_res_tmp, 7);
                sto_res_tmp2 = vextq_u16(sto_res_tmp, dc_val_t, 7);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp1);
                pu2_dst_tmp += dst_strd;

                sto_res_tmp = vextq_u16(dc_val_t, sto_res_tmp, 7);
                sto_res_tmp3 = vextq_u16(sto_res_tmp, dc_val_t, 7);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp2);
                pu2_dst_tmp += dst_strd;

                sto_res_tmp = vextq_u16(dc_val_t, sto_res_tmp, 7);
                sto_res_tmp4 = vextq_u16(sto_res_tmp, dc_val_t, 7);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp3);
                pu2_dst_tmp += dst_strd;

                sto_res_tmp = vextq_u16(dc_val_t, sto_res_tmp, 7);
                sto_res_tmp1 = vextq_u16(sto_res_tmp, dc_val_t, 7);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp4);
                pu2_dst_tmp += dst_strd;

                sto_res_tmp = vextq_u16(dc_val_t, sto_res_tmp, 7);
                sto_res_tmp2 = vextq_u16(sto_res_tmp, dc_val_t, 7);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp1);
                pu2_dst_tmp += dst_strd;

                sto_res_tmp = vextq_u16(dc_val_t, sto_res_tmp, 7);
                sto_res_tmp3 = vextq_u16(sto_res_tmp, dc_val_t, 7);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp2);
                pu2_dst_tmp += dst_strd;

                sto_res_tmp = vextq_u16(dc_val_t, sto_res_tmp, 7);
                sto_res_tmp4 = vextq_u16(sto_res_tmp, dc_val_t, 7);
                vst1q_u16(pu2_dst_tmp, sto_res_tmp3);
                pu2_dst_tmp += dst_strd;
                /* For last set of 8 rows only 7 rows need to be updated since first row is already written */
                if(row != 8)
                    vst1q_u16(pu2_dst_tmp, sto_res_tmp4);
                pu2_dst_tmp += dst_strd;

                for(col = col_count; col > 0; col -= 8)
                {
                    pu2_dst_tmp2 = pu2_dst_tmp1;
                    vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 += dst_strd;
                    vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 += dst_strd;
                    vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 += dst_strd;
                    vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 += dst_strd;
                    vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 += dst_strd;
                    vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 += dst_strd;
                    vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 += dst_strd;

                    /* For last set of 8 rows only 7 rows need to be updated since first row is already written */
                    if(row != 8)
                        vst1q_u16(pu2_dst_tmp1, dc_val_t);
                    pu2_dst_tmp1 = pu2_dst_tmp2 + 8;
                }
            }
            pu2_dst[0] = (pu2_ref[two_nt - 1] + two_dc_val + pu2_ref[two_nt + 1] + 2) >> 2;
        }
    }
    /* loops have been unrolld considering the fact width is multiple of 4  */
    else
    {
        uint16x4_t ref_l = vld1_u16(pu2_ref + 4);
        uint16x4_t ref_t = vld1_u16(pu2_ref + 9);
        uint32x2_t sum_pair = vpaddl_u16(vadd_u16(ref_l, ref_t));
        uint16x4_t dc_val_d, three_dc_d, top_filt, left_filt;

        dc_val = (vget_lane_u32(vpadd_u32(sum_pair, sum_pair), 0) + 4) >> 3;
        two_dc_val = 2 * dc_val;
        three_dc_val = 3 * dc_val + 2;

        dc_val_d = vdup_n_u16((uint16_t)dc_val);
        three_dc_d = vdup_n_u16((uint16_t)three_dc_val);

        top_filt = vshr_n_u16(vadd_u16(ref_t, three_dc_d), 2);
        top_filt = vset_lane_u16((uint16_t)((pu2_ref[7] + two_dc_val + pu2_ref[9] + 2) >> 2), top_filt, 0);
        vst1_u16(pu2_dst, top_filt);

        left_filt = vshr_n_u16(vadd_u16(ref_l, three_dc_d), 2);
        vst1_u16(pu2_dst + dst_strd, vset_lane_u16(vget_lane_u16(left_filt, 2), dc_val_d, 0));
        vst1_u16(pu2_dst + 2 * dst_strd, vset_lane_u16(vget_lane_u16(left_filt, 1), dc_val_d, 0));
        vst1_u16(pu2_dst + 3 * dst_strd, vset_lane_u16(vget_lane_u16(left_filt, 0), dc_val_d, 0));
    }
}
/* INTRA_PRED_LUMA_DC */

/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for horizontal luma variable.
*
* @par Description:
*   Horizontal intraprediction with reference neighboring samples location
*   pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'
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
* @param[in] disable_boundary_filter
*  WORD32 boundary filter disable flag
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
void ihevc_hbd_intra_pred_luma_horz_neonintr(UWORD16 *pu2_ref,
                                             WORD32 src_strd,
                                             UWORD16 *pu2_dst,
                                             WORD32 dst_strd,
                                             WORD32 nt,
                                             WORD32 disable_boundary_filter,
                                             UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 two_nt;
    UNUSED(src_strd);

    two_nt = 2 * nt;

    UWORD16 *pu2_dst_tmp = pu2_dst;
    if(nt == 32)
    {
        UWORD16 *pu2_ref_tmp = pu2_ref + two_nt - 4;
        pu2_dst_tmp = pu2_dst;
        for(row = 0; row < 32; row += 4)
        {
            uint16x4_t ref_v = vld1_u16(pu2_ref_tmp);
            uint16x8_t d0 = vdupq_lane_u16(ref_v, 3);
            uint16x8_t d1 = vdupq_lane_u16(ref_v, 2);
            uint16x8_t d2 = vdupq_lane_u16(ref_v, 1);
            uint16x8_t d3 = vdupq_lane_u16(ref_v, 0);
            pu2_ref_tmp -= 4;

            vst1q_u16(pu2_dst_tmp, d0);
            vst1q_u16(pu2_dst_tmp + 8, d0);
            vst1q_u16(pu2_dst_tmp + 16, d0);
            vst1q_u16(pu2_dst_tmp + 24, d0);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, d1);
            vst1q_u16(pu2_dst_tmp + 8, d1);
            vst1q_u16(pu2_dst_tmp + 16, d1);
            vst1q_u16(pu2_dst_tmp + 24, d1);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, d2);
            vst1q_u16(pu2_dst_tmp + 8, d2);
            vst1q_u16(pu2_dst_tmp + 16, d2);
            vst1q_u16(pu2_dst_tmp + 24, d2);
            pu2_dst_tmp += dst_strd;

            vst1q_u16(pu2_dst_tmp, d3);
            vst1q_u16(pu2_dst_tmp + 8, d3);
            vst1q_u16(pu2_dst_tmp + 16, d3);
            vst1q_u16(pu2_dst_tmp + 24, d3);
            pu2_dst_tmp += dst_strd;
        }
    }
    else
    /* row loop has been unrolled, hence had pu2_ref_val1 and pu2_ref_val2 variables*/
    /* naming of variables made according to the operation(instructions) it performs*/
    /* (eg. shift_val which contains the shifted value,                             */
    /* add_sat which has add and saturated value)                                   */
    /* Loops are unrolled by 4 and 8 considering the fact the input width is either multiple of 4 or 8  */
    /* rows and columns are unrolled by 4, when the width is multiple of 4                              */
    {
        if(0 != (nt & 7))      /* cond for multiple of 4 */
        {
            UWORD16 *pu2_dst_4 = pu2_dst;
            uint16x4_t pu2_ref_val1, pu2_ref_val2;
            uint16x4_t dup_sub, round_val, dup_val;
            uint16x4_t dup_add;
            int16x4_t sub_val, shift_val, add_sat;
            int16x4_t zero_s16 = vdup_n_s16(0);
            int16x4_t max_s16 = vdup_n_s16((1 << bit_depth) - 1);

            pu2_ref_val2 = vld1_u16(pu2_ref + (two_nt - nt));

            if (disable_boundary_filter)
            {
                round_val = vdup_lane_u16(pu2_ref_val2, 3);
            }
            else
            {
                dup_sub = vdup_n_u16(pu2_ref[two_nt]);
                dup_add = vdup_n_u16(pu2_ref[two_nt - 1]);
                pu2_ref_val1 = vld1_u16(pu2_ref + (two_nt + 1));
                sub_val = vsub_s16(vreinterpret_s16_u16(pu2_ref_val1), vreinterpret_s16_u16(dup_sub));
                shift_val = vshr_n_s16(sub_val, 1);

                add_sat = vadd_s16(shift_val, vreinterpret_s16_u16(dup_add));
                add_sat = vmin_s16(vmax_s16(add_sat, zero_s16), max_s16);
                round_val = vreinterpret_u16_s16(add_sat);
            }

            vst1_u16(pu2_dst_4, round_val);
            pu2_dst_4 += dst_strd;

            dup_val = vdup_lane_u16(pu2_ref_val2, 2);
            vst1_u16(pu2_dst_4, dup_val);
            pu2_dst_4 += dst_strd;

            dup_val = vdup_lane_u16(pu2_ref_val2, 1);
            vst1_u16(pu2_dst_4, dup_val);
            pu2_dst_4 += dst_strd;

            dup_val = vdup_lane_u16(pu2_ref_val2, 0);
            vst1_u16(pu2_dst_4, dup_val);
        }

        /* dup_1 - dup_8 are variables to load the duplicated values from the loaded source */
        /* naming of variables made according to the operation(instructions) it performs    */
        /* Loops are unrolled by 4 and 8 considering the fact the input width is either multiple of 4 or 8  */
        /* rows and columns are unrolled by 8, when the width is multiple of 8                              */

        else
        {
            UWORD16 *pu2_ref_tmp_1 = pu2_ref + (two_nt + 1);
            UWORD16 *pu2_ref_tmp_2 = pu2_ref + (two_nt - 1);

            UWORD16 *pu2_dst_tmp_1 = pu2_dst;
            UWORD16 *pu2_dst_tmp_2 = pu2_dst + dst_strd;

            uint16x8_t dup_sub, src_tmp, src_tmp_1, round_val, dup_1, dup_2, dup_3, dup_4, dup_5, dup_6, dup_7, dup_8, rev_res;
            uint16x8_t dup_add;
            int16x8_t sub_res, shift_res, add_res;
            int16x8_t zero_s16 = vdupq_n_s16(0);
            int16x8_t max_s16 = vdupq_n_s16((1 << bit_depth) - 1);

            dup_sub = vdupq_n_u16(pu2_ref[two_nt]);
            dup_add = vdupq_n_u16(pu2_ref[two_nt - 1]);

            for(col = nt; col > 0; col -= 8)
            {
                if (disable_boundary_filter)
                {
                    round_val = dup_add;
                }
                else
                {
                    src_tmp = vld1q_u16(pu2_ref_tmp_1);
                    pu2_ref_tmp_1 += 8;

                    sub_res = vsubq_s16(vreinterpretq_s16_u16(src_tmp), vreinterpretq_s16_u16(dup_sub));
                    shift_res = vshrq_n_s16(sub_res, 1);
                    add_res = vaddq_s16(shift_res, vreinterpretq_s16_u16(dup_add));
                    add_res = vminq_s16(vmaxq_s16(add_res, zero_s16), max_s16);
                    round_val = vreinterpretq_u16_s16(add_res);
                }
                vst1q_u16(pu2_dst_tmp_1, round_val);
                pu2_dst_tmp_1 += 8;
            }

            if(nt == 8)
            {
                pu2_ref_tmp_2 -= 8;
                src_tmp_1 = vrev64q_u16(vld1q_u16(pu2_ref_tmp_2));
                rev_res = vextq_u16(src_tmp_1, src_tmp_1, 4);

                vst1q_u16(pu2_dst_tmp_2, vdupq_lane_u16(vget_low_u16(rev_res), 0));
                pu2_dst_tmp_2 += dst_strd;
                vst1q_u16(pu2_dst_tmp_2, vdupq_lane_u16(vget_low_u16(rev_res), 1));
                pu2_dst_tmp_2 += dst_strd;
                vst1q_u16(pu2_dst_tmp_2, vdupq_lane_u16(vget_low_u16(rev_res), 2));
                pu2_dst_tmp_2 += dst_strd;
                vst1q_u16(pu2_dst_tmp_2, vdupq_lane_u16(vget_low_u16(rev_res), 3));
                pu2_dst_tmp_2 += dst_strd;
                vst1q_u16(pu2_dst_tmp_2, vdupq_lane_u16(vget_high_u16(rev_res), 0));
                pu2_dst_tmp_2 += dst_strd;
                vst1q_u16(pu2_dst_tmp_2, vdupq_lane_u16(vget_high_u16(rev_res), 1));
                pu2_dst_tmp_2 += dst_strd;
                vst1q_u16(pu2_dst_tmp_2, vdupq_lane_u16(vget_high_u16(rev_res), 2));
            }
            else
            {
                for(row = 16; row > 0; row -= 8)
                {
                    pu2_ref_tmp_2 -= 8;

                    src_tmp_1 = vrev64q_u16(vld1q_u16(pu2_ref_tmp_2));
                    rev_res = vextq_u16(src_tmp_1, src_tmp_1, 4);

                    dup_1 = vdupq_lane_u16(vget_low_u16(rev_res), 0);
                    dup_2 = vdupq_lane_u16(vget_low_u16(rev_res), 1);
                    dup_3 = vdupq_lane_u16(vget_low_u16(rev_res), 2);
                    dup_4 = vdupq_lane_u16(vget_low_u16(rev_res), 3);
                    dup_5 = vdupq_lane_u16(vget_high_u16(rev_res), 0);
                    dup_6 = vdupq_lane_u16(vget_high_u16(rev_res), 1);
                    dup_7 = vdupq_lane_u16(vget_high_u16(rev_res), 2);
                    dup_8 = vdupq_lane_u16(vget_high_u16(rev_res), 3);

                    vst1q_u16(pu2_dst_tmp_2, dup_1);
                    vst1q_u16(pu2_dst_tmp_2 + 8, dup_1);
                    pu2_dst_tmp_2 += dst_strd;

                    vst1q_u16(pu2_dst_tmp_2, dup_2);
                    vst1q_u16(pu2_dst_tmp_2 + 8, dup_2);
                    pu2_dst_tmp_2 += dst_strd;

                    vst1q_u16(pu2_dst_tmp_2, dup_3);
                    vst1q_u16(pu2_dst_tmp_2 + 8, dup_3);
                    pu2_dst_tmp_2 += dst_strd;

                    vst1q_u16(pu2_dst_tmp_2, dup_4);
                    vst1q_u16(pu2_dst_tmp_2 + 8, dup_4);
                    pu2_dst_tmp_2 += dst_strd;

                    vst1q_u16(pu2_dst_tmp_2, dup_5);
                    vst1q_u16(pu2_dst_tmp_2 + 8, dup_5);
                    pu2_dst_tmp_2 += dst_strd;

                    vst1q_u16(pu2_dst_tmp_2, dup_6);
                    vst1q_u16(pu2_dst_tmp_2 + 8, dup_6);
                    pu2_dst_tmp_2 += dst_strd;

                    vst1q_u16(pu2_dst_tmp_2, dup_7);
                    vst1q_u16(pu2_dst_tmp_2 + 8, dup_7);
                    pu2_dst_tmp_2 += dst_strd;

                    if(row != 8)
                    {
                        vst1q_u16(pu2_dst_tmp_2, dup_8);
                        vst1q_u16(pu2_dst_tmp_2 + 8, dup_8);
                        pu2_dst_tmp_2 += dst_strd;
                    }
                }
            }
        }
    }
}
/* INTRA_PRED_LUMA_HORZ */

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for vertical luma variable.
*
* @par Description:
*    Vertical intraprediction with reference neighboring samples location
*    pointed by 'pu2_ref' to the TU block location pointed by 'pu2_dst'
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
* @param[in] disable_boundary_filter
*  WORD32 boundary filter disable flag
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
void ihevc_hbd_intra_pred_luma_ver_neonintr(UWORD16 *pu2_ref,
                                            WORD32 src_strd,
                                            UWORD16 *pu2_dst,
                                            WORD32 dst_strd,
                                            WORD32 nt,
                                            WORD32 disable_boundary_filter,
                                            UWORD8 bit_depth)
{
    WORD32 row;
    WORD32 two_nt;
    UNUSED(src_strd);

    two_nt = 2 * nt;

    UWORD16 *pu2_dst_tmp = pu2_dst;
    UWORD16 *pu2_ref_tmp_1 = pu2_ref + two_nt + 1;
    if(nt == 32)
    {
        uint16x8_t r0 = vld1q_u16(pu2_ref_tmp_1);
        uint16x8_t r1 = vld1q_u16(pu2_ref_tmp_1 + 8);
        uint16x8_t r2 = vld1q_u16(pu2_ref_tmp_1 + 16);
        uint16x8_t r3 = vld1q_u16(pu2_ref_tmp_1 + 24);

        pu2_dst_tmp = pu2_dst;
        for(row = 0; row < 32; row += 4)
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
    else
    {
        /* naming of variables made according to the operation(instructions) it performs                    */
        /* (eg. shift_val which contains the shifted value,                                                 */
        /* add_sat which has add and saturated value)                                                       */
        /* Loops are unrolled by 4 and 8 considering the fact the input width is either multiple of 4 or 8  */
        /* rows and columns are unrolled by 4, when the width is multiple of 4                              */

        if(0 != (nt & 7))
        {
            UWORD16 *pu2_ref_val1 = pu2_ref + (two_nt - nt);
            UWORD16 *pu2_ref_val3 = pu2_ref + (two_nt + 2);
            UWORD16 *pu2_dst_val1 = pu2_dst;

            uint16x4_t dup_2_sub, round_val, vext_val;
            uint16x4_t dup_2_add;
            uint16x4_t src_val1, src_val2;
            int16x4_t sub_val, shift_val1, add_sat;
            int16x4_t zero_s16 = vdup_n_s16(0);
            int16x4_t max_s16 = vdup_n_s16((1 << bit_depth) - 1);
            uint64x1_t shift_val2;

            dup_2_sub = vdup_n_u16(pu2_ref[two_nt]);
            dup_2_add = vdup_n_u16(pu2_ref[two_nt + 1]);

            /* unrolling s2_predpixel = pu2_ref[two_nt + 1] + ((pu2_ref[two_nt - 1 - row] - pu2_ref[two_nt]) >> 1); here*/
            if (disable_boundary_filter)
            {
                round_val = vdup_n_u16(pu2_ref[two_nt + 1]);
            }
            else
            {
                src_val1 = vld1_u16(pu2_ref_val1);
                sub_val = vsub_s16(vreinterpret_s16_u16(src_val1), vreinterpret_s16_u16(dup_2_sub));
                shift_val1 = vshr_n_s16(sub_val, 1);
                add_sat = vadd_s16(shift_val1, vreinterpret_s16_u16(dup_2_add));
                add_sat = vmin_s16(vmax_s16(add_sat, zero_s16), max_s16);
                round_val = vreinterpret_u16_s16(add_sat);
            }

            /* unrolling pu2_dst[row * dst_strd + col] = pu2_ref[two_nt + 1 + col]; here*/
            src_val2 = vld1_u16(pu2_ref_val3);
            vext_val = vext_u16(round_val, src_val2, 3);
            vst1_u16(pu2_dst_val1, vext_val);
            pu2_dst_val1 += dst_strd;

            shift_val2 = vshl_n_u64(vreinterpret_u64_u16(round_val), 16);
            vext_val = vext_u16(vreinterpret_u16_u64(shift_val2), src_val2, 3);
            vst1_u16(pu2_dst_val1, vext_val);
            pu2_dst_val1 += dst_strd;

            shift_val2 = vshl_n_u64(vreinterpret_u64_u16(round_val), 32);
            vext_val = vext_u16(vreinterpret_u16_u64(shift_val2), src_val2, 3);
            vst1_u16(pu2_dst_val1, vext_val);
            pu2_dst_val1 += dst_strd;

            shift_val2 = vshl_n_u64(vreinterpret_u64_u16(round_val), 48);
            vext_val = vext_u16(vreinterpret_u16_u64(shift_val2), src_val2, 3);
            vst1_u16(pu2_dst_val1, vext_val);
        }

        /* rows and columns are unrolled by 8, when the width is multiple of 8          */
        else
        {
            UWORD16 *pu2_dst_tmp_1 = pu2_dst;
            UWORD16 *pu2_dst_tmp_2 = pu2_dst + 8;

            UWORD16 *pu2_ref_tmp_1 = pu2_ref + two_nt - 8;
            UWORD16 *pu2_ref_tmp_2 = pu2_ref + two_nt + 2;
            UWORD16 *pu2_ref_tmp_3 = pu2_ref + two_nt + 9;

            uint16x8_t pu2_src_tmp1, pu2_src_tmp2;
            uint16x8_t dup_sub, dup_add, round_val, vext_t;
            int16x8_t subsh_val, addsat_val, sub_val;
            int16x8_t zero_s16 = vdupq_n_s16(0);
            int16x8_t max_s16 = vdupq_n_s16((1 << bit_depth) - 1);

            dup_sub = vdupq_n_u16(pu2_ref[two_nt]);
            dup_add = vdupq_n_u16(pu2_ref[two_nt + 1]);
            pu2_src_tmp2 = vld1q_u16(pu2_ref_tmp_2);

            /* 1. Fill Columns 0..7 for all nt rows (runs 1x for nt=8, 2x for nt=16) */
            for (row = nt; row > 0; row -= 8)
            {
                if (disable_boundary_filter)
                {
                    round_val = vdupq_n_u16(pu2_ref[two_nt + 1]);
                }
                else
                {
                    pu2_src_tmp1 = vld1q_u16(pu2_ref_tmp_1);
                    sub_val = vsubq_s16(vreinterpretq_s16_u16(pu2_src_tmp1), vreinterpretq_s16_u16(dup_sub));
                    subsh_val = vshrq_n_s16(sub_val, 1);
                    addsat_val = vaddq_s16(subsh_val, vreinterpretq_s16_u16(dup_add));
                    addsat_val = vminq_s16(vmaxq_s16(addsat_val, zero_s16), max_s16);
                    round_val = vreinterpretq_u16_s16(addsat_val);
                }

                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                round_val = vextq_u16(round_val, round_val, 7);
                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                round_val = vextq_u16(round_val, round_val, 7);
                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                round_val = vextq_u16(round_val, round_val, 7);
                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                round_val = vextq_u16(round_val, round_val, 7);
                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                round_val = vextq_u16(round_val, round_val, 7);
                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                round_val = vextq_u16(round_val, round_val, 7);
                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                round_val = vextq_u16(round_val, round_val, 7);
                vext_t = vextq_u16(round_val, pu2_src_tmp2, 7);
                vst1q_u16(pu2_dst_tmp_1, vext_t);
                pu2_dst_tmp_1 += dst_strd;

                pu2_ref_tmp_1 -= 8;
            }

            /* 2. Fill Columns 8..15 for all 16 rows (only runs when nt == 16) */
            if (nt == 16)
            {
                pu2_src_tmp2 = vld1q_u16(pu2_ref_tmp_3);
                for (row = 16; row > 0; row -= 8)
                {
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                    vst1q_u16(pu2_dst_tmp_2, pu2_src_tmp2);
                    pu2_dst_tmp_2 += dst_strd;
                }
            }
        }
    }
}
/* INTRA_PRED_LUMA_VER */

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma mode2.
*
* @par Description:
*    Intraprediction for mode 2 (sw angle) with reference neighboring samples
*    location pointed by 'pu2_ref' to the TU block location pointed by
*    'pu2_dst'
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
*  integer intra prediction mode
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
void ihevc_hbd_intra_pred_luma_mode2_neonintr(UWORD16 *pu2_ref,
                                              WORD32 src_strd,
                                              UWORD16 *pu2_dst,
                                              WORD32 dst_strd,
                                              WORD32 nt,
                                              WORD32 mode,
                                              UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 two_nt;
    UNUSED(src_strd);
    UNUSED(mode);
    UNUSED(bit_depth);

    /* rev_res naming has been made to have the reverse result value in it                              */
    /* Loops are unrolled by 4 and 8 considering the fact the input width is either multiple of 4 or 8  */
    /* rows and columns are unrolled by 4, when the width is multiple of 4                              */

    if(0 != (nt & 7))
    {
        UWORD16 *pu2_ref_tmp = pu2_ref;
        UWORD16 *pu2_dst_tmp = pu2_dst;
        uint16x8_t pu2_src_val, rev_res;
        uint16x4_t r_hi, r_lo;

        /* pu2_dst[row + (col * dst_strd)] = pu2_ref[two_nt - col - idx - 1]; */
        pu2_src_val = vld1q_u16(pu2_ref_tmp);
        rev_res = vrev64q_u16(pu2_src_val);
        r_hi = vget_high_u16(rev_res);
        r_lo = vget_low_u16(rev_res);

        vst1_u16(pu2_dst_tmp, vext_u16(r_hi, r_lo, 1));
        pu2_dst_tmp += dst_strd;

        vst1_u16(pu2_dst_tmp, vext_u16(r_hi, r_lo, 2));
        pu2_dst_tmp += dst_strd;

        vst1_u16(pu2_dst_tmp, vext_u16(r_hi, r_lo, 3));
        pu2_dst_tmp += dst_strd;

        vst1_u16(pu2_dst_tmp, r_lo);
        pu2_dst_tmp += dst_strd;
    }

    /* rev_val_second, rev_val_first  to reverse the loaded values in order to get the values in right order */
    /* rows and columns are unrolled by 8, when the width is multiple of 8                              */

    else
    {
        UWORD16 *pu2_ref_two_nt_minus2 = pu2_ref;
        UWORD16 *pu2_dst_tmp = pu2_dst;
        UWORD16 *pu2_dst_tmp_plus8 = pu2_dst;

        uint16x8_t pu2_src_val1, pu2_src_val2, vext_t, rev_val_second, rev_val_first;

        two_nt = 2 * nt;
        pu2_ref_two_nt_minus2 += (two_nt);
        pu2_ref_two_nt_minus2 -= 8;

        for(col = nt; col > 0; col -= 8)
        {
            for(row = nt; row > 0; row -= 8)
            {
                pu2_src_val2 = vrev64q_u16(vld1q_u16(pu2_ref_two_nt_minus2));
                rev_val_first = vextq_u16(pu2_src_val2, pu2_src_val2, 4);

                pu2_ref_two_nt_minus2 -= 8;
                pu2_src_val1 = vrev64q_u16(vld1q_u16(pu2_ref_two_nt_minus2));
                rev_val_second = vextq_u16(pu2_src_val1, pu2_src_val1, 4);

                vext_t = vextq_u16(rev_val_first, rev_val_second, 1);
                vst1q_u16(pu2_dst_tmp, vext_t);
                pu2_dst_tmp += dst_strd;

                vext_t = vextq_u16(rev_val_first, rev_val_second, 2);
                vst1q_u16(pu2_dst_tmp, vext_t);
                pu2_dst_tmp += dst_strd;

                vext_t = vextq_u16(rev_val_first, rev_val_second, 3);
                vst1q_u16(pu2_dst_tmp, vext_t);
                pu2_dst_tmp += dst_strd;

                vext_t = vextq_u16(rev_val_first, rev_val_second, 4);
                vst1q_u16(pu2_dst_tmp, vext_t);
                pu2_dst_tmp += dst_strd;

                vext_t = vextq_u16(rev_val_first, rev_val_second, 5);
                vst1q_u16(pu2_dst_tmp, vext_t);
                pu2_dst_tmp += dst_strd;

                vext_t = vextq_u16(rev_val_first, rev_val_second, 6);
                vst1q_u16(pu2_dst_tmp, vext_t);
                pu2_dst_tmp += dst_strd;

                vext_t = vextq_u16(rev_val_first, rev_val_second, 7);
                vst1q_u16(pu2_dst_tmp, vext_t);
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, rev_val_second);
                pu2_dst_tmp += dst_strd;
            }
            pu2_dst_tmp_plus8 += 8;
            pu2_dst_tmp = pu2_dst_tmp_plus8;
            pu2_ref_two_nt_minus2 += (nt - 8);
        }
    }
}
/* INTRA_PRED_LUMA_MODE2 */

/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for luma mode 18 & mode 34.
*
* @par Description:
*    Intraprediction for mode 34 (ne angle) with reference neighboring
*    samples location pointed by 'pu2_ref' to the TU block location pointed by
*    'pu2_dst'
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
*  integer intra prediction mode
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
void ihevc_hbd_intra_pred_luma_mode_18_34_neonintr(UWORD16 *pu2_ref,
                                                   WORD32 src_strd,
                                                   UWORD16 *pu2_dst,
                                                   WORD32 dst_strd,
                                                   WORD32 nt,
                                                   WORD32 mode,
                                                   UWORD8 bit_depth)
{
    WORD32 row;
    WORD32 two_nt;
    UNUSED(src_strd);
    UNUSED(bit_depth);
    two_nt = 2 * nt;

    UWORD16 *pu2_ref_tmp = pu2_ref;
    UWORD16 *pu2_dst_tmp = pu2_dst;

    /* cond to allow multiples of 8 */
    if(0 == (nt & 7))
    {
        if(mode == 34)
        {
            pu2_ref_tmp += (two_nt + 2);
            if(nt == 8)
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp + 8);

                vst1q_u16(pu2_dst_tmp, s0);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 1));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 3));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 5));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 7));
            }
            else if(nt == 16)
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 24);

                vst1q_u16(pu2_dst_tmp, s0);
                vst1q_u16(pu2_dst_tmp + 8, s1);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 1));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 1));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 3));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 3));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 5));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 5));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 7));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 7));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, s1);
                vst1q_u16(pu2_dst_tmp + 8, s2);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 1));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 1));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 3));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 3));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 5));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 5));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 7));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 7));
            }
            else
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 24);
                pu2_ref_tmp += 32;

                for(row = 32; row > 0; row -= 8)
                {
                    uint16x8_t s4 = vld1q_u16(pu2_ref_tmp);
                    pu2_ref_tmp += 8;

                    vst1q_u16(pu2_dst_tmp, s0);
                    vst1q_u16(pu2_dst_tmp + 8, s1);
                    vst1q_u16(pu2_dst_tmp + 16, s2);
                    vst1q_u16(pu2_dst_tmp + 24, s3);
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 1));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 1));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 1));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 1));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 3));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 3));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 3));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 3));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 5));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 5));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 5));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 5));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 7));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 7));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 7));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 7));
                    pu2_dst_tmp += dst_strd;

                    s0 = s1;
                    s1 = s2;
                    s2 = s3;
                    s3 = s4;
                }
            }
        }
        else /* Loop for mode 18 */
        {
            pu2_ref_tmp += two_nt;
            if(nt == 8)
            {
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp - 8);

                vst1q_u16(pu2_dst_tmp, s1);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 7));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 5));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 3));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 1));
            }
            else if(nt == 16)
            {
                uint16x8_t s0 = vld1q_u16(pu2_ref_tmp - 16);
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp - 8);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 8);

                vst1q_u16(pu2_dst_tmp, s2);
                vst1q_u16(pu2_dst_tmp + 8, s3);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 7));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 7));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 5));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 5));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 3));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 3));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s1, s2, 1));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s2, s3, 1));
                pu2_dst_tmp += dst_strd;

                vst1q_u16(pu2_dst_tmp, s1);
                vst1q_u16(pu2_dst_tmp + 8, s2);
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 7));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 7));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 5));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 5));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 3));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 3));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                pu2_dst_tmp += dst_strd;
                vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 1));
                vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 1));
            }
            else
            {
                uint16x8_t s1 = vld1q_u16(pu2_ref_tmp);
                uint16x8_t s2 = vld1q_u16(pu2_ref_tmp + 8);
                uint16x8_t s3 = vld1q_u16(pu2_ref_tmp + 16);
                uint16x8_t s4 = vld1q_u16(pu2_ref_tmp + 24);
                pu2_ref_tmp -= 8;

                for(row = 32; row > 0; row -= 8)
                {
                    uint16x8_t s0 = vld1q_u16(pu2_ref_tmp);
                    pu2_ref_tmp -= 8;

                    vst1q_u16(pu2_dst_tmp, s1);
                    vst1q_u16(pu2_dst_tmp + 8, s2);
                    vst1q_u16(pu2_dst_tmp + 16, s3);
                    vst1q_u16(pu2_dst_tmp + 24, s4);
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 7));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 7));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 7));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 7));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 6));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 6));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 6));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 6));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 5));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 5));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 5));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 5));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 4));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 4));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 4));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 4));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 3));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 3));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 3));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 3));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 2));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 2));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 2));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 2));
                    pu2_dst_tmp += dst_strd;

                    vst1q_u16(pu2_dst_tmp, vextq_u16(s0, s1, 1));
                    vst1q_u16(pu2_dst_tmp + 8, vextq_u16(s1, s2, 1));
                    vst1q_u16(pu2_dst_tmp + 16, vextq_u16(s2, s3, 1));
                    vst1q_u16(pu2_dst_tmp + 24, vextq_u16(s3, s4, 1));
                    pu2_dst_tmp += dst_strd;

                    s4 = s3;
                    s3 = s2;
                    s2 = s1;
                    s1 = s0;
                }
            }
        }
    }
    else /* loop for multiples of 4 */
    {
        if(mode == 34)
        {
            uint16x8_t s = vld1q_u16(pu2_ref + two_nt + 2);
            vst1_u16(pu2_dst_tmp, vget_low_u16(s));
            pu2_dst_tmp += dst_strd;
            vst1_u16(pu2_dst_tmp, vget_low_u16(vextq_u16(s, s, 1)));
            pu2_dst_tmp += dst_strd;
            vst1_u16(pu2_dst_tmp, vget_low_u16(vextq_u16(s, s, 2)));
            pu2_dst_tmp += dst_strd;
            vst1_u16(pu2_dst_tmp, vget_low_u16(vextq_u16(s, s, 3)));
        }
        else
        {
            uint16x8_t s = vld1q_u16(pu2_ref + two_nt - 3);
            vst1_u16(pu2_dst_tmp, vget_low_u16(vextq_u16(s, s, 3)));
            pu2_dst_tmp += dst_strd;
            vst1_u16(pu2_dst_tmp, vget_low_u16(vextq_u16(s, s, 2)));
            pu2_dst_tmp += dst_strd;
            vst1_u16(pu2_dst_tmp, vget_low_u16(vextq_u16(s, s, 1)));
            pu2_dst_tmp += dst_strd;
            vst1_u16(pu2_dst_tmp, vget_low_u16(s));
        }
    }
}
/* INTRA_PRED_LUMA_MODE_18_34 */

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma mode 3 to mode 9
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
void ihevc_hbd_intra_pred_luma_mode_3_to_9_neonintr(UWORD16 *pu2_ref,
                                                    WORD32 src_strd,
                                                    UWORD16 *pu2_dst,
                                                    WORD32 dst_strd,
                                                    WORD32 nt,
                                                    WORD32 mode,
                                                    UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 intra_pred_ang;
    WORD32 pos, fract;
    UNUSED(src_strd);
    UNUSED(bit_depth);

    /* Intra Pred Angle according to the mode */
    intra_pred_ang = gai4_ihevc_ang_table[mode];

    if(0 == (nt & 7))
    {
        WORD32 two_nt = 2 * nt;

        for(col = 0; col < nt; col += 8)
        {
            WORD32 idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
            uint16x8_t f0, f1, f2, f3, f4, f5, f6, f7;
            uint16x8_t inv_f0, inv_f1, inv_f2, inv_f3, inv_f4, inv_f5, inv_f6, inv_f7;
            UWORD16 *pu2_ref_row = pu2_ref + two_nt - 8;
            UWORD16 *pu2_dst_col = pu2_dst + col;

            pos = (col + 1) * intra_pred_ang;
            idx0 = -(pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx1 = -(pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx2 = -(pos >> 5);
            fract = pos & 31;
            f2 = vdupq_n_u16((uint16_t)fract);
            inv_f2 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx3 = -(pos >> 5);
            fract = pos & 31;
            f3 = vdupq_n_u16((uint16_t)fract);
            inv_f3 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx4 = -(pos >> 5);
            fract = pos & 31;
            f4 = vdupq_n_u16((uint16_t)fract);
            inv_f4 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx5 = -(pos >> 5);
            fract = pos & 31;
            f5 = vdupq_n_u16((uint16_t)fract);
            inv_f5 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx6 = -(pos >> 5);
            fract = pos & 31;
            f6 = vdupq_n_u16((uint16_t)fract);
            inv_f6 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx7 = -(pos >> 5);
            fract = pos & 31;
            f7 = vdupq_n_u16((uint16_t)fract);
            inv_f7 = vdupq_n_u16((uint16_t)(32 - fract));

            for(row = nt; row > 0; row -= 8)
            {
                UWORD16 *p;
                uint16x8_t c0, c1, c2, c3, c4, c5, c6, c7;
                uint16x8x2_t t01, t23, t45, t67;
                uint32x4x2_t s0, s1, s2, s3;

                p = pu2_ref_row + idx0;
                c0 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f0), vld1q_u16(p - 1), f0), 5);
                p = pu2_ref_row + idx1;
                c1 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f1), vld1q_u16(p - 1), f1), 5);
                p = pu2_ref_row + idx2;
                c2 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f2), vld1q_u16(p - 1), f2), 5);
                p = pu2_ref_row + idx3;
                c3 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f3), vld1q_u16(p - 1), f3), 5);
                p = pu2_ref_row + idx4;
                c4 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f4), vld1q_u16(p - 1), f4), 5);
                p = pu2_ref_row + idx5;
                c5 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f5), vld1q_u16(p - 1), f5), 5);
                p = pu2_ref_row + idx6;
                c6 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f6), vld1q_u16(p - 1), f6), 5);
                p = pu2_ref_row + idx7;
                c7 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f7), vld1q_u16(p - 1), f7), 5);

                t01 = vtrnq_u16(c0, c1);
                t23 = vtrnq_u16(c2, c3);
                t45 = vtrnq_u16(c4, c5);
                t67 = vtrnq_u16(c6, c7);

                s0 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]), vreinterpretq_u32_u16(t23.val[0]));
                s1 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]), vreinterpretq_u32_u16(t23.val[1]));
                s2 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[0]), vreinterpretq_u32_u16(t67.val[0]));
                s3 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[1]), vreinterpretq_u32_u16(t67.val[1]));

                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s1.val[1]), vget_high_u32(s3.val[1]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s0.val[1]), vget_high_u32(s2.val[1]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s1.val[0]), vget_high_u32(s3.val[0]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s0.val[0]), vget_high_u32(s2.val[0]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s1.val[1]), vget_low_u32(s3.val[1]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s0.val[1]), vget_low_u32(s2.val[1]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s1.val[0]), vget_low_u32(s3.val[0]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s0.val[0]), vget_low_u32(s2.val[0]))));
                pu2_dst_col += dst_strd;

                pu2_ref_row -= 8;
            }
        }
    }
    else
    {
        UWORD16 *pu2_ref_base = pu2_ref + nt;
        UWORD16 *p;
        uint16x4_t c0, c1, c2, c3;
        uint16x4x2_t t01, t23;
        uint32x2x2_t s0, s1;

        pos = intra_pred_ang;
        p = pu2_ref_base - (pos >> 5);
        fract = pos & 31;
        c0 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p - 1), vdup_n_u16((uint16_t)fract)), 5);

        pos += intra_pred_ang;
        p = pu2_ref_base - (pos >> 5);
        fract = pos & 31;
        c1 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p - 1), vdup_n_u16((uint16_t)fract)), 5);

        pos += intra_pred_ang;
        p = pu2_ref_base - (pos >> 5);
        fract = pos & 31;
        c2 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p - 1), vdup_n_u16((uint16_t)fract)), 5);

        pos += intra_pred_ang;
        p = pu2_ref_base - (pos >> 5);
        fract = pos & 31;
        c3 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p - 1), vdup_n_u16((uint16_t)fract)), 5);

        t01 = vtrn_u16(c0, c1);
        t23 = vtrn_u16(c2, c3);
        s0 = vtrn_u32(vreinterpret_u32_u16(t01.val[0]), vreinterpret_u32_u16(t23.val[0]));
        s1 = vtrn_u32(vreinterpret_u32_u16(t01.val[1]), vreinterpret_u32_u16(t23.val[1]));

        vst1_u16(pu2_dst, vreinterpret_u16_u32(s1.val[1]));
        pu2_dst += dst_strd;
        vst1_u16(pu2_dst, vreinterpret_u16_u32(s0.val[1]));
        pu2_dst += dst_strd;
        vst1_u16(pu2_dst, vreinterpret_u16_u32(s1.val[0]));
        pu2_dst += dst_strd;
        vst1_u16(pu2_dst, vreinterpret_u16_u32(s0.val[0]));
    }
}

/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for luma mode 11 to mode 17
*
* @par Description:
*    Intraprediction for mode 11 to 17 (negative angle, horizontal mode)
*    with reference neighboring samples location pointed by 'pu2_ref' to the
*    TU block location pointed by 'pu2_dst'
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
void ihevc_hbd_intra_pred_luma_mode_11_to_17_neonintr(UWORD16 *pu2_ref,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_dst,
                                                      WORD32 dst_strd,
                                                      WORD32 nt,
                                                      WORD32 mode,
                                                      UWORD8 bit_depth)
{
    WORD32 row, col, k;
    WORD32 two_nt;
    WORD32 intra_pred_ang, inv_ang, inv_ang_sum;
    WORD32 pos, fract;
    WORD32 ref_idx;

    UWORD16 *ref_main;
    UWORD16 *ref_main_tmp;

    UWORD16 *pu2_ref_tmp2 = pu2_ref;

    UWORD16 ref_temp[2 * MAX_CU_SIZE + 1];

    uint16x8_t ref_left_t;
    uint16x4_t ref_left_tmp;
    UNUSED(src_strd);
    UNUSED(bit_depth);

    inv_ang_sum = 128;
    two_nt = 2 * nt;

    intra_pred_ang = gai4_ihevc_ang_table[mode];

    inv_ang = gai4_ihevc_inv_ang_table[mode - 11];

    ref_main = ref_temp + (nt - 1);
    ref_main_tmp = ref_main;

    if(0 == (nt & 7))
    {
        pu2_ref_tmp2 += (two_nt - 7);

        for(k = nt - 1; k >= 0; k -= 8)
        {
            ref_left_t = vrev64q_u16(vld1q_u16(pu2_ref_tmp2));

            ref_left_t = vextq_u16(ref_left_t, ref_left_t, 4);
            vst1q_u16(ref_main_tmp, ref_left_t);
            ref_main_tmp += 8;
            pu2_ref_tmp2 -= 8;
        }
    }
    else
    {
        uint16x4_t rev_val;
        pu2_ref_tmp2 += (two_nt - (nt - 1));

        ref_left_tmp = vld1_u16(pu2_ref_tmp2);

        rev_val = vrev64_u16(ref_left_tmp);
        vst1_u16(ref_main_tmp, rev_val);
    }

    ref_main[nt] = pu2_ref[two_nt - nt];

    /* For horizontal modes, (ref main = ref left) (ref side = ref above) */

    ref_idx = (nt * intra_pred_ang) >> 5;

    /* SIMD Optimization can be done using look-up table for the loop */
    /* For negative angled derive the main reference samples from side */
    /*  reference samples refer to section 8.4.4.2.6 */
    for(k = -1; k > ref_idx; k--)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[two_nt + (inv_ang_sum >> 8)];
    }

    if(0 == (nt & 7))
    {
        /* For the angles other then 45 degree, interpolation btw 2 neighboring */
        /* samples dependent on distance to obtain destination sample */
        for(col = 0; col < nt; col += 8)
        {
            WORD32 idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
            uint16x8_t f0, f1, f2, f3, f4, f5, f6, f7;
            uint16x8_t inv_f0, inv_f1, inv_f2, inv_f3, inv_f4, inv_f5, inv_f6, inv_f7;
            UWORD16 *ref_row = ref_main;
            UWORD16 *pu2_dst_col = pu2_dst + col;

            pos = (col + 1) * intra_pred_ang;
            idx0 = (pos >> 5) + 1;
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx1 = (pos >> 5) + 1;
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx2 = (pos >> 5) + 1;
            fract = pos & 31;
            f2 = vdupq_n_u16((uint16_t)fract);
            inv_f2 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx3 = (pos >> 5) + 1;
            fract = pos & 31;
            f3 = vdupq_n_u16((uint16_t)fract);
            inv_f3 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx4 = (pos >> 5) + 1;
            fract = pos & 31;
            f4 = vdupq_n_u16((uint16_t)fract);
            inv_f4 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx5 = (pos >> 5) + 1;
            fract = pos & 31;
            f5 = vdupq_n_u16((uint16_t)fract);
            inv_f5 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx6 = (pos >> 5) + 1;
            fract = pos & 31;
            f6 = vdupq_n_u16((uint16_t)fract);
            inv_f6 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            idx7 = (pos >> 5) + 1;
            fract = pos & 31;
            f7 = vdupq_n_u16((uint16_t)fract);
            inv_f7 = vdupq_n_u16((uint16_t)(32 - fract));

            for(row = nt; row > 0; row -= 8)
            {
                UWORD16 *p;
                uint16x8_t c0, c1, c2, c3, c4, c5, c6, c7;
                uint16x8x2_t t01, t23, t45, t67;
                uint32x4x2_t s0, s1, s2, s3;

                p = ref_row + idx0;
                c0 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f0), vld1q_u16(p + 1), f0), 5);
                p = ref_row + idx1;
                c1 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f1), vld1q_u16(p + 1), f1), 5);
                p = ref_row + idx2;
                c2 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f2), vld1q_u16(p + 1), f2), 5);
                p = ref_row + idx3;
                c3 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f3), vld1q_u16(p + 1), f3), 5);
                p = ref_row + idx4;
                c4 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f4), vld1q_u16(p + 1), f4), 5);
                p = ref_row + idx5;
                c5 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f5), vld1q_u16(p + 1), f5), 5);
                p = ref_row + idx6;
                c6 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f6), vld1q_u16(p + 1), f6), 5);
                p = ref_row + idx7;
                c7 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(vld1q_u16(p), inv_f7), vld1q_u16(p + 1), f7), 5);

                t01 = vtrnq_u16(c0, c1);
                t23 = vtrnq_u16(c2, c3);
                t45 = vtrnq_u16(c4, c5);
                t67 = vtrnq_u16(c6, c7);

                s0 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]), vreinterpretq_u32_u16(t23.val[0]));
                s1 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]), vreinterpretq_u32_u16(t23.val[1]));
                s2 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[0]), vreinterpretq_u32_u16(t67.val[0]));
                s3 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[1]), vreinterpretq_u32_u16(t67.val[1]));

                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s0.val[0]), vget_low_u32(s2.val[0]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s1.val[0]), vget_low_u32(s3.val[0]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s0.val[1]), vget_low_u32(s2.val[1]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(s1.val[1]), vget_low_u32(s3.val[1]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s0.val[0]), vget_high_u32(s2.val[0]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s1.val[0]), vget_high_u32(s3.val[0]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s0.val[1]), vget_high_u32(s2.val[1]))));
                pu2_dst_col += dst_strd;
                vst1q_u16(pu2_dst_col, vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(s1.val[1]), vget_high_u32(s3.val[1]))));
                pu2_dst_col += dst_strd;

                ref_row += 8;
            }
        }
    }
    else
    {
        UWORD16 *p;
        uint16x4_t c0, c1, c2, c3;
        uint16x4x2_t t01, t23;
        uint32x2x2_t s0, s1;

        pos = intra_pred_ang;
        p = ref_main + (pos >> 5) + 1;
        fract = pos & 31;
        c0 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p + 1), vdup_n_u16((uint16_t)fract)), 5);

        pos += intra_pred_ang;
        p = ref_main + (pos >> 5) + 1;
        fract = pos & 31;
        c1 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p + 1), vdup_n_u16((uint16_t)fract)), 5);

        pos += intra_pred_ang;
        p = ref_main + (pos >> 5) + 1;
        fract = pos & 31;
        c2 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p + 1), vdup_n_u16((uint16_t)fract)), 5);

        pos += intra_pred_ang;
        p = ref_main + (pos >> 5) + 1;
        fract = pos & 31;
        c3 = vrshr_n_u16(vmla_u16(vmul_u16(vld1_u16(p), vdup_n_u16((uint16_t)(32 - fract))), vld1_u16(p + 1), vdup_n_u16((uint16_t)fract)), 5);

        t01 = vtrn_u16(c0, c1);
        t23 = vtrn_u16(c2, c3);
        s0 = vtrn_u32(vreinterpret_u32_u16(t01.val[0]), vreinterpret_u32_u16(t23.val[0]));
        s1 = vtrn_u32(vreinterpret_u32_u16(t01.val[1]), vreinterpret_u32_u16(t23.val[1]));

        vst1_u16(pu2_dst, vreinterpret_u16_u32(s0.val[0]));
        pu2_dst += dst_strd;
        vst1_u16(pu2_dst, vreinterpret_u16_u32(s1.val[0]));
        pu2_dst += dst_strd;
        vst1_u16(pu2_dst, vreinterpret_u16_u32(s0.val[1]));
        pu2_dst += dst_strd;
        vst1_u16(pu2_dst, vreinterpret_u16_u32(s1.val[1]));
    }
}

/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for luma mode 19 to mode 25
*
* @par Description:
*    Intraprediction for mode 19 to 25 (negative angle, vertical mode) with
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
void ihevc_hbd_intra_pred_luma_mode_19_to_25_neonintr(UWORD16 *pu2_ref,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_dst,
                                                      WORD32 dst_strd,
                                                      WORD32 nt,
                                                      WORD32 mode,
                                                      UWORD8 bit_depth)
{
    WORD32 row, k;
    WORD32 two_nt, intra_pred_ang;
    WORD32 inv_ang, inv_ang_sum, pos, fract;
    WORD32 ref_idx;
    UWORD16 *ref_main;
    UWORD16 ref_temp[(2 * MAX_CU_SIZE) + 2];

    UWORD16 *pu2_ref_tmp1 = pu2_ref;
    UWORD16 *pu2_dst_tmp1 = pu2_dst;

    UNUSED(src_strd);
    UNUSED(bit_depth);

    two_nt = 2 * nt;
    intra_pred_ang = gai4_ihevc_ang_table[mode];
    inv_ang = gai4_ihevc_inv_ang_table[mode - 12];

    pu2_ref_tmp1 += two_nt;
    ref_main = ref_temp + nt;

    ref_idx = (nt * intra_pred_ang) >> 5;
    inv_ang_sum = 128;

    for(k = -1; k > ref_idx; k--)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[two_nt - (inv_ang_sum >> 8)];
    }

    UWORD16 *ref_base = ref_main + 1;

    if(nt == 8)
    {
        vst1q_u16(ref_main, vld1q_u16(pu2_ref_tmp1));
        ref_main[8] = pu2_ref_tmp1[8];

        pos = 0;
        for(row = 0; row < 8; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a0, b0, a1, b1, r0, r1;

            pos += intra_pred_ang;
            p0 = ref_base + (pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = ref_base + (pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a0 = vld1q_u16(p0);
            b0 = vld1q_u16(p0 + 1);
            a1 = vld1q_u16(p1);
            b1 = vld1q_u16(p1 + 1);

            r0 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f0), b0, f0), 5);
            r1 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f1), b1, f1), 5);

            vst1q_u16(pu2_dst_tmp1, r0);
            pu2_dst_tmp1 += dst_strd;
            vst1q_u16(pu2_dst_tmp1, r1);
            pu2_dst_tmp1 += dst_strd;
        }
    }
    else if(nt == 16)
    {
        uint16x8_t c0 = vld1q_u16(pu2_ref_tmp1);
        uint16x8_t c1 = vld1q_u16(pu2_ref_tmp1 + 8);
        vst1q_u16(ref_main, c0);
        vst1q_u16(ref_main + 8, c1);
        ref_main[16] = pu2_ref_tmp1[16];

        pos = 0;
        for(row = 0; row < 16; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a00, a01, b00, b01, a10, a11, b10, b11;
            uint16x8_t r00, r01, r10, r11;

            pos += intra_pred_ang;
            p0 = ref_base + (pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = ref_base + (pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            b00 = vextq_u16(a00, a01, 1);
            b01 = vld1q_u16(p0 + 9);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            b10 = vextq_u16(a10, a11, 1);
            b11 = vld1q_u16(p1 + 9);

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
    else if(nt == 32)
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

        pos = 0;
        for(row = 0; row < 32; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a00, a01, a02, a03, b00, b01, b02, b03;
            uint16x8_t a10, a11, a12, a13, b10, b11, b12, b13;
            uint16x8_t r00, r01, r02, r03, r10, r11, r12, r13;

            pos += intra_pred_ang;
            p0 = ref_base + (pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = ref_base + (pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            a02 = vld1q_u16(p0 + 16);
            a03 = vld1q_u16(p0 + 24);
            b00 = vextq_u16(a00, a01, 1);
            b01 = vextq_u16(a01, a02, 1);
            b02 = vextq_u16(a02, a03, 1);
            b03 = vld1q_u16(p0 + 25);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            a12 = vld1q_u16(p1 + 16);
            a13 = vld1q_u16(p1 + 24);
            b10 = vextq_u16(a10, a11, 1);
            b11 = vextq_u16(a11, a12, 1);
            b12 = vextq_u16(a12, a13, 1);
            b13 = vld1q_u16(p1 + 25);

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
    else
    {
        UWORD16 *p0, *p1, *p2, *p3;
        uint16x4_t f0, inv_f0, f1, inv_f1, f2, inv_f2, f3, inv_f3;
        uint16x4_t a0, b0, a1, b1, a2, b2, a3, b3;

        vst1_u16(ref_main, vld1_u16(pu2_ref_tmp1));
        ref_main[4] = pu2_ref_tmp1[4];

        pos = intra_pred_ang;
        p0 = ref_base + (pos >> 5);
        fract = pos & 31;
        f0 = vdup_n_u16((uint16_t)fract);
        inv_f0 = vdup_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p1 = ref_base + (pos >> 5);
        fract = pos & 31;
        f1 = vdup_n_u16((uint16_t)fract);
        inv_f1 = vdup_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p2 = ref_base + (pos >> 5);
        fract = pos & 31;
        f2 = vdup_n_u16((uint16_t)fract);
        inv_f2 = vdup_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p3 = ref_base + (pos >> 5);
        fract = pos & 31;
        f3 = vdup_n_u16((uint16_t)fract);
        inv_f3 = vdup_n_u16((uint16_t)(32 - fract));

        a0 = vld1_u16(p0);
        b0 = vld1_u16(p0 + 1);
        a1 = vld1_u16(p1);
        b1 = vld1_u16(p1 + 1);
        a2 = vld1_u16(p2);
        b2 = vld1_u16(p2 + 1);
        a3 = vld1_u16(p3);
        b3 = vld1_u16(p3 + 1);

        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a0, inv_f0), b0, f0), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a1, inv_f1), b1, f1), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a2, inv_f2), b2, f2), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a3, inv_f3), b3, f3), 5));
    }
}

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma mode 27 to mode 33
*
* @par Description:
*    Intraprediction for mode 27 to 33 (positive angle, vertical mode) with
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
void ihevc_hbd_intra_pred_luma_mode_27_to_33_neonintr(UWORD16 *pu2_ref,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_dst,
                                                      WORD32 dst_strd,
                                                      WORD32 nt,
                                                      WORD32 mode,
                                                      UWORD8 bit_depth)
{
    WORD32 row;
    WORD32 intra_pred_ang;
    WORD32 pos, fract;
    WORD32 two_nt = 2 * nt;
    UWORD16 *pu2_ref_base = pu2_ref + two_nt + 1;
    UWORD16 *pu2_dst_tmp1 = pu2_dst;

    UNUSED(src_strd);
    UNUSED(bit_depth);

    /* Intra Pred Angle according to the mode */
    intra_pred_ang = gai4_ihevc_ang_table[mode];

    if(nt == 8)
    {
        pos = 0;
        for(row = 0; row < 8; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a0, b0, a1, b1, r0, r1;

            pos += intra_pred_ang;
            p0 = pu2_ref_base + (pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = pu2_ref_base + (pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a0 = vld1q_u16(p0);
            b0 = vld1q_u16(p0 + 1);
            a1 = vld1q_u16(p1);
            b1 = vld1q_u16(p1 + 1);

            r0 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a0, inv_f0), b0, f0), 5);
            r1 = vrshrq_n_u16(vmlaq_u16(vmulq_u16(a1, inv_f1), b1, f1), 5);

            vst1q_u16(pu2_dst_tmp1, r0);
            pu2_dst_tmp1 += dst_strd;
            vst1q_u16(pu2_dst_tmp1, r1);
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
            uint16x8_t a00, a01, b00, b01, a10, a11, b10, b11;
            uint16x8_t r00, r01, r10, r11;

            pos += intra_pred_ang;
            p0 = pu2_ref_base + (pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = pu2_ref_base + (pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            b00 = vextq_u16(a00, a01, 1);
            b01 = vld1q_u16(p0 + 9);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            b10 = vextq_u16(a10, a11, 1);
            b11 = vld1q_u16(p1 + 9);

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
    else if(nt == 32)
    {
        pos = 0;
        for(row = 0; row < 32; row += 2)
        {
            UWORD16 *p0, *p1;
            uint16x8_t f0, inv_f0, f1, inv_f1;
            uint16x8_t a00, a01, a02, a03, b00, b01, b02, b03;
            uint16x8_t a10, a11, a12, a13, b10, b11, b12, b13;
            uint16x8_t r00, r01, r02, r03, r10, r11, r12, r13;

            pos += intra_pred_ang;
            p0 = pu2_ref_base + (pos >> 5);
            fract = pos & 31;
            f0 = vdupq_n_u16((uint16_t)fract);
            inv_f0 = vdupq_n_u16((uint16_t)(32 - fract));

            pos += intra_pred_ang;
            p1 = pu2_ref_base + (pos >> 5);
            fract = pos & 31;
            f1 = vdupq_n_u16((uint16_t)fract);
            inv_f1 = vdupq_n_u16((uint16_t)(32 - fract));

            a00 = vld1q_u16(p0);
            a01 = vld1q_u16(p0 + 8);
            a02 = vld1q_u16(p0 + 16);
            a03 = vld1q_u16(p0 + 24);
            b00 = vextq_u16(a00, a01, 1);
            b01 = vextq_u16(a01, a02, 1);
            b02 = vextq_u16(a02, a03, 1);
            b03 = vld1q_u16(p0 + 25);

            a10 = vld1q_u16(p1);
            a11 = vld1q_u16(p1 + 8);
            a12 = vld1q_u16(p1 + 16);
            a13 = vld1q_u16(p1 + 24);
            b10 = vextq_u16(a10, a11, 1);
            b11 = vextq_u16(a11, a12, 1);
            b12 = vextq_u16(a12, a13, 1);
            b13 = vld1q_u16(p1 + 25);

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
    else
    {
        UWORD16 *p0, *p1, *p2, *p3;
        uint16x4_t f0, inv_f0, f1, inv_f1, f2, inv_f2, f3, inv_f3;
        uint16x4_t a0, b0, a1, b1, a2, b2, a3, b3;

        pos = intra_pred_ang;
        p0 = pu2_ref_base + (pos >> 5);
        fract = pos & 31;
        f0 = vdup_n_u16((uint16_t)fract);
        inv_f0 = vdup_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p1 = pu2_ref_base + (pos >> 5);
        fract = pos & 31;
        f1 = vdup_n_u16((uint16_t)fract);
        inv_f1 = vdup_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p2 = pu2_ref_base + (pos >> 5);
        fract = pos & 31;
        f2 = vdup_n_u16((uint16_t)fract);
        inv_f2 = vdup_n_u16((uint16_t)(32 - fract));

        pos += intra_pred_ang;
        p3 = pu2_ref_base + (pos >> 5);
        fract = pos & 31;
        f3 = vdup_n_u16((uint16_t)fract);
        inv_f3 = vdup_n_u16((uint16_t)(32 - fract));

        a0 = vld1_u16(p0);
        b0 = vld1_u16(p0 + 1);
        a1 = vld1_u16(p1);
        b1 = vld1_u16(p1 + 1);
        a2 = vld1_u16(p2);
        b2 = vld1_u16(p2 + 1);
        a3 = vld1_u16(p3);
        b3 = vld1_u16(p3 + 1);

        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a0, inv_f0), b0, f0), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a1, inv_f1), b1, f1), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a2, inv_f2), b2, f2), 5));
        pu2_dst_tmp1 += dst_strd;
        vst1_u16(pu2_dst_tmp1, vrshr_n_u16(vmla_u16(vmul_u16(a3, inv_f3), b3, f3), 5));
    }
}

