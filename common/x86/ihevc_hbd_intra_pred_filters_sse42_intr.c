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
*  ihevc_hbd_intra_pred_filters_sse42_intr.c
*
* @brief
*  Contains function definitions for high bit depth intra prediction
*  interpolation filters (SSE4.2).
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_intra_pred_luma_ref_substitution_sse42()
*  - ihevc_hbd_intra_pred_ref_filtering_sse42()
*  - ihevc_hbd_intra_pred_luma_planar_sse42()
*  - ihevc_hbd_intra_pred_luma_dc_sse42()
*  - ihevc_hbd_intra_pred_luma_ver_sse42()
*  - ihevc_hbd_intra_pred_luma_horz_sse42()
*  - ihevc_hbd_intra_pred_luma_mode2_sse42()
*  - ihevc_hbd_intra_pred_luma_mode_18_34_sse42()
*  - ihevc_hbd_intra_pred_luma_mode_3_to_9_sse42()
*  - ihevc_hbd_intra_pred_luma_mode_11_to_17_sse42()
*  - ihevc_hbd_intra_pred_luma_mode_19_to_25_sse42()
*  - ihevc_hbd_intra_pred_luma_mode_27_to_33_sse42()
*
* @remarks
*  None
*
*******************************************************************************
*/

/**
*******************************************************************************
* @file
*  ihevc_hbd_intra_pred_filters_x86_intr.c
*
* @brief
*  Contains function Definition for intra prediction  interpolation filters
*
*
* @author
* Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_intra_pred_luma_planar()
*  - ihevc_hbd_intra_pred_luma_dc()
*  - ihevc_hbd_intra_pred_luma_horz()
*  - ihevc_hbd_intra_pred_luma_ver()
*  - ihevc_hbd_intra_pred_luma_mode2()
*  - ihevc_hbd_intra_pred_luma_mode_18_34()
*  - ihevc_hbd_intra_pred_luma_mode_3_to_9()
*  - ihevc_hbd_intra_pred_luma_mode_11_to_17()
*  - ihevc_hbd_intra_pred_luma_mode_19_to_25()
*  - ihevc_hbd_intra_pred_luma_mode_27_to_33()
*  - ihevc_hbd_intra_pred_luma_ref_substitution()
*
* @remarks
*  None
*
*******************************************************************************
*/


/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <stdlib.h>

#include "ihevc_typedefs.h"
#include "ihevc_intra_pred.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_common_tables.h"
#include "ihevc_defs.h"
#include "ihevc_hbd_tables_x86_intr.h"

#include <immintrin.h>


/****************************************************************************/
/* Constant Macros                                                          */
/****************************************************************************/
#define MAX_CU_SIZE 64
#define T32_4NT 128
#define T16_4NT 64


/****************************************************************************/
/* Function Macros                                                          */
/****************************************************************************/

/*****************************************************************************/
/* global tables Definition                                                  */
/*****************************************************************************/

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for pu1_ref substitution for higher bit depth
*
*
* @par Description:
*    Reference substitution process for samples unavailable  for prediction
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
* @param[in] nbr_flags
*  WORD32 neighbor availability flags
*
* @param[in] nt
*  WORD32 transform Block size
*
* @param[out] pu2_dst
*  UWORD16 pointer to destination

* @param[in] dst_strd
*  WORD32 Destination stride
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/


void ihevc_hbd_intra_pred_luma_ref_substitution_sse42(UWORD16 *pu2_top_left,
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
    dc_val = 1 << (bit_depth - 1);


    /* Neighbor Flag Structure */
    /* MSB ---> LSB */
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
            bot_left = (nbr_flags & 0x8)>>3;
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
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt + 1 + i] = pu2_top[i];
            }
            else
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt + 1 + i] = 0;
            }

            if(tp_right)
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt + 1 + i] = pu2_top[i];
            }
            else
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt + 1 + i] = 0;
            }
            next = 1;

            /* If bottom -left is not available, reverse substitution process */
            if(bot_left == 0)
            {
                WORD32 a_nbr_flag[5] = { bot_left, left, tp_left, top, tp_right};

                /* Check for the 1st available sample from bottom-left */
                while(!a_nbr_flag[next])
                    next++;

                /* If Left, top-left are available */
                if(next <= 2)
                {
                    idx = nt * next;
                    pu2_ref = pu2_dst[idx];
                    for(i = 0; i < idx; i++)
                        pu2_dst[i] = pu2_ref;
                }
                else /* If top, top-right are available */
                {
                    /* Idx is changed to copy 1 pixel value for top-left ,if top-left is not available */
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
                for(i = 0; i < nt; i++)
                    pu2_dst[nt + i] = pu2_dst[nt - 1];
            }
            /* If top-left is Unavailable, copy the last left value */
            if(tp_left == 0)
                pu2_dst[two_nt] = pu2_dst[two_nt - 1];
            /* If top is Unavailable, copy the last top-left value */
            if(top == 0)
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt + 1 + i] = pu2_dst[two_nt];
            }
            /* If to right is Unavailable, copy the last top value */
            if(tp_right == 0)
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[three_nt + 1 + i] = pu2_dst[three_nt];
            }
        }

        if(nt == 16)
        {
            WORD32 nbr_flags_temp = 0;
            nbr_flags_temp = ((nbr_flags & 0xC)>>2) + ((nbr_flags & 0xC0) >> 4)
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
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt + 1 + i] = pu2_top[i];
            }
            else
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt + 1 + i] = 0;
            }

            if(nbr_flags & 0x3000)
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt + 1 + i] = pu2_top[i];
            }
            else
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt + 1 + i] = 0;
            }
            /* compute trailing zeors based on nbr_flag for substitution process of below left see section . */
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
                /* Reverse Substitution Process */
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
                /* To Obtain the next unavailable idx flag after reverse neighbor substitution */
                /* Devide by 8 to obtain the original index */
                frwd_nbr_flag = (nbr_id_from_bl >> 3);

                /* The Top-left flag is at the last bit location of nbr_flags */
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
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt + 1 + i] = pu2_top[i];
            }
            else
            {
                for(i = 0; i < nt; i++)
                    pu2_dst[two_nt + 1 + i] = 0;
            }

            if(nbr_flags & 0xF000)
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt + 1 + i] = pu2_top[i];
            }
            else
            {
                for(i = nt; i < two_nt; i++)
                    pu2_dst[two_nt + 1 + i] = 0;
            }
            /* compute trailing ones based on mbr_flag for substitution process of below left see section . */
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
                /* Reverse Substitution Process */
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
                /* To Obtain the next unavailable idx flag after reverse neighbor substitution */
                /* Devide by 8 to obtain the original index */
                frwd_nbr_flag = (nbr_id_from_bl >> 3);

                /* The Top-left flag is at the last bit location of nbr_flags */
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
*    Reference DC filtering for neighboring samples dependent  on TU size and
*    mode  Refer to section 8.4.4.2.3 in the standard
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @param[in] intra_smoothing_flags
*  integer bit 3 indicates if intra smoothing is enabled/disabled
*          unconditionally. this is applicable to frext profiles only
*          bit 0 indicates strong intra smoothing enabled/disabled
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_ref_filtering_sse42(UWORD16 *pu2_src,
                                              WORD32 nt,
                                              UWORD16 *pu2_dst,
                                              WORD32 mode,
                                              WORD32 intra_smoothing_flags,
                                              UWORD8 bit_depth)
{
    WORD32 filter_flag;
    WORD32 i; /* Generic indexing variable */
    WORD32 four_nt = 4 * nt;
    UWORD16 au2_flt[(4 * MAX_CU_SIZE) + 1];
    WORD32 bi_linear_int_flag = 0;
    WORD32 abs_cond_left_flag = 0;
    WORD32 abs_cond_top_flag = 0;
    WORD32 dc_val = 1 << (bit_depth - 5);
    __m128i src_temp1, src_temp2, src_temp3, src_temp7;
    __m128i src_temp4, src_temp5, src_temp6, src_temp8;
    __m128i clip_tmp_8x16b;

    WORD32 intra_smoothing_disabled = (intra_smoothing_flags >> 3);
    WORD32 strong_intra_smoothing_enable_flag = intra_smoothing_flags & 1;

    clip_tmp_8x16b = _mm_set1_epi16((WORD16)65535);
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
            if(nt== 4)
            {
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_src));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_src+8));
                _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst + 8), src_temp2);
                pu2_dst[four_nt] = pu2_src[four_nt];
            }
            else if(nt ==8)
            {
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_src));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_src+8));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_src+16));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_src+24));

                _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp4);


                pu2_dst[four_nt] = pu2_src[four_nt];
            }
            else if(nt ==16)
            {
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_src));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_src+8));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_src+16));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_src+24));
                src_temp5 = _mm_loadu_si128((__m128i *)(pu2_src+32));
                src_temp6 = _mm_loadu_si128((__m128i *)(pu2_src+40));
                src_temp7 = _mm_loadu_si128((__m128i *)(pu2_src+48));
                src_temp8 = _mm_loadu_si128((__m128i *)(pu2_src+56));

                _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+32), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+40), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+48), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+56), src_temp8);

                pu2_dst[four_nt] = pu2_src[four_nt];
            }
            else if(nt ==32)
            {
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_src));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_src+8));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_src+16));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_src+24));
                src_temp5 = _mm_loadu_si128((__m128i *)(pu2_src+32));
                src_temp6 = _mm_loadu_si128((__m128i *)(pu2_src+40));
                src_temp7 = _mm_loadu_si128((__m128i *)(pu2_src+48));
                src_temp8 = _mm_loadu_si128((__m128i *)(pu2_src+56));

                _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+32), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+40), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+48), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+56), src_temp8);

                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_src+64));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_src+72));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_src+80));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_src+88));
                src_temp5 = _mm_loadu_si128((__m128i *)(pu2_src+96));
                src_temp6 = _mm_loadu_si128((__m128i *)(pu2_src+104));
                src_temp7 = _mm_loadu_si128((__m128i *)(pu2_src+112));
                src_temp8 = _mm_loadu_si128((__m128i *)(pu2_src+120));

                _mm_storeu_si128((__m128i *)(pu2_dst+64), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+72), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+80), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+88), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+96), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+104), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+112), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+120), src_temp8);

                pu2_dst[four_nt] = pu2_src[four_nt];
            }
        }
    }
    else
    {
        /* If strong intra smoothin is enabled and transform size is 32 */
        if((1 == strong_intra_smoothing_enable_flag) && (32 == nt))
        {
            /* Strong Intra Filtering */
            abs_cond_top_flag = (abs(pu2_src[2 * nt] + pu2_src[4 * nt]
                                            - (2 * pu2_src[3 * nt]))) < dc_val;
            abs_cond_left_flag = (abs(pu2_src[2 * nt] + pu2_src[0]
                                            - (2 * pu2_src[nt]))) < dc_val;

            bi_linear_int_flag = ((1 == abs_cond_left_flag)
                                    && (1 == abs_cond_top_flag));
        }
        /* Extremities Untouched */
        au2_flt[0] = pu2_src[0];
        au2_flt[4 * nt] = pu2_src[4 * nt];

        /* Strong filtering of reference samples */
        if(1 == bi_linear_int_flag)
        {
            au2_flt[2 * nt] = pu2_src[2 * nt];

            for(i = 1; i < (2 * nt) ; i++ )
                au2_flt[i] = (((2 * nt) - i) * pu2_src[0] + i * pu2_src[2 * nt] + 32) >> 6;

            for(i = 1; i < (2 * nt) ; i++ )
                au2_flt[i + (2 * nt)] = (((2 * nt)-i)*pu2_src[2 * nt] + i * pu2_src[4 * nt] + 32) >> 6;
        }
        else
        {
            __m128i const_value_8x16;
            __m128i temp1, temp2, temp3;

            const_value_8x16 = _mm_set1_epi16(2);

            au2_flt[0] = pu2_src[0];
            au2_flt[4 * nt] = pu2_src[4 * nt];

            /* Perform bilinear filtering of Reference Samples */
            for(i = 0; i < (four_nt); i += 8)
            {
                temp1 = _mm_loadu_si128((__m128i *)(pu2_src+i));
                temp2 = _mm_loadu_si128((__m128i *)(pu2_src+1+i));
                temp3 = _mm_loadu_si128((__m128i *)(pu2_src+2+i));

                temp2 = _mm_slli_epi16(temp2,  1);

                temp1 = _mm_add_epi16(temp1, temp2);
                temp1 = _mm_add_epi16(temp1, temp3);
                temp1 = _mm_add_epi16(temp1, const_value_8x16);

                temp1 = _mm_srai_epi16(temp1,  2);

                _mm_storeu_si128((__m128i *)(au2_flt+1+i), temp1);
            }
            au2_flt[4 * nt] = pu2_src[4 * nt];
        }

        if(nt == 4)
        {
            src_temp1 = _mm_loadu_si128((__m128i *)(au2_flt));
            src_temp2 = _mm_loadu_si128((__m128i *)(au2_flt+8));
            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp2);
            pu2_dst[four_nt] = au2_flt[four_nt];
        }
        else if(nt == 8)
        {
            src_temp1 = _mm_loadu_si128((__m128i *)(au2_flt));
            src_temp2 = _mm_loadu_si128((__m128i *)(au2_flt+8));
            src_temp3 = _mm_loadu_si128((__m128i *)(au2_flt+16));
            src_temp4 = _mm_loadu_si128((__m128i *)(au2_flt+24));

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp4);

            pu2_dst[four_nt] = au2_flt[four_nt];
        }
        else if(nt == 16)
        {
            src_temp1 = _mm_loadu_si128((__m128i *)(au2_flt));
            src_temp2 = _mm_loadu_si128((__m128i *)(au2_flt+8));
            src_temp3 = _mm_loadu_si128((__m128i *)(au2_flt+16));
            src_temp4 = _mm_loadu_si128((__m128i *)(au2_flt+24));
            src_temp5 = _mm_loadu_si128((__m128i *)(au2_flt+32));
            src_temp6 = _mm_loadu_si128((__m128i *)(au2_flt+40));
            src_temp7 = _mm_loadu_si128((__m128i *)(au2_flt+48));
            src_temp8 = _mm_loadu_si128((__m128i *)(au2_flt+56));

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+32), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+40), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+48), src_temp7);
            _mm_storeu_si128((__m128i *)(pu2_dst+56), src_temp8);

            pu2_dst[four_nt] = au2_flt[four_nt];
        }
        else if(nt == 32)
        {
            src_temp1 = _mm_loadu_si128((__m128i *)(au2_flt));
            src_temp2 = _mm_loadu_si128((__m128i *)(au2_flt+8));
            src_temp3 = _mm_loadu_si128((__m128i *)(au2_flt+16));
            src_temp4 = _mm_loadu_si128((__m128i *)(au2_flt+24));
            src_temp5 = _mm_loadu_si128((__m128i *)(au2_flt+32));
            src_temp6 = _mm_loadu_si128((__m128i *)(au2_flt+40));
            src_temp7 = _mm_loadu_si128((__m128i *)(au2_flt+48));
            src_temp8 = _mm_loadu_si128((__m128i *)(au2_flt+56));

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+32), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+40), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+48), src_temp7);
            _mm_storeu_si128((__m128i *)(pu2_dst+56), src_temp8);

            src_temp1 = _mm_loadu_si128((__m128i *)(au2_flt+64));
            src_temp2 = _mm_loadu_si128((__m128i *)(au2_flt+72));
            src_temp3 = _mm_loadu_si128((__m128i *)(au2_flt+80));
            src_temp4 = _mm_loadu_si128((__m128i *)(au2_flt+88));
            src_temp5 = _mm_loadu_si128((__m128i *)(au2_flt+96));
            src_temp6 = _mm_loadu_si128((__m128i *)(au2_flt+104));
            src_temp7 = _mm_loadu_si128((__m128i *)(au2_flt+112));
            src_temp8 = _mm_loadu_si128((__m128i *)(au2_flt+120));

            _mm_storeu_si128((__m128i *)(pu2_dst+64), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+72), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+80), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+88), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+96), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+104), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+112), src_temp7);
            _mm_storeu_si128((__m128i *)(pu2_dst+120), src_temp8);

            pu2_dst[four_nt] = au2_flt[four_nt];
        }
    }
}

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma planar
*
* @par Description:
*    Planar Intraprediction with reference neighboring samples location
*    pointed by 'pu2_ref' to the TU block location  pointed by 'pu2_dst'  Refer
*    to section 8.4.4.2.4 in the standard
*
* @param[in] pu2_src
*  UWORD8 pointer to the source
*
* @param[out] pu2_dst
*  UWORD8 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
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
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_luma_planar_sse42(UWORD16 *pu2_ref,
                                            WORD32 src_strd,
                                            UWORD16 *pu2_dst,
                                            WORD32 dst_strd,
                                            WORD32 nt,
                                            WORD32 mode,
                                            UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 log2nt = 5;
    WORD32 two_nt, three_nt;

    __m128i const_temp_4x32b, const_temp1_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b;
    __m128i col_8x16b, const_temp5_4x32b, const_temp6_4x32b, zero_8x16b, mask_4x32b, const_temp7_4x32b;

    switch(nt)
    {
        case 32:
            log2nt = 5;
            break;
        case 16:
            log2nt = 4;
            break;
        case 8:
            log2nt = 3;
            break;
        case 4:
            log2nt = 2;
            break;
        default:
            break;
    }

    two_nt = 2 * nt;
    three_nt = 3 * nt;

    /* Planar filtering */

    /* setting vallues in  registera */

    const_temp_4x32b  = _mm_set1_epi16(pu2_ref[three_nt + 1]);
    const_temp1_4x32b = _mm_set1_epi16(pu2_ref[nt - 1]);
    const_temp4_4x32b = _mm_set1_epi16(nt - 1);
    const_temp6_4x32b = _mm_set1_epi16(nt);
    const_temp7_4x32b = _mm_set1_epi16(4);

    zero_8x16b = _mm_set1_epi32(0);

    mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

    if(nt%8==0)
    {
        /* for nt multiple of 8 */
        const_temp7_4x32b = _mm_set1_epi16(8);

        for(row = 0; row < nt; row++)
        {
            __m128i res_temp_8x16b, row_8x16b, res_temp1_8x16b, res_temp2_8x16b;
            __m128i res_temp3_8x16b;

            const_temp2_4x32b  = _mm_set1_epi16(pu2_ref[two_nt - 1 - row]);
            const_temp3_4x32b  = _mm_set1_epi16((row + 1));
            row_8x16b = _mm_set1_epi16((nt - 1 - row));

            const_temp5_4x32b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
            col_8x16b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);

            const_temp5_4x32b = _mm_sub_epi16(const_temp4_4x32b, const_temp5_4x32b);

            /* (row + 1) * pu2_ref[nt - 1] */
            res_temp_8x16b  = _mm_mullo_epi16(const_temp3_4x32b,  const_temp1_4x32b);

            /* (row + 1) * pu2_ref[nt - 1] + nt) */
            res_temp_8x16b = _mm_add_epi16(res_temp_8x16b, const_temp6_4x32b);

            for(col = 0; col < nt; col += 8)
            {
                __m128i src_temp_8x16b;

                /* loding 16bit 8 pixles */
                src_temp_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 + col));

                /* (nt - 1 - row) * pu2_ref[two_nt + 1 + col] */
                res_temp1_8x16b  = _mm_mullo_epi16(src_temp_8x16b,  row_8x16b);

                /* (col + 1) * pu2_ref[three_nt + 1] */
                res_temp2_8x16b  = _mm_mullo_epi16(const_temp_4x32b,  col_8x16b);

                /* (nt - 1 - col)* pu2_ref[two_nt - 1 - row] */
                res_temp3_8x16b  = _mm_mullo_epi16(const_temp2_4x32b,  const_temp5_4x32b);

                res_temp1_8x16b = _mm_add_epi16(res_temp_8x16b, res_temp1_8x16b);
                res_temp1_8x16b = _mm_add_epi16(res_temp1_8x16b, res_temp2_8x16b);
                res_temp1_8x16b = _mm_add_epi16(res_temp1_8x16b, res_temp3_8x16b);

                res_temp1_8x16b = _mm_srli_epi16(res_temp1_8x16b, (log2nt + 1));


                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd) + col), res_temp1_8x16b);

                const_temp5_4x32b = _mm_sub_epi16(const_temp5_4x32b, const_temp7_4x32b);
                col_8x16b = _mm_add_epi16(col_8x16b, const_temp7_4x32b);
            } /* inner loop ends here */
        }
    }
    else
    {
        /* for nt multiple of 4 */
        for(row = 0; row < nt; row++)
        {
            __m128i res_temp_8x16b, row_8x16b, res_temp1_8x16b, res_temp2_8x16b;
            __m128i res_temp3_8x16b;

            const_temp2_4x32b  = _mm_set1_epi16(pu2_ref[two_nt - 1 - row]);
            const_temp3_4x32b  = _mm_set1_epi16((row + 1));
            row_8x16b = _mm_set1_epi16((nt - 1 - row));

            const_temp5_4x32b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
            col_8x16b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);

            const_temp5_4x32b = _mm_sub_epi16(const_temp4_4x32b, const_temp5_4x32b);

            /* (row + 1) * pu2_ref[nt - 1] */
            res_temp_8x16b  = _mm_mullo_epi16(const_temp3_4x32b,  const_temp1_4x32b);

            /* (row + 1) * pu2_ref[nt - 1] + nt) */
            res_temp_8x16b = _mm_add_epi16(res_temp_8x16b, const_temp6_4x32b);

            for(col = 0; col < nt; col += 4)
            {
                __m128i src_temp_8x16b;

                /* loding 16bit 8 pixles */
                src_temp_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 + col));


                /* (nt - 1 - row) * pu2_ref[two_nt + 1 + col] */
                res_temp1_8x16b  = _mm_mullo_epi16(src_temp_8x16b,  row_8x16b);

                /* (col + 1) * pu2_ref[three_nt + 1] */
                res_temp2_8x16b  = _mm_mullo_epi16(const_temp_4x32b,  col_8x16b);

                /* (nt - 1 - col)* pu2_ref[two_nt - 1 - row] */
                res_temp3_8x16b  = _mm_mullo_epi16(const_temp2_4x32b,  const_temp5_4x32b);

                res_temp1_8x16b = _mm_add_epi16(res_temp_8x16b, res_temp1_8x16b);
                res_temp1_8x16b = _mm_add_epi16(res_temp1_8x16b, res_temp2_8x16b);
                res_temp1_8x16b = _mm_add_epi16(res_temp1_8x16b, res_temp3_8x16b);

                res_temp1_8x16b = _mm_srli_epi16(res_temp1_8x16b, (log2nt + 1));


                _mm_storel_epi64((__m128i *)(pu2_dst+(row * dst_strd) + col), res_temp1_8x16b);
                const_temp5_4x32b = _mm_sub_epi32(const_temp5_4x32b, const_temp7_4x32b);
                col_8x16b = _mm_add_epi32(col_8x16b, const_temp7_4x32b);
            } /* inner loop ends here */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma dc
*
* @par Description:
*   Intraprediction for DC mode with reference neighboring  samples location
*   pointed by 'pu2_ref' to the TU block  location pointed by 'pu2_dst'  Refer
*   to section 8.4.4.2.5 in the standard
*
* @param[in] pu2_src
*  UWORD8 pointer to the source
*
* @param[out] pu2_dst
*  UWORD8 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
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
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_luma_dc_sse42(UWORD16 *pu2_ref,
                                        WORD32 src_strd,
                                        UWORD16 *pu2_dst,
                                        WORD32 dst_strd,
                                        WORD32 nt,
                                        WORD32 mode,
                                        UWORD8 bit_depth)
{
    WORD32 acc_dc;
    WORD32 dc_val, two_dc_val, three_dc_val;
    WORD32 row;
    WORD32 log2nt = 5;
    WORD32 two_nt, three_nt;
    __m128i src_temp1, src_temp7, src_temp3, src_temp4, src_temp5, src_temp6;
    __m128i src_temp8, src_temp9, src_temp10, src_temp2;
    __m128i m_zero = _mm_set1_epi32(0);
    __m128i const_16x8b;

    switch(nt)
    {
        case 32:
            log2nt = 5;
            break;
        case 16:
            log2nt = 4;
            break;
        case 8:
            log2nt = 3;
            break;
        case 4:
            log2nt = 2;
            break;
        default:
            break;
    }
    two_nt = 2 * nt;
    three_nt = 3 * nt;

    acc_dc = 0;
    /* Calculate DC value for the transform block */

    if(nt == 32)
    {
        __m128i temp;
        WORD32 itr_count;
        const_16x8b = _mm_cmpeq_epi16(m_zero, m_zero);
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);
        src_temp3 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt));
        src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+8));
        src_temp5 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+16));
        src_temp6 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+24));
        src_temp7 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+32));
        src_temp8 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+40));
        src_temp9 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+48));
        src_temp10 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+56));

        src_temp3 = _mm_add_epi16(src_temp3, src_temp4);
        src_temp5 = _mm_add_epi16(src_temp5, src_temp6);
        src_temp7 = _mm_add_epi16(src_temp7, src_temp8);
        src_temp9 = _mm_add_epi16(src_temp9, src_temp10);

        src_temp3 = _mm_add_epi16(src_temp3, src_temp5);
        src_temp7 = _mm_add_epi16(src_temp7, src_temp9);

        src_temp3 = _mm_add_epi16(src_temp3, src_temp7);

        src_temp3 = _mm_madd_epi16(src_temp3, const_16x8b);

        src_temp4 = _mm_shuffle_epi32(src_temp3, 0x01);
        src_temp5 = _mm_shuffle_epi32(src_temp3, 0x02);
        src_temp6 = _mm_shuffle_epi32(src_temp3, 0x03);

        src_temp3 = _mm_add_epi32(src_temp3, src_temp4);
        src_temp5 = _mm_add_epi32(src_temp5, src_temp6);
        src_temp3 = _mm_add_epi32(src_temp3, src_temp5);

        acc_dc = _mm_cvtsi128_si32(src_temp3);

        acc_dc += pu2_ref[three_nt];
        acc_dc -= pu2_ref[two_nt];

        /* computing acc_dc value */
        dc_val = (acc_dc + nt) >> (log2nt + 1);

        two_dc_val = 2 * dc_val;
        three_dc_val = 3 * dc_val;

        temp = _mm_set1_epi16(dc_val);

        for(itr_count = 0; itr_count < 2; itr_count++)
        {
            /* pu2_dst[(row * dst_strd) + col] = dc_val; */
            _mm_storeu_si128((__m128i *)(pu2_dst+((0) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)), temp);

            _mm_storeu_si128((__m128i *)(pu2_dst+((8) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((9) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((10) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((11) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((12) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((13) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((14) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+((15) * dst_strd)), temp);

            _mm_storeu_si128((__m128i *)(pu2_dst+8+((0)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((1)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((2)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((3)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((4)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((5)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((6)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((7)  * dst_strd)), temp);

            _mm_storeu_si128((__m128i *)(pu2_dst+8+((8)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((9)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((10) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((11) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((12) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((13) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((14) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((15) * dst_strd)), temp);

            _mm_storeu_si128((__m128i *)(pu2_dst+16+((0)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((1)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((2)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((3)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((4)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((5)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((6)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((7)  * dst_strd)), temp);

            _mm_storeu_si128((__m128i *)(pu2_dst+16+((8)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((9)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((10) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((11) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((12) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((13) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((14) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((15) * dst_strd)), temp);

            _mm_storeu_si128((__m128i *)(pu2_dst+24+((0)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((1)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((2)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((3)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((4)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((5)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((6)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((7)  * dst_strd)), temp);

            _mm_storeu_si128((__m128i *)(pu2_dst+24+((8)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((9)  * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((10) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((11) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((12) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((13) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((14) * dst_strd)), temp);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((15) * dst_strd)), temp);

            pu2_dst += 16*dst_strd;
        }
    }
    else
    {
        __m128i  zero_8x16b, mask_4x32b;
        __m128i sm1 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASK4_HBD[0]);

        /* DC filtering for the first top row and first left column */

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        if(nt ==4) /* nt multiple of 4 */
        {
            src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt));
            src_temp2 =  _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 ));

            src_temp4 = _mm_hadd_epi16(src_temp4, m_zero);
            src_temp4 = _mm_hadd_epi16(src_temp4, m_zero);
            src_temp4 = _mm_hadd_epi16(src_temp4, m_zero);

            acc_dc = _mm_cvtsi128_si32(src_temp4);
            acc_dc += pu2_ref[three_nt];
            acc_dc -= pu2_ref[two_nt];

            /* computing acc_dc value */

            dc_val = (acc_dc + nt) >> (log2nt + 1);

            three_dc_val = 3 * dc_val;

            /* loding 8-bit 16 pixel */
            src_temp1 = _mm_set1_epi16(three_dc_val+2);
            two_dc_val = 2 * dc_val;

            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2 */
            src_temp2 = _mm_add_epi16(src_temp2, src_temp1);

            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2) >> 2 */
            src_temp2 = _mm_srli_epi16(src_temp2, 2);

            _mm_storel_epi64((__m128i *)pu2_dst, src_temp2);

            /* retore  first value */
            pu2_dst[0] = ((pu2_ref[two_nt - 1] + two_dc_val + pu2_ref[two_nt + 1] + 2)
                >> 2);

            for(row = 1; row < nt; row++)
                pu2_dst[row * dst_strd] = (pu2_ref[two_nt - 1 - row] + three_dc_val + 2)
                >> 2;

            src_temp2 = _mm_insert_epi16(src_temp2, dc_val, 0);

            src_temp2 =  _mm_shuffle_epi8(src_temp2, sm1);
            src_temp3 =  _mm_shuffle_epi8(src_temp2, sm1);
            src_temp4 =  _mm_shuffle_epi8(src_temp2, sm1);

            src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[(1 * dst_strd)+0], 0);
            src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[(2 * dst_strd)+0], 0);
            src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[(3 * dst_strd)+0], 0);

            _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp2);
            _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp3);
            _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp4);

        }
        else if(nt==8)
        {
            /* if nt%8==0 */

            src_temp3 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt));
            src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+1));
            const_16x8b = _mm_cmpeq_epi16(src_temp3, src_temp3);
            const_16x8b = _mm_srli_epi16(const_16x8b, 15);

            src_temp3 = _mm_add_epi16(src_temp3, src_temp4);
            src_temp3 = _mm_madd_epi16(src_temp3, const_16x8b);

            src_temp4 = _mm_shuffle_epi32(src_temp3, 0x01);
            src_temp5 = _mm_shuffle_epi32(src_temp3, 0x02);
            src_temp6 = _mm_shuffle_epi32(src_temp3, 0x03);

            src_temp3 = _mm_add_epi32(src_temp3, src_temp4);
            src_temp5 = _mm_add_epi32(src_temp5, src_temp6);
            src_temp3 = _mm_add_epi32(src_temp3, src_temp5);

            acc_dc = _mm_cvtsi128_si32(src_temp3);

            /* computing acc_dc value */

            dc_val = (acc_dc + nt) >> (log2nt + 1);

            three_dc_val = 3 * dc_val;
            src_temp1 = _mm_set1_epi16(three_dc_val+2);
            two_dc_val = 2 * dc_val;

            /* loding 8-bit 16 pixel */
            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1));

            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2 */
            src_temp2 = _mm_add_epi16(src_temp2, src_temp1);

            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2)>>2 */
            src_temp2 = _mm_srli_epi16(src_temp2, 2);

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp2);

            /* retore  first value */
            pu2_dst[0] = ((pu2_ref[two_nt - 1] + two_dc_val + pu2_ref[two_nt + 1] + 2)
                >> 2);

            for(row = 1; row < nt; row++)
                pu2_dst[row * dst_strd] = (pu2_ref[two_nt - 1 - row] + three_dc_val + 2)
                >> 2;

            /* Fill the remaining rows with DC value */

            src_temp1 = _mm_set1_epi16(dc_val);
            src_temp2 = _mm_set1_epi16(dc_val);
            src_temp3 = _mm_set1_epi16(dc_val);
            src_temp4 = _mm_set1_epi16(dc_val);
            src_temp5 = _mm_set1_epi16(dc_val);
            src_temp6 = _mm_set1_epi16(dc_val);
            src_temp7 = _mm_set1_epi16(dc_val);

            src_temp1 = _mm_insert_epi16(src_temp1, pu2_dst[((1)*dst_strd)], 0);
            src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[((2)*dst_strd)], 0);
            src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[((3)*dst_strd)], 0);
            src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[((4)*dst_strd)], 0);
            src_temp5 = _mm_insert_epi16(src_temp5, pu2_dst[((5)*dst_strd)], 0);
            src_temp6 = _mm_insert_epi16(src_temp6, pu2_dst[((6)*dst_strd)], 0);
            src_temp7 = _mm_insert_epi16(src_temp7, pu2_dst[((7)*dst_strd)], 0);

            _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)), src_temp7);
        }
        else if(nt==16)
        {
            /* if nt%8==0 */
            src_temp3 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt));
            src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+8));
            src_temp5 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+17));
            src_temp6 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+25));

            const_16x8b = _mm_cmpeq_epi16(src_temp3, src_temp3);
            const_16x8b = _mm_srli_epi16(const_16x8b, 15);

            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1));
            src_temp10 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1+8));

            src_temp3 = _mm_add_epi16(src_temp3, src_temp4);
            src_temp5 = _mm_add_epi16(src_temp5, src_temp6);
            src_temp3 = _mm_add_epi16(src_temp3, src_temp5);

            src_temp3 = _mm_madd_epi16(src_temp3, const_16x8b);

            src_temp4 = _mm_shuffle_epi32(src_temp3, 0x01);
            src_temp5 = _mm_shuffle_epi32(src_temp3, 0x02);
            src_temp6 = _mm_shuffle_epi32(src_temp3, 0x03);

            src_temp3 = _mm_add_epi32(src_temp3, src_temp4);
            src_temp5 = _mm_add_epi32(src_temp5, src_temp6);
            src_temp3 = _mm_add_epi32(src_temp3, src_temp5);

            acc_dc = _mm_cvtsi128_si32(src_temp3);

            /* computing acc_dc value */

            dc_val = (acc_dc + nt) >> (log2nt + 1);

            three_dc_val = 3 * dc_val;
            src_temp1 = _mm_set1_epi16(three_dc_val+2);
            two_dc_val = 2 * dc_val;

            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2 */
            src_temp2 = _mm_add_epi16(src_temp2, src_temp1);
            src_temp10 = _mm_add_epi16(src_temp10, src_temp1);
            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2)>>2 */
            src_temp2 = _mm_srli_epi16(src_temp2, 2);
            src_temp10 = _mm_srli_epi16(src_temp10, 2);

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp10);

            /* retore  first value */
            pu2_dst[0] = ((pu2_ref[two_nt - 1] + two_dc_val + pu2_ref[two_nt + 1] + 2)
                >> 2);

            for(row = 1; row < nt; row++)
                pu2_dst[row * dst_strd] = (pu2_ref[two_nt - 1 - row] + three_dc_val + 2)
                >> 2;
            /* Fill the remaining rows with DC value */
            src_temp1 =  _mm_set1_epi16(dc_val);
            src_temp2 =  _mm_set1_epi16(dc_val);
            src_temp3 =  _mm_set1_epi16(dc_val);
            src_temp4 =  _mm_set1_epi16(dc_val);
            src_temp5 =  _mm_set1_epi16(dc_val);
            src_temp6 =  _mm_set1_epi16(dc_val);
            src_temp7 =  _mm_set1_epi16(dc_val);

            /* for(row = 1; row < nt; row +=8) */
            {

                _mm_storeu_si128((__m128i *)(pu2_dst+8+((1) * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((2) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((3) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((4) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((5) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((6) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((7) * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((8) * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((9) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((10) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((11) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((12) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((13) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((14) * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((15) * dst_strd)), src_temp7);

                src_temp1 = _mm_insert_epi16(src_temp1, pu2_dst[((1)*dst_strd)], 0);
                src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[((2)*dst_strd)], 0);
                src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[((3)*dst_strd)], 0);
                src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[((4)*dst_strd)], 0);
                src_temp5 = _mm_insert_epi16(src_temp5, pu2_dst[((5)*dst_strd)], 0);
                src_temp6 = _mm_insert_epi16(src_temp6, pu2_dst[((6)*dst_strd)], 0);
                src_temp7 = _mm_insert_epi16(src_temp7, pu2_dst[((7)*dst_strd)], 0);

                _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)), src_temp7);

                src_temp1 = _mm_insert_epi16(src_temp1, pu2_dst[((8)*dst_strd)] , 0);
                src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[((9)*dst_strd)] , 0);
                src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[((10)*dst_strd)], 0);
                src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[((11)*dst_strd)], 0);
                src_temp5 = _mm_insert_epi16(src_temp5, pu2_dst[((12)*dst_strd)], 0);
                src_temp6 = _mm_insert_epi16(src_temp6, pu2_dst[((13)*dst_strd)], 0);
                src_temp7 = _mm_insert_epi16(src_temp7, pu2_dst[((14)*dst_strd)], 0);

                _mm_storeu_si128((__m128i *)(pu2_dst+((8) * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+((9) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+((10) * dst_strd)), src_temp3);

                src_temp1 = _mm_insert_epi16(src_temp1, pu2_dst[((15)*dst_strd)], 0);

                _mm_storeu_si128((__m128i *)(pu2_dst+((11) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+((12) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+((13) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+((14) * dst_strd)), src_temp7);

                _mm_storeu_si128((__m128i *)(pu2_dst+((15) * dst_strd)), src_temp1);
            }
        }
        else if(nt==32)
        {
            /* if nt%8==0 */
            __m128i src_temp11, src_temp12, src_temp13, src_temp14, src_temp15, src_temp16, src_temp17;

            const_16x8b = _mm_cmpeq_epi16(m_zero, m_zero);
            const_16x8b = _mm_srli_epi16(const_16x8b, 15);
            src_temp3 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt));
            src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+8));
            src_temp5 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+16));
            src_temp6 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+24));
            src_temp7 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+32));
            src_temp8 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+40));
            src_temp9 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+48));
            src_temp10 =  _mm_loadu_si128((__m128i *)(pu2_ref+nt+56));

            /* loding 8-bit 16 pixel */
            src_temp11 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 ));
            src_temp12 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 +8));
            src_temp13 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 + 16));
            src_temp14 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 + 24));

            src_temp3 = _mm_add_epi16(src_temp3, src_temp4);
            src_temp5 = _mm_add_epi16(src_temp5, src_temp6);
            src_temp7 = _mm_add_epi16(src_temp7, src_temp8);
            src_temp9 = _mm_add_epi16(src_temp9, src_temp10);

            src_temp3 = _mm_add_epi16(src_temp3, src_temp5);
            src_temp7 = _mm_add_epi16(src_temp7, src_temp9);

            src_temp3 = _mm_add_epi16(src_temp3, src_temp7);

            src_temp3 = _mm_madd_epi16(src_temp3, const_16x8b);

            src_temp4 = _mm_shuffle_epi32(src_temp3, 0x01);
            src_temp5 = _mm_shuffle_epi32(src_temp3, 0x02);
            src_temp6 = _mm_shuffle_epi32(src_temp3, 0x03);

            src_temp3 = _mm_add_epi32(src_temp3, src_temp4);
            src_temp5 = _mm_add_epi32(src_temp5, src_temp6);
            src_temp3 = _mm_add_epi32(src_temp3, src_temp5);

            acc_dc = _mm_cvtsi128_si32(src_temp3);

            acc_dc += pu2_ref[three_nt];
            acc_dc -= pu2_ref[two_nt];

            /* computing acc_dc value */

            dc_val = (acc_dc + nt) >> (log2nt + 1);

            three_dc_val = 3 * dc_val;
            src_temp1 = _mm_set1_epi16(three_dc_val+2);
            two_dc_val = 2 * dc_val;

            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2 */
            src_temp11 = _mm_add_epi16(src_temp11, src_temp1);
            src_temp12 = _mm_add_epi16(src_temp12, src_temp1);
            src_temp13 = _mm_add_epi16(src_temp13, src_temp1);
            src_temp14 = _mm_add_epi16(src_temp14, src_temp1);

            /* (pu2_ref[two_nt + 1 + col] + three_dc_val + 2)>>2 */
            src_temp11 = _mm_srli_epi16(src_temp11, 2);
            src_temp12 = _mm_srli_epi16(src_temp12, 2);
            src_temp13 = _mm_srli_epi16(src_temp13, 2);
            src_temp14 = _mm_srli_epi16(src_temp14, 2);

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp11);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp12);
            _mm_storeu_si128((__m128i *)(pu2_dst+ 16), src_temp13);
            _mm_storeu_si128((__m128i *)(pu2_dst+ 24), src_temp14);

            /* retore  first value */
            pu2_dst[0] = ((pu2_ref[two_nt - 1] + two_dc_val + pu2_ref[two_nt + 1] + 2)
                >> 2);

            for(row = 1; row < nt; row++)
                pu2_dst[row * dst_strd] = (pu2_ref[two_nt - 1 - row] + three_dc_val + 2)
                >> 2;
            /* Fill the remaining rows with DC value */
            src_temp1 = _mm_insert_epi16(src_temp1, dc_val, 0);

            src_temp2 =  src_temp1;
            src_temp3 = src_temp1;
            src_temp4 =  src_temp1;
            src_temp5 =  src_temp1;
            src_temp6 =  src_temp1;
            src_temp7 =  src_temp1;

            src_temp12 = src_temp1;
            src_temp13 = src_temp1;
            src_temp14 = src_temp1;
            src_temp15 = src_temp1;
            src_temp16 = src_temp1;
            src_temp17 = src_temp1;
            src_temp11 = src_temp1;

            /* for(row = 1; row < nt; row+=8) */
            {
                row=1;
                src_temp1 = _mm_insert_epi16(src_temp1, pu2_dst[((1)*dst_strd)], 0);
                src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[((2)*dst_strd)], 0);
                src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[((3)*dst_strd)], 0);
                src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[((4)*dst_strd)], 0);
                src_temp5 = _mm_insert_epi16(src_temp5, pu2_dst[((5)*dst_strd)], 0);
                src_temp6 = _mm_insert_epi16(src_temp6, pu2_dst[((6)*dst_strd)], 0);
                src_temp7 = _mm_insert_epi16(src_temp7, pu2_dst[((7)*dst_strd)], 0);

                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)+8), src_temp11);
                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)+16), src_temp11);
                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)+24), src_temp11);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)+8), src_temp12);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)+16), src_temp12);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)+24), src_temp12);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)+8), src_temp13);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)+16), src_temp13);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)+24), src_temp13);

                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)+8), src_temp14);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)+16), src_temp14);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)+24), src_temp14);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)+8), src_temp15);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)+16), src_temp15);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)+24), src_temp15);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)+8), src_temp16);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)+16), src_temp16);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)+24), src_temp16);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)+8), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)+16), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)+24), src_temp17);
            }
            for(row = 8; row < nt; row += 8)
            {
                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)), src_temp11);
                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)+8), src_temp11);
                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)+16), src_temp11);
                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)+24), src_temp11);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)), src_temp12);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)+8), src_temp12);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)+16), src_temp12);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)+24), src_temp12);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)), src_temp13);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)+8), src_temp13);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)+16), src_temp13);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)+24), src_temp13);

                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)), src_temp14);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)+8), src_temp14);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)+16), src_temp14);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)+24), src_temp14);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)), src_temp15);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)+8), src_temp15);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)+16), src_temp15);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)+24), src_temp15);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)), src_temp16);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)+8), src_temp16);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)+16), src_temp16);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)+24), src_temp16);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)+8), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)+16), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)+24), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+7) * dst_strd)), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+7) * dst_strd)+8), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+7) * dst_strd)+16), src_temp17);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+7) * dst_strd)+24), src_temp17);
            }
        }
    }
}
/**
*******************************************************************************
*
* @brief
*     Intra prediction interpolation filter for vertical luma variable.
*
* @par Description:
*    Horizontal intraprediction with reference neighboring  samples location
*    pointed by 'pu2_ref' to the TU block  location pointed by 'pu2_dst'  Refer
*    to section 8.4.4.2.6 in the standard (Special case)
*
* @param[in] pu2_src
*  UWORD8 pointer to the source
*
* @param[out] pu2_dst
*  UWORD8 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
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
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_luma_ver_sse42(UWORD16 *pu2_ref,
                                         WORD32 src_strd,
                                         UWORD16 *pu2_dst,
                                         WORD32 dst_strd,
                                         WORD32 nt,
                                         WORD32 disable_boundary_filter,
                                         UWORD8 bit_depth)
{
    WORD32 row;
    WORD16 s2_predpixel;
    WORD32 two_nt = 2 * nt;
    __m128i src_temp0, src_temp1, src_temp2, src_temp3, src_temp4, src_temp5, src_temp6, src_temp7;
    __m128i src_temp8;

    if(nt == 32)
    {
        __m128i temp1, temp2, temp3, temp4;
        WORD32 itr_count;

        temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1));
        temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 + 8));
        temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 + 16));
        temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 + 24));

        for(itr_count = 0; itr_count < 4; itr_count++)
        {
            /* pu2_dst[(row * dst_strd) + col] = dc_val; */
            _mm_storeu_si128((__m128i *)(pu2_dst+((0) * dst_strd)), temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)), temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)), temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)), temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)), temp1);

            _mm_storeu_si128((__m128i *)(pu2_dst+8+((0)  * dst_strd)), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((1)  * dst_strd)), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((2)  * dst_strd)), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((3)  * dst_strd)), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((4)  * dst_strd)), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((5)  * dst_strd)), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((6)  * dst_strd)), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((7)  * dst_strd)), temp2);

            _mm_storeu_si128((__m128i *)(pu2_dst+16+((0)  * dst_strd)), temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((1)  * dst_strd)), temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((2)  * dst_strd)), temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((3)  * dst_strd)), temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((4)  * dst_strd)), temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((5)  * dst_strd)), temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((6)  * dst_strd)), temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+((7)  * dst_strd)), temp3);

            _mm_storeu_si128((__m128i *)(pu2_dst+24+((0)  * dst_strd)), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((1)  * dst_strd)), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((2)  * dst_strd)), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((3)  * dst_strd)), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((4)  * dst_strd)), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((5)  * dst_strd)), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((6)  * dst_strd)), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+((7)  * dst_strd)), temp4);

            pu2_dst += 8*dst_strd;
        }
    }
    else
    {
        /* Filtering done for the 1st column */
        if(disable_boundary_filter)
        {
            for(row = nt-1; row >=0; row--)
            {
                pu2_dst[row * dst_strd] = pu2_ref[two_nt + 1];
            }
        }
        else
        {
            for(row = nt-1; row >=0; row--)
            {
                s2_predpixel = pu2_ref[two_nt + 1]
                + ((pu2_ref[two_nt - 1 - row] - pu2_ref[two_nt]) >> 1);
                pu2_dst[row * dst_strd] = CLIP3(s2_predpixel, 0, ((1<<bit_depth)-1));
            }
        }

        /* Replication to next columns */

        if(nt ==4)
        {
            src_temp2 =   _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1 ));
            src_temp3 =  src_temp2;
            src_temp4 =  src_temp2;
            src_temp5 =  src_temp2;

            src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[(0 * dst_strd)], 0);
            src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[(1 * dst_strd)], 0);
            src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[(2 * dst_strd)], 0);
            src_temp5 = _mm_insert_epi16(src_temp5, pu2_dst[(3 * dst_strd)], 0);

            _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp2);
            _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp3);
            _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp4);
            _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp5);

        }
        else if(nt==8)
        {

            src_temp0 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1));
            src_temp1 = src_temp0;
            src_temp2 = src_temp0;
            src_temp3 = src_temp0;
            src_temp4 = src_temp0;
            src_temp5 = src_temp0;
            src_temp6 = src_temp0;
            src_temp7 = src_temp0;

            src_temp0 = _mm_insert_epi16(src_temp0, pu2_dst[((0)*dst_strd)], 0);
            src_temp1 = _mm_insert_epi16(src_temp1, pu2_dst[((1)*dst_strd)], 0);
            src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[((2)*dst_strd)], 0);
            src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[((3)*dst_strd)], 0);
            src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[((4)*dst_strd)], 0);
            src_temp5 = _mm_insert_epi16(src_temp5, pu2_dst[((5)*dst_strd)], 0);
            src_temp6 = _mm_insert_epi16(src_temp6, pu2_dst[((6)*dst_strd)], 0);
            src_temp7 = _mm_insert_epi16(src_temp7, pu2_dst[((7)*dst_strd)], 0);

            _mm_storeu_si128((__m128i *)(pu2_dst+((0) * dst_strd)), src_temp0);
            _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)), src_temp7);

        }
        else if(nt ==16)
        {
            src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1+8));
            for(row = 0; row < nt; row += 8)
            {

                src_temp0 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 1));
                src_temp1 = src_temp0;
                src_temp2 = src_temp0;
                src_temp3 = src_temp0;
                src_temp4 = src_temp0;
                src_temp5 = src_temp0;
                src_temp6 = src_temp0;
                src_temp7 = src_temp0;

                src_temp0 = _mm_insert_epi16(src_temp0, pu2_dst[((row+0)*dst_strd)], 0);
                src_temp1 = _mm_insert_epi16(src_temp1, pu2_dst[((row+1)*dst_strd)], 0);
                src_temp2 = _mm_insert_epi16(src_temp2, pu2_dst[((row+2)*dst_strd)], 0);
                src_temp3 = _mm_insert_epi16(src_temp3, pu2_dst[((row+3)*dst_strd)], 0);
                src_temp4 = _mm_insert_epi16(src_temp4, pu2_dst[((row+4)*dst_strd)], 0);
                src_temp5 = _mm_insert_epi16(src_temp5, pu2_dst[((row+5)*dst_strd)], 0);
                src_temp6 = _mm_insert_epi16(src_temp6, pu2_dst[((row+6)*dst_strd)], 0);
                src_temp7 = _mm_insert_epi16(src_temp7, pu2_dst[((row+7)*dst_strd)], 0);

                _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd)), src_temp0);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+7) * dst_strd)), src_temp7);

                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+0) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+1) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+2) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+3) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+4) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+5) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+6) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+7) * dst_strd)), src_temp8);
            }
        }
    }
}
/**
*******************************************************************************
*
* @brief
*     Intra prediction interpolation filter for horizontal luma variable.
*
* @par Description:
*      Horizontal intraprediction(mode 10) with reference  samples location
*      pointed by 'pu2_ref' to the TU block  location pointed by 'pu2_dst'  Refer
*      to section 8.4.4.2.6 in the standard (Special case)
*
* @param[in] pu2_src
*  UWORD8 pointer to the source
*
* @param[out] pu2_dst
*  UWORD8 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
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
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_luma_horz_sse42(UWORD16 *pu2_ref,
                                          WORD32 src_strd,
                                          UWORD16 *pu2_dst,
                                          WORD32 dst_strd,
                                          WORD32 nt,
                                          WORD32 disable_boundary_filter,
                                          UWORD8 bit_depth)
{
    WORD32 row;
    WORD32 two_nt;

    two_nt = 2 * nt;

    if(nt == 32)
    {
        __m128i src_temp1, src_temp2, src_temp3, src_temp4, src_temp5, src_temp6, src_temp7, src_temp8;
        __m128i sm = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASK4_HBD[0]);

        for(row = 0; row < nt; row += 8)
        {
            {
                src_temp1 =  _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 1-row-7));

                src_temp2 =  _mm_srli_si128(src_temp1, 2);
                src_temp3 =  _mm_srli_si128(src_temp1, 4);
                src_temp4 =  _mm_srli_si128(src_temp1, 6);
                src_temp5 =  _mm_srli_si128(src_temp1, 8);
                src_temp6 =  _mm_srli_si128(src_temp1, 10);
                src_temp7 =  _mm_srli_si128(src_temp1, 12);
                src_temp8 =  _mm_srli_si128(src_temp1, 14);

                src_temp8 =  _mm_shuffle_epi8(src_temp8, sm);
                src_temp7 =  _mm_shuffle_epi8(src_temp7, sm);
                src_temp6 =  _mm_shuffle_epi8(src_temp6, sm);
                src_temp5 =  _mm_shuffle_epi8(src_temp5, sm);
                src_temp4 =  _mm_shuffle_epi8(src_temp4, sm);
                src_temp3 =  _mm_shuffle_epi8(src_temp3, sm);
                src_temp2 =  _mm_shuffle_epi8(src_temp2, sm);
                src_temp1 =  _mm_shuffle_epi8(src_temp1, sm);

                _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+4) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+5) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+6) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+((row+7) * dst_strd)), src_temp1);

                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+0) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+1) * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+2) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+3) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+4) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+5) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+6) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+8+((row+7) * dst_strd)), src_temp1);

                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+0) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+1) * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+2) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+3) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+4) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+5) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+6) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+16+((row+7) * dst_strd)), src_temp1);

                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+0) * dst_strd)), src_temp8);
                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+1) * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+2) * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+3) * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+4) * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+5) * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+6) * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+24+((row+7) * dst_strd)), src_temp1);
            }
        }
    }
    else
    {
        __m128i src_temp1, src_temp2, src_temp3, src_temp4, src_temp5, src_temp6;
        __m128i src_temp10, zero_8x16b, mask_4x32b, src_temp7;

        /* DC filtering for the first top row and first left column */

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        /* Filtering done for the 1st row */
        if(!disable_boundary_filter)
        {
            src_temp2 =  _mm_set1_epi16(pu2_ref[two_nt - 1 ]);
            src_temp10 =  _mm_set1_epi16(pu2_ref[two_nt]);

            /* loding 8-bit 16 pixels */
            src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+1));

            /* (pu2_ref[two_nt + 1 + col] - pu2_ref[two_nt]) */
            src_temp3 = _mm_sub_epi16(src_temp4, src_temp10);

            /* ((pu2_ref[two_nt + 1 + col] - pu2_ref[two_nt]) >> 1) */
            src_temp3 = _mm_srai_epi16(src_temp3, 1);

            /* pu2_ref[two_nt - 1]+((pu2_ref[two_nt + 1 + col] - pu2_ref[two_nt]) >> 1) */
            src_temp3 = _mm_add_epi16(src_temp2, src_temp3);
        }

        if(nt==4)
        {
            if(disable_boundary_filter)
            {
                src_temp3 = _mm_set1_epi16(pu2_ref[two_nt - 1]);
            }
            else
            {
                __m128i clip_tmp_8x16b;
                clip_tmp_8x16b = _mm_cmpeq_epi16(src_temp3, src_temp3);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                src_temp3 = _mm_min_epi16(src_temp3, clip_tmp_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                src_temp3 = _mm_max_epi16(src_temp3, clip_tmp_8x16b);
            }

            _mm_storel_epi64((__m128i *)pu2_dst, src_temp3);

            src_temp2 =  _mm_set1_epi16(pu2_ref[two_nt - 2 ]);
            src_temp3 =  _mm_set1_epi16(pu2_ref[two_nt - 3 ]);
            src_temp4 =  _mm_set1_epi16(pu2_ref[two_nt - 4]);

            _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp2);
            _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp3);
            _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp4);
        }
        else if(nt==8)
        {
            if(disable_boundary_filter)
            {
                src_temp10 = _mm_set1_epi16(pu2_ref[two_nt - 1]);
            }
            else
            {
                __m128i clip_tmp_8x16b;
                clip_tmp_8x16b = _mm_cmpeq_epi16(src_temp3, src_temp3);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                src_temp3 = _mm_min_epi16(src_temp3, clip_tmp_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                src_temp10 = _mm_max_epi16(src_temp3, clip_tmp_8x16b);
            }

            src_temp1 =  _mm_set1_epi16(pu2_ref[two_nt - 2 ]);
            src_temp2 =  _mm_set1_epi16(pu2_ref[two_nt - 3 ]);
            src_temp3 =  _mm_set1_epi16(pu2_ref[two_nt - 4 ]);
            src_temp4 =  _mm_set1_epi16(pu2_ref[two_nt - 5 ]);
            src_temp5 =  _mm_set1_epi16(pu2_ref[two_nt - 6 ]);
            src_temp6 =  _mm_set1_epi16(pu2_ref[two_nt - 7 ]);
            src_temp7 =  _mm_set1_epi16(pu2_ref[two_nt - 8 ]);

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp10);

            _mm_storeu_si128((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+(4 * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+(5 * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+(6 * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+(7 * dst_strd)), src_temp7);
        }
        else if(nt==16)
        {
            if(disable_boundary_filter)
            {
                src_temp3 = _mm_set1_epi16(pu2_ref[two_nt - 1]);
                src_temp10 = src_temp3;
            }
            else
            {
                __m128i clip_tmp_8x16b;
                clip_tmp_8x16b = _mm_cmpeq_epi16(src_temp3, src_temp3);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+1+8));

                src_temp10 = _mm_sub_epi16(src_temp4, src_temp10);
                src_temp10 = _mm_srai_epi16(src_temp10, 1);
                src_temp10 = _mm_add_epi16(src_temp2, src_temp10);

                src_temp3 = _mm_min_epi16(src_temp3, clip_tmp_8x16b);
                src_temp10 = _mm_min_epi16(src_temp10, clip_tmp_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                src_temp3 = _mm_max_epi16(src_temp3, clip_tmp_8x16b);
                src_temp10 = _mm_max_epi16(src_temp10, clip_tmp_8x16b);
            }

            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp10);

            src_temp1 =  _mm_set1_epi16(pu2_ref[two_nt - 2]);
            src_temp2 =  _mm_set1_epi16(pu2_ref[two_nt - 3]);
            src_temp3 =  _mm_set1_epi16(pu2_ref[two_nt - 4]);
            src_temp4 =  _mm_set1_epi16(pu2_ref[two_nt - 5]);
            src_temp5 =  _mm_set1_epi16(pu2_ref[two_nt - 6]);
            src_temp6 =  _mm_set1_epi16(pu2_ref[two_nt - 7]);
            src_temp7 =  _mm_set1_epi16(pu2_ref[two_nt - 8]);
            src_temp10 =  _mm_set1_epi16(pu2_ref[two_nt -9]);

            _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)), src_temp7);
            _mm_storeu_si128((__m128i *)(pu2_dst+((8) * dst_strd)), src_temp10);

            _mm_storeu_si128((__m128i *)(pu2_dst+8+((1) * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((2) * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((3) * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((4) * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((5) * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((6) * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((7) * dst_strd)), src_temp7);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((8) * dst_strd)), src_temp10);

            src_temp1 =  _mm_set1_epi16(pu2_ref[two_nt - 10]);
            src_temp2 =  _mm_set1_epi16(pu2_ref[two_nt - 11 ]);
            src_temp3 =  _mm_set1_epi16(pu2_ref[two_nt - 12 ]);
            src_temp4 =  _mm_set1_epi16(pu2_ref[two_nt - 13 ]);
            src_temp5 =  _mm_set1_epi16(pu2_ref[two_nt - 14 ]);
            src_temp6 =  _mm_set1_epi16(pu2_ref[two_nt - 15 ]);
            src_temp7 =  _mm_set1_epi16(pu2_ref[two_nt - 16 ]);

            _mm_storeu_si128((__m128i *)(pu2_dst+((9)  * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+((10) * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((11) * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+((12) * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((13) * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+((14) * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((15) * dst_strd)), src_temp7);

            _mm_storeu_si128((__m128i *)(pu2_dst+8+((9)  * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((10) * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((11) * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((12) * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((13) * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((14) * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+((15) * dst_strd)), src_temp7);
        }
    }
}
/**
*******************************************************************************
*
* @brief
*     Intra prediction interpolation filter for luma mode2.
*
* @par Description:
*    Intraprediction for mode 2 (sw angle) with reference  neighboring samples
*    location pointed by 'pu2_ref' to the  TU block location pointed by
*    'pu2_dst'  Refer to section 8.4.4.2.6 in the standard
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
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_luma_mode2_sse42(UWORD16 *pu2_ref,
                                           WORD32 src_strd,
                                           UWORD16 *pu2_dst,
                                           WORD32 dst_strd,
                                           WORD32 nt,
                                           WORD32 mode,
                                           UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 two_nt= 2 * nt;

    __m128i src_temp1, src_temp2, src_temp3, src_temp4, src_temp5, src_temp6, src_temp7, src_temp8;
    __m128i  mask_4x32b, sm1, sm2, sm3;

    mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

    sm1 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY1_HBD[0]);
    sm2 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY2_HBD[0]);
    sm3 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY3_HBD[0]);

    /* For the angle 45, replication is done from the corresponding angle */
    /* intra_pred_ang = tan(angle) in q5 format */

    if(nt==4)
    {
        /* pu2_ref[two_nt - row - (col+1) - 1] */
        src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -8));
        src_temp2 = _mm_srli_si128(src_temp1, 2);
        src_temp3 = _mm_srli_si128(src_temp1, 4);
        src_temp4 = _mm_srli_si128(src_temp1, 6);

        src_temp4= _mm_shuffle_epi8(src_temp4, sm1);
        src_temp3= _mm_shuffle_epi8(src_temp3, sm1);
        src_temp2= _mm_shuffle_epi8(src_temp2, sm1);
        src_temp1= _mm_shuffle_epi8(src_temp1, sm1);

        /* pu2_dst[(row * dst_strd) + col] = pu2_ref[two_nt - 1 - row]; */
        _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp4);
        _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp3);
        _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp2);
        _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp1);
    }
    else if(nt==8)
    {
        /* pu2_ref[two_nt - row - (col+1) - 1] */
        src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 16));
        src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 15));
        src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 14));
        src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 13));
        src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 12));
        src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 11));
        src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 10));
        src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 9));

        src_temp1 = _mm_shuffle_epi8(src_temp1, sm2);
        src_temp2 = _mm_shuffle_epi8(src_temp2, sm2);
        src_temp3 = _mm_shuffle_epi8(src_temp3, sm2);
        src_temp4 = _mm_shuffle_epi8(src_temp4, sm2);
        src_temp5 = _mm_shuffle_epi8(src_temp5, sm2);
        src_temp6 = _mm_shuffle_epi8(src_temp6, sm2);
        src_temp7 = _mm_shuffle_epi8(src_temp7, sm2);
        src_temp8 = _mm_shuffle_epi8(src_temp8, sm2);

        _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), src_temp8);
        _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), src_temp7 );
        _mm_storeu_si128((__m128i *)(pu2_dst + (2 * dst_strd)), src_temp6 );
        _mm_storeu_si128((__m128i *)(pu2_dst + (3 * dst_strd)), src_temp5 );
        _mm_storeu_si128((__m128i *)(pu2_dst + (4 * dst_strd)), src_temp4 );
        _mm_storeu_si128((__m128i *)(pu2_dst + (5 * dst_strd)), src_temp3 );
        _mm_storeu_si128((__m128i *)(pu2_dst + (6 * dst_strd)), src_temp2 );
        _mm_storeu_si128((__m128i *)(pu2_dst + (7 * dst_strd)), src_temp1 );
    }
    else
    {
        for(row = 0; row < nt; row += 8)
        {
            for(col = 0; col <nt; col += 8)
            {
                /* pu2_ref[two_nt - row - (col+1) - 1] */
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+0)- (col+8)-1));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+1)- (col+8)-1));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+2)- (col+8)-1));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+3)- (col+8)-1));
                src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+4)- (col+8)-1));
                src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+5)- (col+8)-1));
                src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+6)- (col+8)-1));
                src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - (row+7)- (col+8)-1));

                src_temp1 = _mm_shuffle_epi8(src_temp1, sm3);
                src_temp2 = _mm_shuffle_epi8(src_temp2, sm3);
                src_temp3 = _mm_shuffle_epi8(src_temp3, sm3);
                src_temp4 = _mm_shuffle_epi8(src_temp4, sm3);
                src_temp5 = _mm_shuffle_epi8(src_temp5, sm3);
                src_temp6 = _mm_shuffle_epi8(src_temp6, sm3);
                src_temp7 = _mm_shuffle_epi8(src_temp7, sm3);
                src_temp8 = _mm_shuffle_epi8(src_temp8, sm3);

                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+0) * dst_strd)), src_temp1 );
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+1) * dst_strd)), src_temp2 );
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+2) * dst_strd)), src_temp3 );
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+3) * dst_strd)), src_temp4 );
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+4) * dst_strd)), src_temp5 );
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+5) * dst_strd)), src_temp6 );
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+6) * dst_strd)), src_temp7 );
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+7) * dst_strd)), src_temp8 );
            }
        }
    }
}
/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma mode 18 & mode 34.
*
* @par Description:
*    Intraprediction for mode 34 (ne angle) and  mode 18 (nw angle) with
*    reference  neighboring samples location pointed by 'pu2_ref' to the  TU
*    block location pointed by 'pu2_dst'
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
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_luma_mode_18_34_sse42(UWORD16 *pu2_ref,
                                                WORD32 src_strd,
                                                UWORD16 *pu2_dst,
                                                WORD32 dst_strd,
                                                WORD32 nt,
                                                WORD32 mode,
                                                UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 two_nt = 2 * nt;
    __m128i src_temp1, src_temp2, src_temp3, src_temp4, src_temp5, src_temp6, src_temp7, src_temp8;

    if(mode == 34)
    {
        if(nt==4)
        {
            /* pu2_ref[two_nt + col + idx + 1] */
            src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 2));
            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 3));
            src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 4));
            src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 5));

            /* pu2_dst[(row * dst_strd) + col] = pu2_ref[two_nt - 1 - row]; */

            _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp1);
            _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp2);
            _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp3);
            _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp4);
        }
        else if(nt==8)
        {
            /* pu2_ref[two_nt + col + idx + 1] */
            src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 2));
            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 3));
            src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 4));
            src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 5));
            src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 6));
            src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 7));
            src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 8));
            src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + 9));

            _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst + (2 * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst + (3 * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst + (4 * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst + (5 * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst + (6 * dst_strd)), src_temp7);
            _mm_storeu_si128((__m128i *)(pu2_dst + (7 * dst_strd)), src_temp8);

        }
        else if(nt ==16)
        {
            for(row = 0; row < nt; row += 8)
            {
                for(col = 0; col < nt; col += 8)
                {
                    /* pu2_ref[two_nt + col + idx + 1] */
                    src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+0) + col + 2));
                    src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+1) + col + 2));
                    src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+2) + col + 2));
                    src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+3) + col + 2));
                    src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+4) + col + 2));
                    src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+5) + col + 2));
                    src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+6) + col + 2));
                    src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (row+7) + col + 2));

                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ ((row+0) * dst_strd)), src_temp1);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ ((row+1) * dst_strd)), src_temp2);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+2) * dst_strd)), src_temp3);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+3) * dst_strd)), src_temp4);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+4) * dst_strd)), src_temp5);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+5) * dst_strd)), src_temp6);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+6) * dst_strd)), src_temp7);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + ((row+7) * dst_strd)), src_temp8);
                }
            }
        }
        else
        {
            __m128i src_temp9, src_temp10, src_temp11, src_temp12, src_temp13, src_temp14, src_temp15, src_temp16;
            for(row = 0; row < nt; row += 8)
            {
                for(col = 0; col < nt; col += 16)
                {
                    /* pu2_ref[two_nt + col + idx + 1] */
                    src_temp1  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (0+ col) + 2));
                    src_temp9  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (0+ col + 8) + 2));
                    src_temp2  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (1+ col) + 2));
                    src_temp10 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (1+ col + 8) + 2));
                    src_temp3  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (2+ col) + 2));
                    src_temp11 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (2+ col + 8) + 2));
                    src_temp4  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (3+ col) + 2));
                    src_temp12 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (3+ col + 8) + 2));

                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (0 * dst_strd)), src_temp1 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + 8 + (0 * dst_strd)), src_temp9 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (1 * dst_strd)), src_temp2 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col +8 + (1 * dst_strd)), src_temp10);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (2 * dst_strd)), src_temp3 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col +8 + (2 * dst_strd)), src_temp11);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (3 * dst_strd)), src_temp4 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col +8 + (3 * dst_strd)), src_temp12);

                    src_temp5  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (4+ col) + 2));
                    src_temp13 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (4+col+8) + 2));
                    src_temp6  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (5+ col) + 2));
                    src_temp14 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (5+col+8) + 2));
                    src_temp7  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (6+ col) + 2));
                    src_temp15 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (6+col+8) + 2));
                    src_temp8  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (7+ col) + 2));
                    src_temp16 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt + (7+col+8) + 2));

                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (4 * dst_strd)), src_temp5 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + 8 + (4 * dst_strd)), src_temp13);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col  + (5 * dst_strd)), src_temp6 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + 8 + (5 * dst_strd)), src_temp14);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col  + (6 * dst_strd)), src_temp7 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + 8 + (6 * dst_strd)), src_temp15);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col  + (7 * dst_strd)), src_temp8 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + 8 + (7 * dst_strd)), src_temp16);
                }

                pu2_ref += 8;
                pu2_dst += 8*dst_strd;
            }
        }
    }
    else
    {
        if(nt==4)
        {
            /* pu2_ref[two_nt + col + idx + 1] */
            src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 3 ));
            src_temp2 = _mm_srli_si128(src_temp1, 2);
            src_temp3 = _mm_srli_si128(src_temp1, 4);
            src_temp4 = _mm_srli_si128(src_temp1, 6);

            /* pu2_dst[(row * dst_strd) + col] = pu2_ref[two_nt - 1 - row]; */
            _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp4);
            _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp3);
            _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp2);
            _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp1);
        }
        else if(nt==8)
        {
            /* pu2_ref[two_nt + col + idx + 1] */
            src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 7 ));
            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 6 ));
            src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 5 ));
            src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 4 ));
            src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 3 ));
            src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 2 ));
            src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 1 ));
            src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt - 0 ));

            _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), src_temp8);
            _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), src_temp7);
            _mm_storeu_si128((__m128i *)(pu2_dst + (2 * dst_strd)), src_temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst + (3 * dst_strd)), src_temp5);
            _mm_storeu_si128((__m128i *)(pu2_dst + (4 * dst_strd)), src_temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst + (5 * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst + (6 * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst + (7 * dst_strd)), src_temp1);
        }
        else if(nt==16)
        {
            for(row = 0; row < nt; row += 8)
            {
                for(col = 0; col < nt; col += 8)
                {
                    /* pu2_ref[two_nt + col + idx + 1] */
                    src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+0) ));
                    src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+1) ));
                    src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+2) ));
                    src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+3) ));
                    src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+4) ));
                    src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+5) ));
                    src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+6) ));
                    src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt+col- (row+7) ));

                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+0) * dst_strd)), src_temp1);
                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+1) * dst_strd)), src_temp2);
                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+2) * dst_strd)), src_temp3);
                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+3) * dst_strd)), src_temp4);
                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+4) * dst_strd)), src_temp5);
                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+5) * dst_strd)), src_temp6);
                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+6) * dst_strd)), src_temp7);
                    _mm_storeu_si128((__m128i *)(pu2_dst +col+ ( (row+7) * dst_strd)), src_temp8);
                }
            }
        }
        else
        {
            __m128i src_temp9, src_temp10, src_temp11, src_temp12, src_temp13, src_temp14, src_temp15, src_temp16;
            for(row = 0; row < nt; row += 8)
            {
                for(col = 0; col < nt; col += 16)
                {
                    /* pu2_ref[two_nt + col + idx + 1] */
                    src_temp1  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -0+ col));
                    src_temp9  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -0+ col+8));
                    src_temp2  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -1+ col));
                    src_temp10 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -1+col+8));
                    src_temp3  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -2+ col));
                    src_temp11 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -2+col+8));
                    src_temp4  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -3+ col));
                    src_temp12 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -3+col+8));

                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (0 * dst_strd)), src_temp1 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (0 * dst_strd)), src_temp9 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (1 * dst_strd)), src_temp2 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (1 * dst_strd)), src_temp10);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (2 * dst_strd)), src_temp3 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (2 * dst_strd)), src_temp11);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (3 * dst_strd)), src_temp4 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (3 * dst_strd)), src_temp12);

                    src_temp5  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -4+ col));
                    src_temp13 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -4+col+8));
                    src_temp6  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -5+ col));
                    src_temp14 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -5+col+8));
                    src_temp7  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -6+ col));
                    src_temp15 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -6+col+8));
                    src_temp8  = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -7+ col));
                    src_temp16 = _mm_loadu_si128((__m128i *)(pu2_ref+two_nt -7+col+8));

                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (4 * dst_strd)), src_temp5 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (4 * dst_strd)), src_temp13);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (5 * dst_strd)), src_temp6 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (5 * dst_strd)), src_temp14);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (6 * dst_strd)), src_temp7 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (6 * dst_strd)), src_temp15);
                    _mm_storeu_si128((__m128i *)(pu2_dst + col + (7 * dst_strd)), src_temp8 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + col+ 8 + (7 * dst_strd)), src_temp16);
                }

                pu2_ref -= 8;
                pu2_dst += 8*dst_strd;
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma mode 3 to mode 9
*
* @par Description:
*    Intraprediction for mode 3 to 9  (positive angle, horizontal mode ) with
*    reference  neighboring samples location pointed by 'pu2_ref' to the  TU
*    block location pointed by 'pu2_dst'
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
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_luma_mode_3_to_9_sse42(UWORD16 *pu2_ref,
                                                 WORD32 src_strd,
                                                 UWORD16 *pu2_dst,
                                                 WORD32 dst_strd,
                                                 WORD32 nt,
                                                 WORD32 mode,
                                                 UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 two_nt = 2 * nt;
    WORD32 intra_pred_ang;

    __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
    __m128i fract_4x32b, zero_8x16b, intra_pred_ang_4x32b;
    __m128i row_4x32b, two_nt_4x32b, ref_main_idx_4x32b, res_temp5_4x32b, sm1, sm2;

    /* Intra Pred Angle according to the mode */
    intra_pred_ang = gai4_ihevc_ang_table[mode];

    /* For the angles other then 45 degree, interpolation btw 2 neighboring */
    /* samples dependent on distance to obtain destination sample */

    /* For the angles other then 45 degree, interpolation btw 2 neighboring */
    /* samples dependent on distance to obtain destination sample */

    const_temp_4x32b  = _mm_set1_epi32(16);
    const_temp2_4x32b = _mm_set1_epi32(31);
    const_temp3_4x32b = _mm_set1_epi32(32);
    const_temp4_4x32b = _mm_set1_epi32(4);

    two_nt_4x32b = _mm_set1_epi32(two_nt-nt);

    zero_8x16b = _mm_set1_epi16(0);
    mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

    sm1 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY1_HBD[0]);
    sm2 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY2_HBD[0]);

    /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
    intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

    row_4x32b = _mm_set_epi32(4, 3, 2, 1);

    if(nt==4)
    {
        WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;
        WORD8  ai1_fract_temp_val[16], ai1_row_temp_val[16];

        __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
        __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;

        __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
        __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;
        __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;

        /* pos = ((row + 1) * intra_pred_ang); */
        res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

        /* idx = pos >> 5; */
        fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

        /* fract = pos & (31); */
        ref_main_idx_4x32b = _mm_sub_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));

        /* (32 - fract) */
        row_4x32b = _mm_sub_epi32(const_temp3_4x32b , fract_4x32b);

        _mm_storeu_si128((__m128i *)(ai1_fract_temp_val), fract_4x32b);
        _mm_storeu_si128((__m128i *)(ai1_row_temp_val),  row_4x32b);

        fract1_8x16b = _mm_set1_epi16(ai1_fract_temp_val[0]);  /* col=0 */
        fract2_8x16b = _mm_set1_epi16(ai1_fract_temp_val[4]);  /* col=1 */
        fract3_8x16b = _mm_set1_epi16(ai1_fract_temp_val[8]);  /* col=2 */
        fract4_8x16b = _mm_set1_epi16(ai1_fract_temp_val[12]);  /* col=3 */

        temp1_8x16b = _mm_set1_epi16(ai1_row_temp_val[0]);  /* col=0 */
        temp2_8x16b = _mm_set1_epi16(ai1_row_temp_val[4]);  /* col=1 */
        temp3_8x16b = _mm_set1_epi16(ai1_row_temp_val[8]);  /* col=2 */
        temp4_8x16b = _mm_set1_epi16(ai1_row_temp_val[12]);  /* col=3 */

        temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
        temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
        temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
        temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

        ref_main_temp0 = _mm_srli_si128(ref_main_idx_4x32b , 4);  /* next 32 bit values */
        ref_main_temp1 = _mm_srli_si128(ref_main_idx_4x32b , 8);  /* next 32 bit values */
        ref_main_temp2 = _mm_srli_si128(ref_main_idx_4x32b , 12); /* next 32 bit values */
        ref_main_idx1  = _mm_cvtsi128_si32(ref_main_idx_4x32b);    /* col=0 */
        ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* col=1 */
        ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* col=2 */
        ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* col=3 */

        /* loding 8-bit 16 pixels */
        src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1-1)); /* col=0 */
        src_temp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx2-1)); /* col=1 */
        src_temp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx3-1)); /* col=2 */
        src_temp4_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx4-1)); /* col=3 */

        src_temp5_8x16b = _mm_srli_si128(src_temp1_8x16b, 2); /* col=0 */
        src_temp6_8x16b = _mm_srli_si128(src_temp2_8x16b, 2); /* col=1 */
        src_temp7_8x16b = _mm_srli_si128(src_temp3_8x16b, 2); /* col=2 */
        src_temp8_8x16b = _mm_srli_si128(src_temp4_8x16b, 2); /* col=3 */

        src_temp1_8x16b =  _mm_unpacklo_epi16(src_temp5_8x16b, src_temp1_8x16b); /* col=0 */
        src_temp2_8x16b =  _mm_unpacklo_epi16(src_temp6_8x16b, src_temp2_8x16b); /* col=1 */
        src_temp3_8x16b =  _mm_unpacklo_epi16(src_temp7_8x16b, src_temp3_8x16b); /* col=2 */
        src_temp4_8x16b =  _mm_unpacklo_epi16(src_temp8_8x16b, src_temp4_8x16b); /* col=3 */

        /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
        src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
        src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
        src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
        src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

        /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
        src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
        src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
        src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
        src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

        /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
        src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
        src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
        src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
        src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

        /* converting 16 bit to 8 bit */
        src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, zero_8x16b); /* col=0 */
        src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, zero_8x16b); /* col=1 */
        src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, zero_8x16b); /* col=2 */
        src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, zero_8x16b); /* col=3 */

        src_temp1_8x16b =_mm_shuffle_epi8(src_temp1_8x16b, sm1);
        src_temp2_8x16b =_mm_shuffle_epi8(src_temp2_8x16b, sm1);
        src_temp3_8x16b =_mm_shuffle_epi8(src_temp3_8x16b, sm1);
        src_temp4_8x16b =_mm_shuffle_epi8(src_temp4_8x16b, sm1);

        src_temp5_8x16b = _mm_unpacklo_epi16(src_temp1_8x16b, src_temp2_8x16b);
        src_temp6_8x16b = _mm_unpacklo_epi16(src_temp3_8x16b, src_temp4_8x16b);

        src_temp7_8x16b = _mm_unpacklo_epi32(src_temp5_8x16b, src_temp6_8x16b);
        src_temp1_8x16b  = _mm_shuffle_epi32(src_temp7_8x16b, _MM_SHUFFLE(3, 2, 3, 2));
        src_temp2_8x16b  = _mm_unpackhi_epi32(src_temp5_8x16b, src_temp6_8x16b);
        src_temp3_8x16b  = _mm_shuffle_epi32(src_temp2_8x16b, _MM_SHUFFLE(3, 2, 3, 2));

        _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp7_8x16b);
        _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp1_8x16b);
        _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp2_8x16b);
        _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp3_8x16b);

    }
    else if(nt==16 || nt==32)
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        const_temp4_4x32b = _mm_set1_epi16(8);
        const_temp3_4x32b = _mm_set1_epi16(32);
        two_nt_4x32b = _mm_set1_epi16(two_nt);

        for(col = 0; col < nt; col += 8)
        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD16 pi2_ref_main_idx5, pi2_ref_main_idx6, pi2_ref_main_idx7, pi2_ref_main_idx8;
            WORD8  ai1_fract_temp0_val[16], ai1_fract_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i fract5_8x16b, fract6_8x16b, fract7_8x16b, fract8_8x16b;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i temp11_8x16b, temp12_8x16b, temp13_8x16b, temp14_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* idx = pos >> 5; */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* fract = pos & (31); */
            ref_main_idx_4x32b = _mm_sub_epi16(two_nt_4x32b , _mm_srai_epi16(res_temp5_4x32b,  5));

            row_4x32b= _mm_add_epi16(row_4x32b, const_temp4_4x32b);

            _mm_storeu_si128((__m128i *)(ai1_fract_temp0_val), fract_4x32b);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[6]);  /* col=3 */

            fract5_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[8]);  /* col=5 */
            fract6_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[10]);  /* col=6 */
            fract7_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[12]);  /* col=7 */
            fract8_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[14]);  /* col=8 */

            /* (32 - fract) */
            fract_4x32b = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);
            _mm_storeu_si128((__m128i *)(ai1_fract_temp1_val), fract_4x32b);

            temp1_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            temp11_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[8]);  /* col=0 */
            temp12_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[10]);  /* col=1 */
            temp13_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[12]);  /* col=2 */
            temp14_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[14]);  /* col=3 */

            temp11_8x16b = _mm_unpacklo_epi16(temp11_8x16b, fract5_8x16b);
            temp12_8x16b = _mm_unpacklo_epi16(temp12_8x16b, fract6_8x16b);
            temp13_8x16b = _mm_unpacklo_epi16(temp13_8x16b, fract7_8x16b);
            temp14_8x16b = _mm_unpacklo_epi16(temp14_8x16b, fract8_8x16b);

            pi2_ref_main_idx5 = _mm_extract_epi16(ref_main_idx_4x32b, 4);    /* col=5 */
            pi2_ref_main_idx6 = _mm_extract_epi16(ref_main_idx_4x32b, 5);    /* col=6 */
            pi2_ref_main_idx7 = _mm_extract_epi16(ref_main_idx_4x32b, 6);    /* col=7 */
            pi2_ref_main_idx8 = _mm_extract_epi16(ref_main_idx_4x32b, 7);    /* col=8 */

            for(row = 0; row < nt; row += 8)
            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;
                __m128i src_temp11_8x16b, src_temp12_8x16b, src_temp13_8x16b, src_temp14_8x16b;
                __m128i src_temp15_8x16b, src_temp16_8x16b, src_temp17_8x16b, src_temp18_8x16b;
                __m128i src_tmp1_8x16b, src_tmp2_8x16b, src_tmp3_8x16b, src_tmp4_8x16b;
                __m128i src_tmp5_8x16b, src_tmp6_8x16b, src_tmp7_8x16b, src_tmp8_8x16b;

                /* loding 8-bit 16 pixels */
                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1-1-(8+row))); /* col=0 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2-1-(8+row))); /* col=1 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3-1-(8+row))); /* col=2 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4-1-(8+row))); /* col=3 */

                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1-(8+row))); /* col=0 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2-(8+row))); /* col=1 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3-(8+row))); /* col=2 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4-(8+row))); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* loding 8-bit 16 pixels */
                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx5-1-(8+row))); /* col=5 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx6-1-(8+row))); /* col=6 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx7-1-(8+row))); /* col=7 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx8-1-(8+row))); /* col=8 */

                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx5-(8+row))); /* col=5 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx6-(8+row))); /* col=6 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx7-(8+row))); /* col=7 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx8-(8+row))); /* col=8 */

                src_temp11_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=5 */
                src_temp12_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=6 */
                src_temp13_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=7 */
                src_temp14_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=8 */

                src_temp15_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=5 */
                src_temp16_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=6 */
                src_temp17_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=7 */
                src_temp18_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=8 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp11_8x16b = _mm_madd_epi16(src_temp11_8x16b, temp11_8x16b);
                src_temp12_8x16b = _mm_madd_epi16(src_temp12_8x16b, temp12_8x16b);
                src_temp13_8x16b = _mm_madd_epi16(src_temp13_8x16b, temp13_8x16b);
                src_temp14_8x16b = _mm_madd_epi16(src_temp14_8x16b, temp14_8x16b);

                src_temp15_8x16b = _mm_madd_epi16(src_temp15_8x16b, temp11_8x16b);
                src_temp16_8x16b = _mm_madd_epi16(src_temp16_8x16b, temp12_8x16b);
                src_temp17_8x16b = _mm_madd_epi16(src_temp17_8x16b, temp13_8x16b);
                src_temp18_8x16b = _mm_madd_epi16(src_temp18_8x16b, temp14_8x16b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);   /* col=0 */
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);   /* col=1 */
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);   /* col=2 */
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);   /* col=3 */

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp11_8x16b = _mm_add_epi32(src_temp11_8x16b, const_temp_4x32b);
                src_temp12_8x16b = _mm_add_epi32(src_temp12_8x16b, const_temp_4x32b);
                src_temp13_8x16b = _mm_add_epi32(src_temp13_8x16b, const_temp_4x32b);
                src_temp14_8x16b = _mm_add_epi32(src_temp14_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp11_8x16b = _mm_srai_epi32(src_temp11_8x16b,  5);   /* col=5 */
                src_temp12_8x16b = _mm_srai_epi32(src_temp12_8x16b,  5);   /* col=6 */
                src_temp13_8x16b = _mm_srai_epi32(src_temp13_8x16b,  5);   /* col=7 */
                src_temp14_8x16b = _mm_srai_epi32(src_temp14_8x16b,  5);   /* col=8 */

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp15_8x16b = _mm_add_epi32(src_temp15_8x16b, const_temp_4x32b);
                src_temp16_8x16b = _mm_add_epi32(src_temp16_8x16b, const_temp_4x32b);
                src_temp17_8x16b = _mm_add_epi32(src_temp17_8x16b, const_temp_4x32b);
                src_temp18_8x16b = _mm_add_epi32(src_temp18_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp15_8x16b = _mm_srai_epi32(src_temp15_8x16b,  5);   /* col=5 */
                src_temp16_8x16b = _mm_srai_epi32(src_temp16_8x16b,  5);   /* col=6 */
                src_temp17_8x16b = _mm_srai_epi32(src_temp17_8x16b,  5);   /* col=7 */
                src_temp18_8x16b = _mm_srai_epi32(src_temp18_8x16b,  5);   /* col=8 */

                /* converting 32 bit to 16 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                /* converting 32 bit to 16 bit */
                src_temp11_8x16b = _mm_packus_epi32(src_temp11_8x16b, src_temp15_8x16b); /* col=5 */
                src_temp12_8x16b = _mm_packus_epi32(src_temp12_8x16b, src_temp16_8x16b); /* col=6 */
                src_temp13_8x16b = _mm_packus_epi32(src_temp13_8x16b, src_temp17_8x16b); /* col=7 */
                src_temp14_8x16b = _mm_packus_epi32(src_temp14_8x16b, src_temp18_8x16b); /* col=8 */

                src_temp1_8x16b =_mm_shuffle_epi8(src_temp1_8x16b, sm2);
                src_temp2_8x16b =_mm_shuffle_epi8(src_temp2_8x16b, sm2);
                src_temp3_8x16b =_mm_shuffle_epi8(src_temp3_8x16b, sm2);
                src_temp4_8x16b =_mm_shuffle_epi8(src_temp4_8x16b, sm2);

                src_temp11_8x16b =_mm_shuffle_epi8(src_temp11_8x16b, sm2);
                src_temp12_8x16b =_mm_shuffle_epi8(src_temp12_8x16b, sm2);
                src_temp13_8x16b =_mm_shuffle_epi8(src_temp13_8x16b, sm2);
                src_temp14_8x16b =_mm_shuffle_epi8(src_temp14_8x16b, sm2);

                src_temp5_8x16b = _mm_unpacklo_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi16(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi16(src_temp3_8x16b, src_temp4_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp16_8x16b = _mm_unpacklo_epi16(src_temp13_8x16b, src_temp14_8x16b);
                src_temp17_8x16b = _mm_unpackhi_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi16(src_temp13_8x16b, src_temp14_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi32(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi32(src_temp7_8x16b, src_temp8_8x16b);

                src_temp11_8x16b = _mm_unpacklo_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp12_8x16b = _mm_unpackhi_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp13_8x16b = _mm_unpacklo_epi32(src_temp17_8x16b, src_temp18_8x16b);
                src_temp14_8x16b = _mm_unpackhi_epi32(src_temp17_8x16b, src_temp18_8x16b);

                src_temp5_8x16b = _mm_unpacklo_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp6_8x16b = _mm_unpackhi_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp7_8x16b = _mm_unpacklo_epi64(src_temp2_8x16b, src_temp12_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi64(src_temp2_8x16b, src_temp12_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp16_8x16b = _mm_unpackhi_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp17_8x16b = _mm_unpacklo_epi64(src_temp4_8x16b, src_temp14_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi64(src_temp4_8x16b, src_temp14_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * row)), src_temp5_8x16b);          /* row=0 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+1))), src_temp6_8x16b);       /* row=1 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+2))), src_temp7_8x16b);       /* row=2 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+3))), src_temp8_8x16b);       /* row=3 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+4))), src_temp15_8x16b);       /* row=4 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+5))), src_temp16_8x16b);       /* row=5 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+6))), src_temp17_8x16b);       /* row=6 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+7))), src_temp18_8x16b);       /* row=7 */
            }
        }
    }
    else
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        const_temp4_4x32b = _mm_set1_epi16(8);
        const_temp3_4x32b = _mm_set1_epi16(32);
        two_nt_4x32b = _mm_set1_epi16(two_nt-nt);
        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD16 pi2_ref_main_idx5, pi2_ref_main_idx6, pi2_ref_main_idx7, pi2_ref_main_idx8;
            WORD8  ai1_fract_temp0_val[16], ai1_fract_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i fract5_8x16b, fract6_8x16b, fract7_8x16b, fract8_8x16b;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i temp11_8x16b, temp12_8x16b, temp13_8x16b, temp14_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* idx = pos >> 5; */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* fract = pos & (31); */
            ref_main_idx_4x32b = _mm_sub_epi16(two_nt_4x32b , _mm_srai_epi16(res_temp5_4x32b,  5));

            _mm_storeu_si128((__m128i *)(ai1_fract_temp0_val), fract_4x32b);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[6]);  /* col=3 */

            fract5_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[8]);  /* col=5 */
            fract6_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[10]);  /* col=6 */
            fract7_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[12]);  /* col=7 */
            fract8_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[14]);  /* col=8 */

            /* (32 - fract) */
            fract_4x32b = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);
            _mm_storeu_si128((__m128i *)(ai1_fract_temp1_val), fract_4x32b);

            temp1_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            temp11_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[8]);  /* col=0 */
            temp12_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[10]);  /* col=1 */
            temp13_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[12]);  /* col=2 */
            temp14_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[14]);  /* col=3 */

            temp11_8x16b = _mm_unpacklo_epi16(temp11_8x16b, fract5_8x16b);
            temp12_8x16b = _mm_unpacklo_epi16(temp12_8x16b, fract6_8x16b);
            temp13_8x16b = _mm_unpacklo_epi16(temp13_8x16b, fract7_8x16b);
            temp14_8x16b = _mm_unpacklo_epi16(temp14_8x16b, fract8_8x16b);

            pi2_ref_main_idx5 = _mm_extract_epi16(ref_main_idx_4x32b, 4);    /* col=5 */
            pi2_ref_main_idx6 = _mm_extract_epi16(ref_main_idx_4x32b, 5);    /* col=6 */
            pi2_ref_main_idx7 = _mm_extract_epi16(ref_main_idx_4x32b, 6);    /* col=7 */
            pi2_ref_main_idx8 = _mm_extract_epi16(ref_main_idx_4x32b, 7);    /* col=8 */

            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;

                __m128i src_temp11_8x16b, src_temp12_8x16b, src_temp13_8x16b, src_temp14_8x16b;
                __m128i src_temp15_8x16b, src_temp16_8x16b, src_temp17_8x16b, src_temp18_8x16b;

                __m128i src_tmp1_8x16b, src_tmp2_8x16b, src_tmp3_8x16b, src_tmp4_8x16b;
                __m128i src_tmp5_8x16b, src_tmp6_8x16b, src_tmp7_8x16b, src_tmp8_8x16b;

                /* loding 16-bit 8 pixels */
                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1-1)); /* col=0 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2-1)); /* col=1 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3-1)); /* col=2 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4-1)); /* col=3 */

                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1)); /* col=0 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2)); /* col=1 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3)); /* col=2 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4)); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* loding 16-bit 8 pixels */
                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx5-1)); /* col=5 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx6-1)); /* col=6 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx7-1)); /* col=7 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx8-1)); /* col=8 */

                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx5)); /* col=5 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx6)); /* col=6 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx7)); /* col=7 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx8)); /* col=8 */

                src_temp11_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=5 */
                src_temp12_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=6 */
                src_temp13_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=7 */
                src_temp14_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=8 */

                src_temp15_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=5 */
                src_temp16_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=6 */
                src_temp17_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=7 */
                src_temp18_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=8 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp11_8x16b = _mm_madd_epi16(src_temp11_8x16b, temp11_8x16b);
                src_temp12_8x16b = _mm_madd_epi16(src_temp12_8x16b, temp12_8x16b);
                src_temp13_8x16b = _mm_madd_epi16(src_temp13_8x16b, temp13_8x16b);
                src_temp14_8x16b = _mm_madd_epi16(src_temp14_8x16b, temp14_8x16b);

                src_temp15_8x16b = _mm_madd_epi16(src_temp15_8x16b, temp11_8x16b);
                src_temp16_8x16b = _mm_madd_epi16(src_temp16_8x16b, temp12_8x16b);
                src_temp17_8x16b = _mm_madd_epi16(src_temp17_8x16b, temp13_8x16b);
                src_temp18_8x16b = _mm_madd_epi16(src_temp18_8x16b, temp14_8x16b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);   /* col=0 */
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);   /* col=1 */
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);   /* col=2 */
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);   /* col=3 */

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp11_8x16b = _mm_add_epi32(src_temp11_8x16b, const_temp_4x32b);
                src_temp12_8x16b = _mm_add_epi32(src_temp12_8x16b, const_temp_4x32b);
                src_temp13_8x16b = _mm_add_epi32(src_temp13_8x16b, const_temp_4x32b);
                src_temp14_8x16b = _mm_add_epi32(src_temp14_8x16b, const_temp_4x32b);

                src_temp15_8x16b = _mm_add_epi32(src_temp15_8x16b, const_temp_4x32b);
                src_temp16_8x16b = _mm_add_epi32(src_temp16_8x16b, const_temp_4x32b);
                src_temp17_8x16b = _mm_add_epi32(src_temp17_8x16b, const_temp_4x32b);
                src_temp18_8x16b = _mm_add_epi32(src_temp18_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp11_8x16b = _mm_srai_epi32(src_temp11_8x16b,  5);   /* col=5 */
                src_temp12_8x16b = _mm_srai_epi32(src_temp12_8x16b,  5);   /* col=6 */
                src_temp13_8x16b = _mm_srai_epi32(src_temp13_8x16b,  5);   /* col=7 */
                src_temp14_8x16b = _mm_srai_epi32(src_temp14_8x16b,  5);   /* col=8 */

                src_temp15_8x16b = _mm_srai_epi32(src_temp15_8x16b,  5);   /* col=5 */
                src_temp16_8x16b = _mm_srai_epi32(src_temp16_8x16b,  5);   /* col=6 */
                src_temp17_8x16b = _mm_srai_epi32(src_temp17_8x16b,  5);   /* col=7 */
                src_temp18_8x16b = _mm_srai_epi32(src_temp18_8x16b,  5);   /* col=8 */

                /* converting 32 bit to 16 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                /* converting 16 bit to 8 bit */
                src_temp11_8x16b = _mm_packus_epi32(src_temp11_8x16b, src_temp15_8x16b); /* col=5 */
                src_temp12_8x16b = _mm_packus_epi32(src_temp12_8x16b, src_temp16_8x16b); /* col=6 */
                src_temp13_8x16b = _mm_packus_epi32(src_temp13_8x16b, src_temp17_8x16b); /* col=7 */
                src_temp14_8x16b = _mm_packus_epi32(src_temp14_8x16b, src_temp18_8x16b); /* col=8 */

                src_temp1_8x16b =_mm_shuffle_epi8(src_temp1_8x16b, sm2);
                src_temp2_8x16b =_mm_shuffle_epi8(src_temp2_8x16b, sm2);
                src_temp3_8x16b =_mm_shuffle_epi8(src_temp3_8x16b, sm2);
                src_temp4_8x16b =_mm_shuffle_epi8(src_temp4_8x16b, sm2);

                src_temp11_8x16b =_mm_shuffle_epi8(src_temp11_8x16b, sm2);
                src_temp12_8x16b =_mm_shuffle_epi8(src_temp12_8x16b, sm2);
                src_temp13_8x16b =_mm_shuffle_epi8(src_temp13_8x16b, sm2);
                src_temp14_8x16b =_mm_shuffle_epi8(src_temp14_8x16b, sm2);

                src_temp5_8x16b = _mm_unpacklo_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi16(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi16(src_temp3_8x16b, src_temp4_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp16_8x16b = _mm_unpacklo_epi16(src_temp13_8x16b, src_temp14_8x16b);
                src_temp17_8x16b = _mm_unpackhi_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi16(src_temp13_8x16b, src_temp14_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi32(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi32(src_temp7_8x16b, src_temp8_8x16b);

                src_temp11_8x16b = _mm_unpacklo_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp12_8x16b = _mm_unpackhi_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp13_8x16b = _mm_unpacklo_epi32(src_temp17_8x16b, src_temp18_8x16b);
                src_temp14_8x16b = _mm_unpackhi_epi32(src_temp17_8x16b, src_temp18_8x16b);

                src_temp5_8x16b = _mm_unpacklo_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp6_8x16b = _mm_unpackhi_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp7_8x16b = _mm_unpacklo_epi64(src_temp2_8x16b, src_temp12_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi64(src_temp2_8x16b, src_temp12_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp16_8x16b = _mm_unpackhi_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp17_8x16b = _mm_unpacklo_epi64(src_temp4_8x16b, src_temp14_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi64(src_temp4_8x16b, src_temp14_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 0)), src_temp5_8x16b);          /* row=0 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 1)), src_temp6_8x16b);       /* row=1 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 2)), src_temp7_8x16b);       /* row=2 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 3)), src_temp8_8x16b);       /* row=3 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 4)), src_temp15_8x16b);       /* row=4 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 5)), src_temp16_8x16b);       /* row=5 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 6)), src_temp17_8x16b);       /* row=6 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 7)), src_temp18_8x16b);       /* row=7 */
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for luma mode 11 to mode 17
*
* @par Description:
*    Intraprediction for mode 11 to 17  (negative angle, horizontal mode )
*    with reference  neighboring samples location pointed by 'pu2_ref' to the
*    TU block location pointed by 'pu2_dst'
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
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_luma_mode_11_to_17_sse42(UWORD16 *pu2_ref,
                                                   WORD32 src_strd,
                                                   UWORD16 *pu2_dst,
                                                   WORD32 dst_strd,
                                                   WORD32 nt,
                                                   WORD32 mode,
                                                   UWORD8 bit_depth)
{
    /* This function and ihevc_intra_pred_luma_mode_19_to_25 are same except */
    /* for ref main & side samples assignment,can be combined for */
    /* optimzation */

    WORD32 row, col, k;
    WORD32 two_nt;
    WORD32 intra_pred_ang, inv_ang, inv_ang_sum;
    WORD32 ref_idx;

    __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
    __m128i fract_4x32b, zero_8x16b, intra_pred_ang_4x32b;
    __m128i row_4x32b, two_nt_4x32b, ref_main_idx_4x32b, res_temp5_4x32b;

    UWORD16 ref_temp[2 * MAX_CU_SIZE + 1];
    UWORD16 *ref_main;

    inv_ang_sum = 128;
    two_nt    = 2 * nt;

    intra_pred_ang = gai4_ihevc_ang_table[mode];

    inv_ang = gai4_ihevc_inv_ang_table[mode - 11];
    /* Intermediate reference samples for negative angle modes */
    /* This have to be removed during optimization */
    /* For horizontal modes, (ref main = ref left) (ref side = ref above) */

    ref_main = ref_temp + nt - 1;
    for(k = 0; k < nt + 1; k++)
        ref_temp[k + nt - 1] = pu2_ref[two_nt - k];

    ref_main = ref_temp + nt - 1 ;
    ref_idx = (nt * intra_pred_ang) >> 5;

    /* SIMD Optimization can be done using look-up table for the loop */
    /* For negative angled derive the main reference samples from side */
    /* reference samples refer to section 8.4.4.2.6 */
    for(k = -1; k > ref_idx; k--)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[two_nt + (inv_ang_sum >> 8)];
    }

    /* For the angles other then 45 degree, interpolation btw 2 neighboring */
    /* samples dependent on distance to obtain destination sample */
    const_temp_4x32b  = _mm_set1_epi32(16);
    const_temp2_4x32b = _mm_set1_epi32(31);
    const_temp3_4x32b = _mm_set1_epi32(32);
    const_temp4_4x32b = _mm_set1_epi32(4);

    two_nt_4x32b = _mm_set1_epi32(1);

    zero_8x16b = _mm_set1_epi16(0);
    mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

    /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
    intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

    row_4x32b = _mm_set_epi32(4, 3, 2, 1);

    if(nt==4)
    {
        WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;
        WORD8  ai1_fract_temp_val[16], ai1_row_temp_val[16];

        __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
        __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;

        __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
        __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;
        __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;

        /* pos = ((row + 1) * intra_pred_ang); */
        res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

        /* idx = pos >> 5; */
        fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

        /* fract = pos & (31); */
        ref_main_idx_4x32b = _mm_add_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));

        /* (32 - fract) */
        row_4x32b = _mm_sub_epi32(const_temp3_4x32b , fract_4x32b);

        _mm_storeu_si128((__m128i *)(ai1_fract_temp_val), fract_4x32b);
        _mm_storeu_si128((__m128i *)(ai1_row_temp_val),  row_4x32b);

        fract1_8x16b = _mm_set1_epi16(ai1_fract_temp_val[0]);  /* col=0 */
        fract2_8x16b = _mm_set1_epi16(ai1_fract_temp_val[4]);  /* col=1 */
        fract3_8x16b = _mm_set1_epi16(ai1_fract_temp_val[8]);  /* col=2 */
        fract4_8x16b = _mm_set1_epi16(ai1_fract_temp_val[12]);  /* col=3 */

        temp1_8x16b = _mm_set1_epi16(ai1_row_temp_val[0]);  /* col=0 */
        temp2_8x16b = _mm_set1_epi16(ai1_row_temp_val[4]);  /* col=1 */
        temp3_8x16b = _mm_set1_epi16(ai1_row_temp_val[8]);  /* col=2 */
        temp4_8x16b = _mm_set1_epi16(ai1_row_temp_val[12]);  /* col=3 */

        temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
        temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
        temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
        temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

        ref_main_temp0 = _mm_srli_si128(ref_main_idx_4x32b , 4);  /* next 32 bit values */
        ref_main_temp1 = _mm_srli_si128(ref_main_idx_4x32b , 8);  /* next 32 bit values */
        ref_main_temp2 = _mm_srli_si128(ref_main_idx_4x32b , 12); /* next 32 bit values */
        ref_main_idx1  = _mm_cvtsi128_si32(ref_main_idx_4x32b);    /* col=0 */
        ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* col=1 */
        ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* col=2 */
        ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* col=3 */

        /* loding 8-bit 16 pixels */
        src_temp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1)); /* col=0 */
        src_temp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx2)); /* col=1 */
        src_temp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx3)); /* col=2 */
        src_temp8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx4)); /* col=3 */

        src_temp1_8x16b = _mm_srli_si128(src_temp5_8x16b, 2); /* col=0 */
        src_temp2_8x16b = _mm_srli_si128(src_temp6_8x16b, 2); /* col=1 */
        src_temp3_8x16b = _mm_srli_si128(src_temp7_8x16b, 2); /* col=2 */
        src_temp4_8x16b = _mm_srli_si128(src_temp8_8x16b, 2); /* col=3 */

        src_temp1_8x16b =  _mm_unpacklo_epi16(src_temp5_8x16b, src_temp1_8x16b); /* col=0 */
        src_temp2_8x16b =  _mm_unpacklo_epi16(src_temp6_8x16b, src_temp2_8x16b); /* col=1 */
        src_temp3_8x16b =  _mm_unpacklo_epi16(src_temp7_8x16b, src_temp3_8x16b); /* col=2 */
        src_temp4_8x16b =  _mm_unpacklo_epi16(src_temp8_8x16b, src_temp4_8x16b); /* col=3 */

        /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
        src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
        src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
        src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
        src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

        /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
        src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
        src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
        src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
        src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

        /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
        src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
        src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
        src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
        src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

        /* converting 32 bit to 16 bit */
        src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, zero_8x16b); /* col=0 */
        src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, zero_8x16b); /* col=1 */
        src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, zero_8x16b); /* col=2 */
        src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, zero_8x16b); /* col=3 */

        src_temp5_8x16b = _mm_unpacklo_epi16(src_temp1_8x16b, src_temp2_8x16b);
        src_temp6_8x16b = _mm_unpacklo_epi16(src_temp3_8x16b, src_temp4_8x16b);

        src_temp7_8x16b  = _mm_unpacklo_epi32(src_temp5_8x16b, src_temp6_8x16b);
        src_temp1_8x16b  = _mm_shuffle_epi32(src_temp7_8x16b, _MM_SHUFFLE(3, 2, 3, 2));
        src_temp2_8x16b  = _mm_unpackhi_epi32(src_temp5_8x16b, src_temp6_8x16b);
        src_temp3_8x16b  = _mm_shuffle_epi32(src_temp2_8x16b, _MM_SHUFFLE(3, 2, 3, 2));

        _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp7_8x16b);
        _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp1_8x16b);
        _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp2_8x16b);
        _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp3_8x16b);
    }
    else if(nt==16 || nt==32)
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        const_temp4_4x32b = _mm_set1_epi16(8);
        const_temp3_4x32b =_mm_set1_epi16(32);
        two_nt_4x32b = _mm_set1_epi16(1);

        for(col = 0; col < nt; col += 8)
        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD16 pi2_ref_main_idx5, pi2_ref_main_idx6, pi2_ref_main_idx7, pi2_ref_main_idx8;
            WORD8  ai1_fract_temp0_val[16], ai1_fract_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i fract5_8x16b, fract6_8x16b, fract7_8x16b, fract8_8x16b;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i temp11_8x16b, temp12_8x16b, temp13_8x16b, temp14_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* idx = pos >> 5; */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* fract = pos & (31); */
            ref_main_idx_4x32b = _mm_add_epi16(two_nt_4x32b , _mm_srai_epi16(res_temp5_4x32b,  5));

            row_4x32b= _mm_add_epi16(row_4x32b, const_temp4_4x32b);

            _mm_storeu_si128((__m128i *)(ai1_fract_temp0_val), fract_4x32b);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[6]);  /* col=3 */

            fract5_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[8]);  /* col=5 */
            fract6_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[10]);  /* col=6 */
            fract7_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[12]);  /* col=7 */
            fract8_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[14]);  /* col=8 */

            /* (32 - fract) */
            fract_4x32b = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);
            _mm_storeu_si128((__m128i *)(ai1_fract_temp1_val), fract_4x32b);

            temp1_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            temp11_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[8]);  /* col=0 */
            temp12_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[10]);  /* col=1 */
            temp13_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[12]);  /* col=2 */
            temp14_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[14]);  /* col=3 */

            temp11_8x16b = _mm_unpacklo_epi16(temp11_8x16b, fract5_8x16b);
            temp12_8x16b = _mm_unpacklo_epi16(temp12_8x16b, fract6_8x16b);
            temp13_8x16b = _mm_unpacklo_epi16(temp13_8x16b, fract7_8x16b);
            temp14_8x16b = _mm_unpacklo_epi16(temp14_8x16b, fract8_8x16b);

            pi2_ref_main_idx5 = _mm_extract_epi16(ref_main_idx_4x32b, 4);    /* col=5 */
            pi2_ref_main_idx6 = _mm_extract_epi16(ref_main_idx_4x32b, 5);    /* col=6 */
            pi2_ref_main_idx7 = _mm_extract_epi16(ref_main_idx_4x32b, 6);    /* col=7 */
            pi2_ref_main_idx8 = _mm_extract_epi16(ref_main_idx_4x32b, 7);    /* col=8 */

            for(row = 0; row < nt; row += 8)
            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;


                __m128i src_temp11_8x16b, src_temp12_8x16b, src_temp13_8x16b, src_temp14_8x16b;
                __m128i src_temp15_8x16b, src_temp16_8x16b, src_temp17_8x16b, src_temp18_8x16b;

                __m128i src_tmp1_8x16b, src_tmp2_8x16b, src_tmp3_8x16b, src_tmp4_8x16b;
                __m128i src_tmp5_8x16b, src_tmp6_8x16b, src_tmp7_8x16b, src_tmp8_8x16b;

                /* loding 16-bit 8 pixels */
                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1+row)); /* col=0 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2+row)); /* col=1 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3+row)); /* col=2 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4+row)); /* col=3 */


                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1+row+1)); /* col=0 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2+row+1)); /* col=1 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3+row+1)); /* col=2 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4+row+1)); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* loding 16-bit 8 pixels */
                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx5+row)); /* col=5 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx6+row)); /* col=6 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx7+row)); /* col=7 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx8+row)); /* col=8 */

                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx5+row+1)); /* col=5 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx6+row+1)); /* col=6 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx7+row+1)); /* col=7 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx8+row+1)); /* col=8 */

                src_temp11_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=5 */
                src_temp12_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=6 */
                src_temp13_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=7 */
                src_temp14_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=8 */

                src_temp15_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=5 */
                src_temp16_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=6 */
                src_temp17_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=7 */
                src_temp18_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=8 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp11_8x16b = _mm_madd_epi16(src_temp11_8x16b, temp11_8x16b);
                src_temp12_8x16b = _mm_madd_epi16(src_temp12_8x16b, temp12_8x16b);
                src_temp13_8x16b = _mm_madd_epi16(src_temp13_8x16b, temp13_8x16b);
                src_temp14_8x16b = _mm_madd_epi16(src_temp14_8x16b, temp14_8x16b);

                src_temp15_8x16b = _mm_madd_epi16(src_temp15_8x16b, temp11_8x16b);
                src_temp16_8x16b = _mm_madd_epi16(src_temp16_8x16b, temp12_8x16b);
                src_temp17_8x16b = _mm_madd_epi16(src_temp17_8x16b, temp13_8x16b);
                src_temp18_8x16b = _mm_madd_epi16(src_temp18_8x16b, temp14_8x16b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);   /* col=0 */
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);   /* col=1 */
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);   /* col=2 */
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);   /* col=3 */

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp11_8x16b = _mm_add_epi32(src_temp11_8x16b, const_temp_4x32b);
                src_temp12_8x16b = _mm_add_epi32(src_temp12_8x16b, const_temp_4x32b);
                src_temp13_8x16b = _mm_add_epi32(src_temp13_8x16b, const_temp_4x32b);
                src_temp14_8x16b = _mm_add_epi32(src_temp14_8x16b, const_temp_4x32b);

                src_temp15_8x16b = _mm_add_epi32(src_temp15_8x16b, const_temp_4x32b);
                src_temp16_8x16b = _mm_add_epi32(src_temp16_8x16b, const_temp_4x32b);
                src_temp17_8x16b = _mm_add_epi32(src_temp17_8x16b, const_temp_4x32b);
                src_temp18_8x16b = _mm_add_epi32(src_temp18_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp11_8x16b = _mm_srai_epi32(src_temp11_8x16b,  5);   /* col=5 */
                src_temp12_8x16b = _mm_srai_epi32(src_temp12_8x16b,  5);   /* col=6 */
                src_temp13_8x16b = _mm_srai_epi32(src_temp13_8x16b,  5);   /* col=7 */
                src_temp14_8x16b = _mm_srai_epi32(src_temp14_8x16b,  5);   /* col=8 */

                src_temp15_8x16b = _mm_srai_epi32(src_temp15_8x16b,  5);   /* col=5 */
                src_temp16_8x16b = _mm_srai_epi32(src_temp16_8x16b,  5);   /* col=6 */
                src_temp17_8x16b = _mm_srai_epi32(src_temp17_8x16b,  5);   /* col=7 */
                src_temp18_8x16b = _mm_srai_epi32(src_temp18_8x16b,  5);   /* col=8 */

                /* converting 32 bit to 16 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                /* converting 32 bit to 16 bit */
                src_temp11_8x16b = _mm_packus_epi32(src_temp11_8x16b, src_temp15_8x16b); /* col=5 */
                src_temp12_8x16b = _mm_packus_epi32(src_temp12_8x16b, src_temp16_8x16b); /* col=6 */
                src_temp13_8x16b = _mm_packus_epi32(src_temp13_8x16b, src_temp17_8x16b); /* col=7 */
                src_temp14_8x16b = _mm_packus_epi32(src_temp14_8x16b, src_temp18_8x16b); /* col=8 */

                src_temp5_8x16b = _mm_unpacklo_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi16(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi16(src_temp3_8x16b, src_temp4_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp16_8x16b = _mm_unpacklo_epi16(src_temp13_8x16b, src_temp14_8x16b);
                src_temp17_8x16b = _mm_unpackhi_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi16(src_temp13_8x16b, src_temp14_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi32(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi32(src_temp7_8x16b, src_temp8_8x16b);

                src_temp11_8x16b = _mm_unpacklo_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp12_8x16b = _mm_unpackhi_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp13_8x16b = _mm_unpacklo_epi32(src_temp17_8x16b, src_temp18_8x16b);
                src_temp14_8x16b = _mm_unpackhi_epi32(src_temp17_8x16b, src_temp18_8x16b);


                src_temp5_8x16b = _mm_unpacklo_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp6_8x16b = _mm_unpackhi_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp7_8x16b = _mm_unpacklo_epi64(src_temp2_8x16b, src_temp12_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi64(src_temp2_8x16b, src_temp12_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp16_8x16b = _mm_unpackhi_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp17_8x16b = _mm_unpacklo_epi64(src_temp4_8x16b, src_temp14_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi64(src_temp4_8x16b, src_temp14_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * row)), src_temp5_8x16b);          /* row=0 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+1))), src_temp6_8x16b);       /* row=1 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+2))), src_temp7_8x16b);       /* row=2 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+3))), src_temp8_8x16b);       /* row=3 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+4))), src_temp15_8x16b);       /* row=4 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+5))), src_temp16_8x16b);       /* row=5 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+6))), src_temp17_8x16b);       /* row=6 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+7))), src_temp18_8x16b);       /* row=7 */
            }
        }
    }
    else
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        const_temp4_4x32b = _mm_set1_epi16(8);
        const_temp3_4x32b = _mm_set1_epi16(32);
        two_nt_4x32b = _mm_set1_epi16(1);

        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD16 pi2_ref_main_idx5, pi2_ref_main_idx6, pi2_ref_main_idx7, pi2_ref_main_idx8;
            WORD8  ai1_fract_temp0_val[16], ai1_fract_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i fract5_8x16b, fract6_8x16b, fract7_8x16b, fract8_8x16b;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i temp11_8x16b, temp12_8x16b, temp13_8x16b, temp14_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* idx = pos >> 5; */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* fract = pos & (31); */
            ref_main_idx_4x32b = _mm_add_epi16(two_nt_4x32b , _mm_srai_epi16(res_temp5_4x32b,  5));

            _mm_storeu_si128((__m128i *)(ai1_fract_temp0_val), fract_4x32b);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[6]);  /* col=3 */

            fract5_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[8]);  /* col=5 */
            fract6_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[10]);  /* col=6 */
            fract7_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[12]);  /* col=7 */
            fract8_8x16b = _mm_set1_epi16(ai1_fract_temp0_val[14]);  /* col=8 */

            /* (32 - fract) */
            fract_4x32b = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);
            _mm_storeu_si128((__m128i *)(ai1_fract_temp1_val), fract_4x32b);

            temp1_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            temp11_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[8]);  /* col=0 */
            temp12_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[10]);  /* col=1 */
            temp13_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[12]);  /* col=2 */
            temp14_8x16b = _mm_set1_epi16(ai1_fract_temp1_val[14]);  /* col=3 */

            temp11_8x16b = _mm_unpacklo_epi16(temp11_8x16b, fract5_8x16b);
            temp12_8x16b = _mm_unpacklo_epi16(temp12_8x16b, fract6_8x16b);
            temp13_8x16b = _mm_unpacklo_epi16(temp13_8x16b, fract7_8x16b);
            temp14_8x16b = _mm_unpacklo_epi16(temp14_8x16b, fract8_8x16b);

            pi2_ref_main_idx5 = _mm_extract_epi16(ref_main_idx_4x32b, 4);    /* col=5 */
            pi2_ref_main_idx6 = _mm_extract_epi16(ref_main_idx_4x32b, 5);    /* col=6 */
            pi2_ref_main_idx7 = _mm_extract_epi16(ref_main_idx_4x32b, 6);    /* col=7 */
            pi2_ref_main_idx8 = _mm_extract_epi16(ref_main_idx_4x32b, 7);    /* col=8 */

            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;

                __m128i src_temp11_8x16b, src_temp12_8x16b, src_temp13_8x16b, src_temp14_8x16b;
                __m128i src_temp15_8x16b, src_temp16_8x16b, src_temp17_8x16b, src_temp18_8x16b;

                __m128i src_tmp1_8x16b, src_tmp2_8x16b, src_tmp3_8x16b, src_tmp4_8x16b;
                __m128i src_tmp5_8x16b, src_tmp6_8x16b, src_tmp7_8x16b, src_tmp8_8x16b;

                /* loding 16-bit 8 pixels */
                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1)); /* col=0 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2)); /* col=1 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3)); /* col=2 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4)); /* col=3 */

                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1+1)); /* col=0 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2+1)); /* col=1 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3+1)); /* col=2 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4+1)); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* loding 8-bit 16 pixels */
                src_tmp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx5)); /* col=5 */
                src_tmp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx6)); /* col=6 */
                src_tmp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx7)); /* col=7 */
                src_tmp8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx8)); /* col=8 */

                src_tmp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx5+1)); /* col=5 */
                src_tmp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx6+1)); /* col=6 */
                src_tmp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx7+1)); /* col=7 */
                src_tmp4_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx8+1)); /* col=8 */

                src_temp11_8x16b =  _mm_unpacklo_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp12_8x16b =  _mm_unpacklo_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp13_8x16b =  _mm_unpacklo_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp14_8x16b =  _mm_unpacklo_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                src_temp15_8x16b =  _mm_unpackhi_epi16(src_tmp5_8x16b, src_tmp1_8x16b); /* col=0 */
                src_temp16_8x16b =  _mm_unpackhi_epi16(src_tmp6_8x16b, src_tmp2_8x16b); /* col=1 */
                src_temp17_8x16b =  _mm_unpackhi_epi16(src_tmp7_8x16b, src_tmp3_8x16b); /* col=2 */
                src_temp18_8x16b =  _mm_unpackhi_epi16(src_tmp8_8x16b, src_tmp4_8x16b); /* col=3 */

                /* fract*(pu2_ref[ref_main_idx + 1]- pu2_ref[ref_main_idx]) */
                src_temp11_8x16b = _mm_madd_epi16(src_temp11_8x16b, temp11_8x16b);
                src_temp12_8x16b = _mm_madd_epi16(src_temp12_8x16b, temp12_8x16b);
                src_temp13_8x16b = _mm_madd_epi16(src_temp13_8x16b, temp13_8x16b);
                src_temp14_8x16b = _mm_madd_epi16(src_temp14_8x16b, temp14_8x16b);

                src_temp15_8x16b = _mm_madd_epi16(src_temp15_8x16b, temp11_8x16b);
                src_temp16_8x16b = _mm_madd_epi16(src_temp16_8x16b, temp12_8x16b);
                src_temp17_8x16b = _mm_madd_epi16(src_temp17_8x16b, temp13_8x16b);
                src_temp18_8x16b = _mm_madd_epi16(src_temp18_8x16b, temp14_8x16b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* row=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* row=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* row=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* row=3 */

                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
                src_temp11_8x16b = _mm_add_epi32(src_temp11_8x16b, const_temp_4x32b);
                src_temp12_8x16b = _mm_add_epi32(src_temp12_8x16b, const_temp_4x32b);
                src_temp13_8x16b = _mm_add_epi32(src_temp13_8x16b, const_temp_4x32b);
                src_temp14_8x16b = _mm_add_epi32(src_temp14_8x16b, const_temp_4x32b);

                src_temp15_8x16b = _mm_add_epi32(src_temp15_8x16b, const_temp_4x32b);
                src_temp16_8x16b = _mm_add_epi32(src_temp16_8x16b, const_temp_4x32b);
                src_temp17_8x16b = _mm_add_epi32(src_temp17_8x16b, const_temp_4x32b);
                src_temp18_8x16b = _mm_add_epi32(src_temp18_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp11_8x16b = _mm_srai_epi32(src_temp11_8x16b,  5);   /* col=5 */
                src_temp12_8x16b = _mm_srai_epi32(src_temp12_8x16b,  5);   /* col=6 */
                src_temp13_8x16b = _mm_srai_epi32(src_temp13_8x16b,  5);   /* col=7 */
                src_temp14_8x16b = _mm_srai_epi32(src_temp14_8x16b,  5);   /* col=8 */

                src_temp15_8x16b = _mm_srai_epi32(src_temp15_8x16b,  5);   /* col=5 */
                src_temp16_8x16b = _mm_srai_epi32(src_temp16_8x16b,  5);   /* col=6 */
                src_temp17_8x16b = _mm_srai_epi32(src_temp17_8x16b,  5);   /* col=7 */
                src_temp18_8x16b = _mm_srai_epi32(src_temp18_8x16b,  5);   /* col=8 */

                /* converting 32 bit to 16 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                /* converting 32 bit to 16 bit */
                src_temp11_8x16b = _mm_packus_epi32(src_temp11_8x16b, src_temp15_8x16b); /* col=4 */
                src_temp12_8x16b = _mm_packus_epi32(src_temp12_8x16b, src_temp16_8x16b); /* col=5 */
                src_temp13_8x16b = _mm_packus_epi32(src_temp13_8x16b, src_temp17_8x16b); /* col=6 */
                src_temp14_8x16b = _mm_packus_epi32(src_temp14_8x16b, src_temp18_8x16b); /* col=7 */

                src_temp5_8x16b = _mm_unpacklo_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi16(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi16(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi16(src_temp3_8x16b, src_temp4_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp16_8x16b = _mm_unpacklo_epi16(src_temp13_8x16b, src_temp14_8x16b);
                src_temp17_8x16b = _mm_unpackhi_epi16(src_temp11_8x16b, src_temp12_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi16(src_temp13_8x16b, src_temp14_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi32(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi32(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi32(src_temp7_8x16b, src_temp8_8x16b);

                src_temp11_8x16b = _mm_unpacklo_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp12_8x16b = _mm_unpackhi_epi32(src_temp15_8x16b, src_temp16_8x16b);
                src_temp13_8x16b = _mm_unpacklo_epi32(src_temp17_8x16b, src_temp18_8x16b);
                src_temp14_8x16b = _mm_unpackhi_epi32(src_temp17_8x16b, src_temp18_8x16b);


                src_temp5_8x16b = _mm_unpacklo_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp6_8x16b = _mm_unpackhi_epi64(src_temp1_8x16b, src_temp11_8x16b);
                src_temp7_8x16b = _mm_unpacklo_epi64(src_temp2_8x16b, src_temp12_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi64(src_temp2_8x16b, src_temp12_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp16_8x16b = _mm_unpackhi_epi64(src_temp3_8x16b, src_temp13_8x16b);
                src_temp17_8x16b = _mm_unpacklo_epi64(src_temp4_8x16b, src_temp14_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi64(src_temp4_8x16b, src_temp14_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 0)), src_temp5_8x16b);          /* row=0 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 1)), src_temp6_8x16b);       /* row=1 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 2)), src_temp7_8x16b);       /* row=2 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 3)), src_temp8_8x16b);       /* row=3 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 4)), src_temp15_8x16b);       /* row=4 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 5)), src_temp16_8x16b);       /* row=5 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 6)), src_temp17_8x16b);       /* row=6 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 7)), src_temp18_8x16b);       /* row=7 */
            }
        }
    }
}
/**
*******************************************************************************
*
* @brief
*   Intra prediction interpolation filter for luma mode 19 to mode 25
*
* @par Description:
*    Intraprediction for mode 19 to 25  (negative angle, vertical mode ) with
*    reference  neighboring samples location pointed by 'pu2_ref' to the  TU
*    block location pointed by 'pu2_dst'
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
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_luma_mode_19_to_25_sse42(UWORD16 *pu2_ref,
                                                   WORD32 src_strd,
                                                   UWORD16 *pu2_dst,
                                                   WORD32 dst_strd,
                                                   WORD32 nt,
                                                   WORD32 mode,
                                                   UWORD8 bit_depth)
{
    WORD32 row, k;
    WORD32 two_nt, intra_pred_ang, idx;
    WORD32 inv_ang, inv_ang_sum, pos, fract;
    WORD32 ref_main_idx, ref_idx;
    UWORD16 ref_temp[(2 * MAX_CU_SIZE) + 1];
    UWORD16 *ref_main;

    __m128i zero_8x16b, fract_8x16b, const_temp_8x16b;
    __m128i temp1, temp2, temp3, temp4;

    two_nt = 2 * nt;
    intra_pred_ang = gai4_ihevc_ang_table[mode];
    inv_ang = gai4_ihevc_inv_ang_table[mode - 12];

    /* Intermediate reference samples for negative angle modes */
    /* This have to be removed during optimization */
    /* For horizontal modes, (ref main = ref above) (ref side = ref left) */
    ref_main = ref_temp + nt - 1;
    for(k = 0; k < (nt + 1); k++)
        ref_temp[k + nt - 1] = pu2_ref[two_nt + k];

    ref_idx = (nt * intra_pred_ang) >> 5;
    inv_ang_sum = 128;

    /* SIMD Optimization can be done using look-up table for the loop */
    /* For negative angled derive the main reference samples from side */
    /* reference samples refer to section 8.4.4.2.6 */

    for(k = -1; k > ref_idx; k--)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[two_nt - (inv_ang_sum >> 8)];
    }

    const_temp_8x16b = _mm_set1_epi32(16);

    if(nt==32)
    {
        WORD32 temp_32;

        /* unroll the col loop (inner) */
        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 1)
        {
            __m128i  src_values0, src_values1, src_values2, src_values3;
            __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;
            __m128i  src_temp4_8x16b, src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp_32 = 32-fract;
            ref_main_idx = idx + 1; /* col from 0-31 */

            fract_8x16b = _mm_set1_epi16(fract);
            temp1 = _mm_set1_epi16(temp_32);

            temp1 = _mm_unpacklo_epi16(temp1, fract_8x16b);

            src_values0 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx));     /* col = 0-7 */
            src_values1 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+8));   /* col = 8-15 */
            src_values2 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+16));  /* col = 16-23 */
            src_values3 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+24));  /* col = 24-31 */

            src_temp4_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+1));
            src_temp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+9));
            src_temp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+17));
            src_temp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+25));

            src_temp0_8x16b = _mm_unpacklo_epi16(src_values0, src_temp4_8x16b);
            src_temp1_8x16b = _mm_unpacklo_epi16(src_values1, src_temp5_8x16b);
            src_temp2_8x16b = _mm_unpacklo_epi16(src_values2, src_temp6_8x16b);
            src_temp3_8x16b = _mm_unpacklo_epi16(src_values3, src_temp7_8x16b);

            src_temp4_8x16b = _mm_unpackhi_epi16(src_values0, src_temp4_8x16b);
            src_temp5_8x16b = _mm_unpackhi_epi16(src_values1, src_temp5_8x16b);
            src_temp6_8x16b = _mm_unpackhi_epi16(src_values2, src_temp6_8x16b);
            src_temp7_8x16b = _mm_unpackhi_epi16(src_values3, src_temp7_8x16b);

            src_temp0_8x16b = _mm_madd_epi16(src_temp0_8x16b, temp1);
            src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1);
            src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp1);
            src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp1);

            src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp1);
            src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1);
            src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp1);
            src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp1);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_temp0_8x16b = _mm_add_epi32(src_temp0_8x16b, const_temp_8x16b);
            src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_8x16b);
            src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_8x16b);
            src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_8x16b);

            src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_8x16b);
            src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_8x16b);
            src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_8x16b);
            src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_temp0_8x16b = _mm_srai_epi32(src_temp0_8x16b,  5);
            src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);
            src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);
            src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);

            src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);
            src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);
            src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);
            src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);

            /* converting 32 bit to 16 bit */
            src_temp0_8x16b = _mm_packus_epi32(src_temp0_8x16b, src_temp4_8x16b);
            src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b);
            src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b);

            /* loding 8-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp0_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp1_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp2_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp3_8x16b);

            pu2_dst += dst_strd;
        }
    }
    else if(nt == 16) /* for nt = 16 case */
    {
        WORD32 ref_main_idx1 , fract1, temp_32, temp1_32;
        __m128i fract1_8x16b;

        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 2)
        {
            __m128i  src_values0, src_values1, src_values2, src_values3;
            __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;
            __m128i  src_temp4_8x16b, src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp_32=32-fract;
            ref_main_idx = idx + 1; /* col from 0-15 */

            pos = ((row + 2) * intra_pred_ang);
            idx = pos >> 5;
            fract1 = pos & (31);
            temp1_32=32-fract1;
            ref_main_idx1 = idx + 1; /* col from 0-15 */

            fract_8x16b = _mm_set1_epi16(fract);
            fract1_8x16b = _mm_set1_epi16(fract1);

            temp1= _mm_set1_epi16(temp_32);
            temp2 = _mm_set1_epi16(temp1_32);

            temp1 = _mm_unpacklo_epi16(temp1, fract_8x16b);
            temp2 = _mm_unpacklo_epi16(temp2, fract1_8x16b);

            /* row=0 */
            src_values0 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx));     /* col = 0-7 */
            src_values1 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+8));   /* col = 8-15 */

            /* row=1 */
            src_values2 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1));  /* col = 0-7 */
            src_values3 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+8));  /* col = 8-15 */

            /* row=0 */
            src_temp4_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+1));
            src_temp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+9));

            /* row =1 */
            src_temp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+1));
            src_temp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+9));

            src_temp0_8x16b = _mm_unpacklo_epi16(src_values0, src_temp4_8x16b);
            src_temp1_8x16b = _mm_unpacklo_epi16(src_values1, src_temp5_8x16b);
            src_temp2_8x16b = _mm_unpacklo_epi16(src_values2, src_temp6_8x16b);
            src_temp3_8x16b = _mm_unpacklo_epi16(src_values3, src_temp7_8x16b);

            src_temp4_8x16b = _mm_unpackhi_epi16(src_values0, src_temp4_8x16b);
            src_temp5_8x16b = _mm_unpackhi_epi16(src_values1, src_temp5_8x16b);
            src_temp6_8x16b = _mm_unpackhi_epi16(src_values2, src_temp6_8x16b);
            src_temp7_8x16b = _mm_unpackhi_epi16(src_values3, src_temp7_8x16b);

            /* row=0 */
            src_temp0_8x16b = _mm_madd_epi16(src_temp0_8x16b, temp1);
            src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1);
            /* row=1 */
            src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2);
            src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp2);
            /* row=0 */
            src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp1);
            src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1);
            /* row=1 */
            src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2);
            src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp2);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_temp0_8x16b = _mm_add_epi32(src_temp0_8x16b, const_temp_8x16b);
            src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_8x16b);
            src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_8x16b);
            src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_8x16b);

            src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_8x16b);
            src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_8x16b);
            src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_8x16b);
            src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_temp0_8x16b = _mm_srai_epi32(src_temp0_8x16b,  5);
            src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);
            src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);
            src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);

            src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);
            src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);
            src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);
            src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);

            /* converting 16 bit to 8 bit */
            src_temp0_8x16b = _mm_packus_epi32(src_temp0_8x16b, src_temp4_8x16b);
            src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b);
            src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b);

            /* loding 16-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp0_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp1_8x16b);

            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd), src_temp2_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd+8), src_temp3_8x16b);

            pu2_dst += 2*dst_strd;
        }
    }
    else if(nt==8)
    {
        __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
        __m128i src_values10, src_values11, zero_8x16b, intra_pred_ang_4x32b;
        __m128i row_4x32b, two_nt_4x32b, src_values12;

        const_temp_4x32b  = _mm_set1_epi32(16);
        const_temp2_4x32b = _mm_set1_epi32(31);
        const_temp3_4x32b = _mm_set1_epi32(32);
        const_temp4_4x32b = _mm_set1_epi32(4);

        two_nt_4x32b = _mm_set1_epi32(1);

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
        intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

        row_4x32b = _mm_set_epi32(4, 3, 2, 1);

        for(row = 0; row < nt; row += 4)
        {
            WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;
            WORD8  ai1_src_temp0_val[16], ai1_src_temp1_val[16];

            __m128i res_temp5_4x32b;
            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i src_values0, src_values1, src_values2, src_values3, src_values13;
            __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;
            __m128i src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            src_values12 = _mm_add_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));

            row_4x32b= _mm_add_epi32(row_4x32b, const_temp4_4x32b);

            /* idx = pos >> 5; */
            src_values11= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi32(const_temp3_4x32b , src_values11);

            _mm_storeu_si128((__m128i *)(ai1_src_temp1_val),  src_values11);
            _mm_storeu_si128((__m128i *)(ai1_src_temp0_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_src_temp1_val[0]);  /* row=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_src_temp1_val[4]);  /* row=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_src_temp1_val[8]);  /* row=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_src_temp1_val[12]);  /* row=3 */

            temp1 = _mm_set1_epi16(ai1_src_temp0_val[0]);  /* row=0 */
            temp2 = _mm_set1_epi16(ai1_src_temp0_val[4]);  /* row=1 */
            temp3 = _mm_set1_epi16(ai1_src_temp0_val[8]);  /* row=2 */
            temp4 = _mm_set1_epi16(ai1_src_temp0_val[12]);  /* row=3 */

            temp1 = _mm_unpacklo_epi16(temp1, fract1_8x16b);
            temp2 = _mm_unpacklo_epi16(temp2, fract2_8x16b);
            temp3 = _mm_unpacklo_epi16(temp3, fract3_8x16b);
            temp4 = _mm_unpacklo_epi16(temp4, fract4_8x16b);

            ref_main_temp0 = _mm_srli_si128(src_values12 , 4);  /* next 32 bit values */
            ref_main_temp1 = _mm_srli_si128(src_values12 , 8);  /* next 32 bit values */
            ref_main_temp2 = _mm_srli_si128(src_values12 , 12); /* next 32 bit values */
            ref_main_idx1  = _mm_cvtsi128_si32(src_values12);    /* row=0 */
            ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* row=1 */
            ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* row=2 */
            ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* row=3 */

            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx2));   /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx3));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx4));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+1));
            src_values11 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx2+1));
            src_values12 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx3+1));
            src_values13 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx4+1));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp1);
            src_values1 = _mm_madd_epi16(src_values1, temp2);
            src_values2 = _mm_madd_epi16(src_values2, temp3);
            src_values3 = _mm_madd_epi16(src_values3, temp4);

            src_values10 = _mm_madd_epi16(src_values10, temp1);
            src_values11 = _mm_madd_epi16(src_values11, temp2);
            src_values12 = _mm_madd_epi16(src_values12, temp3);
            src_values13 = _mm_madd_epi16(src_values13, temp4);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 16-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)), src_values0);       /* row=0 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)), src_values1);   /* row=1 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)), src_values2);   /* row=2 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)), src_values3);   /* row=3 */
        }
    }
    else /* if nt =4 */
    {
        __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
        __m128i src_values10 , src_values11, zero_8x16b, intra_pred_ang_4x32b;
        __m128i row_4x32b, two_nt_4x32b, src_values12;

        const_temp_4x32b  = _mm_set1_epi32(16);
        const_temp2_4x32b = _mm_set1_epi32(31);
        const_temp3_4x32b = _mm_set1_epi32(32);
        const_temp4_4x32b = _mm_set1_epi32(4);

        two_nt_4x32b = _mm_set1_epi32(1);

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
        intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

        row_4x32b = _mm_set_epi32(4, 3, 2, 1);
        {
            WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;
            WORD8  ai1_src_temp0_val[16], ai1_src_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b, res_temp5_4x32b;
            __m128i src_values0, src_values1, src_values2, src_values3, src_values13;
            __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;
            __m128i src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            src_values12 = _mm_add_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));

            ref_main_temp0 = _mm_srli_si128(src_values12 , 4);  /* next 32 bit values */
            ref_main_temp1 = _mm_srli_si128(src_values12 , 8);  /* next 32 bit values */
            ref_main_temp2 = _mm_srli_si128(src_values12 , 12); /* next 32 bit values */
            ref_main_idx1  = _mm_cvtsi128_si32(src_values12);    /* row=0 */
            ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* row=1 */
            ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* row=2 */
            ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* row=3 */

            /* idx = pos >> 5; */
            src_values11= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi32(const_temp3_4x32b , src_values11);

            _mm_storeu_si128((__m128i *)(ai1_src_temp1_val),  src_values11);
            _mm_storeu_si128((__m128i *)(ai1_src_temp0_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_src_temp1_val[0]);  /* row=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_src_temp1_val[4]);  /* row=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_src_temp1_val[8]);  /* row=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_src_temp1_val[12]);  /* row=3 */

            temp1 = _mm_set1_epi16(ai1_src_temp0_val[0]);  /* row=0 */
            temp2 = _mm_set1_epi16(ai1_src_temp0_val[4]);  /* row=1 */
            temp3 = _mm_set1_epi16(ai1_src_temp0_val[8]);  /* row=2 */
            temp4 = _mm_set1_epi16(ai1_src_temp0_val[12]);  /* row=3 */

            temp1 = _mm_unpacklo_epi16(temp1, fract1_8x16b);
            temp2 = _mm_unpacklo_epi16(temp2, fract2_8x16b);
            temp3 = _mm_unpacklo_epi16(temp3, fract3_8x16b);
            temp4 = _mm_unpacklo_epi16(temp4, fract4_8x16b);

            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx2));   /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx3));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx4));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+1));
            src_values11 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx2+1));
            src_values12 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx3+1));
            src_values13 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx4+1));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp1);
            src_values1 = _mm_madd_epi16(src_values1, temp2);
            src_values2 = _mm_madd_epi16(src_values2, temp3);
            src_values3 = _mm_madd_epi16(src_values3, temp4);

            src_values10 = _mm_madd_epi16(src_values10, temp1);
            src_values11 = _mm_madd_epi16(src_values11, temp2);
            src_values12 = _mm_madd_epi16(src_values12, temp3);
            src_values13 = _mm_madd_epi16(src_values13, temp4);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 16-bit 4 pixels values */
            _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_values0);       /* row=0 */
            _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_values1);   /* row=1 */
            _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_values2);   /* row=2 */
            _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_values3);   /* row=3 */
        }
    }
}
/**
*******************************************************************************
*
* @brief
*    Intra prediction interpolation filter for luma mode 27 to mode 33
*
* @par Description:
*    Intraprediction for mode 27 to 33  (positive angle, vertical mode ) with
*    reference  neighboring samples location pointed by 'pu2_ref' to the  TU
*    block location pointed by 'pu2_dst'
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
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_luma_mode_27_to_33_sse42(UWORD16 *pu2_ref,
                                                   WORD32 src_strd,
                                                   UWORD16 *pu2_dst,
                                                   WORD32 dst_strd,
                                                   WORD32 nt,
                                                   WORD32 mode,
                                                   UWORD8 bit_depth)
{
    WORD32 row;
    WORD32 two_nt, pos, fract;
    WORD32 intra_pred_ang;
    WORD32 idx, ref_main_idx;

    __m128i zero_8x16b, fract_8x16b, const_temp_8x16b;
    __m128i temp1, temp2, temp3, temp4;

    two_nt = 2 * nt;
    intra_pred_ang = gai4_ihevc_ang_table[mode];

    const_temp_8x16b = _mm_set1_epi32(16);

    if(nt==32)
    {
        WORD32 temp_32;
        /* unroll the col loop (inner) */
        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 1)
        {
            __m128i  src_values0, src_values1, src_values2, src_values3;
            __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;
            __m128i  src_temp4_8x16b, src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp_32 = 32-fract;
            ref_main_idx = two_nt + idx + 1; /* col from 0-31 */

            fract_8x16b = _mm_set1_epi16(fract);
            temp1 = _mm_set1_epi16(temp_32);

            temp1 = _mm_unpacklo_epi16(temp1, fract_8x16b);

            src_values0 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx));     /* col = 0-7 */
            src_values1 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+8));   /* col = 8-15 */
            src_values2 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+16));  /* col = 16-23 */
            src_values3 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+24));  /* col = 24-31 */

            src_temp4_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+1));
            src_temp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+9));
            src_temp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+17));
            src_temp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+25));

            src_temp0_8x16b = _mm_unpacklo_epi16(src_values0, src_temp4_8x16b);
            src_temp1_8x16b = _mm_unpacklo_epi16(src_values1, src_temp5_8x16b);
            src_temp2_8x16b = _mm_unpacklo_epi16(src_values2, src_temp6_8x16b);
            src_temp3_8x16b = _mm_unpacklo_epi16(src_values3, src_temp7_8x16b);

            src_temp4_8x16b = _mm_unpackhi_epi16(src_values0, src_temp4_8x16b);
            src_temp5_8x16b = _mm_unpackhi_epi16(src_values1, src_temp5_8x16b);
            src_temp6_8x16b = _mm_unpackhi_epi16(src_values2, src_temp6_8x16b);
            src_temp7_8x16b = _mm_unpackhi_epi16(src_values3, src_temp7_8x16b);

            src_temp0_8x16b = _mm_madd_epi16(src_temp0_8x16b, temp1);
            src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1);
            src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp1);
            src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp1);

            src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp1);
            src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1);
            src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp1);
            src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp1);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_temp0_8x16b = _mm_add_epi32(src_temp0_8x16b, const_temp_8x16b);
            src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_8x16b);
            src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_8x16b);
            src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_8x16b);

            src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_8x16b);
            src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_8x16b);
            src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_8x16b);
            src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_temp0_8x16b = _mm_srai_epi32(src_temp0_8x16b,  5);
            src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);
            src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);
            src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);

            src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);
            src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);
            src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);
            src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);

            /* converting 32 bit to 16 bit */
            src_temp0_8x16b = _mm_packus_epi32(src_temp0_8x16b, src_temp4_8x16b);
            src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b);
            src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b);

            /* loding 8-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp0_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp1_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+16), src_temp2_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+24), src_temp3_8x16b);

            pu2_dst += dst_strd;
        }
    }
    else if(nt == 16) /* for nt = 16 case */
    {
        WORD32 ref_main_idx1 , fract1, temp_32, temp1_32;
        __m128i fract1_8x16b;

        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 2)
        {
            __m128i  src_values0, src_values1, src_values2, src_values3;
            __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;
            __m128i  src_temp4_8x16b, src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp_32 = 32-fract;
            ref_main_idx = two_nt + idx + 1; /* col from 0-15 */

            pos = ((row + 2) * intra_pred_ang);
            idx = pos >> 5;
            fract1 = pos & (31);
            temp1_32 = 32-fract1;
            ref_main_idx1 = two_nt + idx + 1; /* col from 0-15 */

            fract_8x16b = _mm_set1_epi16(fract);
            fract1_8x16b = _mm_set1_epi16(fract1);

            temp1= _mm_set1_epi16(temp_32);
            temp2 = _mm_set1_epi16(temp1_32);

            temp1 = _mm_unpacklo_epi16(temp1, fract_8x16b);
            temp2 = _mm_unpacklo_epi16(temp2, fract1_8x16b);

            /* row=0 */
            src_values0 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx));     /* col = 0-7 */
            src_values1 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+8));   /* col = 8-15 */

            /* row=1 */
            src_values2 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1));  /* col = 0-7 */
            src_values3 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+8));  /* col = 8-15 */

            /* row=0 */
            src_temp4_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+1));
            src_temp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+9));

            /* row =1 */
            src_temp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+1));
            src_temp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+9));

            src_temp0_8x16b = _mm_unpacklo_epi16(src_values0, src_temp4_8x16b);
            src_temp1_8x16b = _mm_unpacklo_epi16(src_values1, src_temp5_8x16b);
            src_temp2_8x16b = _mm_unpacklo_epi16(src_values2, src_temp6_8x16b);
            src_temp3_8x16b = _mm_unpacklo_epi16(src_values3, src_temp7_8x16b);

            src_temp4_8x16b = _mm_unpackhi_epi16(src_values0, src_temp4_8x16b);
            src_temp5_8x16b = _mm_unpackhi_epi16(src_values1, src_temp5_8x16b);
            src_temp6_8x16b = _mm_unpackhi_epi16(src_values2, src_temp6_8x16b);
            src_temp7_8x16b = _mm_unpackhi_epi16(src_values3, src_temp7_8x16b);

            /* row=0 */
            src_temp0_8x16b = _mm_madd_epi16(src_temp0_8x16b, temp1);
            src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1);
            /* row=1 */
            src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2);
            src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp2);
            /* row=0 */
            src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp1);
            src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1);
            /* row=1 */
            src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2);
            src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp2);


            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_temp0_8x16b = _mm_add_epi32(src_temp0_8x16b, const_temp_8x16b);
            src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_8x16b);
            src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_8x16b);
            src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_8x16b);

            src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_8x16b);
            src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_8x16b);
            src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_8x16b);
            src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_temp0_8x16b = _mm_srai_epi32(src_temp0_8x16b,  5);
            src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);
            src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);
            src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);

            src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);
            src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);
            src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);
            src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);

            /* converting 32 bit to 16 bit */
            src_temp0_8x16b = _mm_packus_epi32(src_temp0_8x16b, src_temp4_8x16b);
            src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b);
            src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b);

            /* loding 16-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_temp0_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_temp1_8x16b);

            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd), src_temp2_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd+8), src_temp3_8x16b);

            pu2_dst += 2*dst_strd;
        }
    }
    else if(nt==8)
    {
        __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
        __m128i src_values10, src_values11, zero_8x16b, intra_pred_ang_4x32b;
        __m128i row_4x32b, two_nt_4x32b, src_values12;

        const_temp_4x32b  = _mm_set1_epi32(16);
        const_temp2_4x32b = _mm_set1_epi32(31);
        const_temp3_4x32b = _mm_set1_epi32(32);
        const_temp4_4x32b = _mm_set1_epi32(4);

        two_nt_4x32b = _mm_set1_epi32(two_nt+1);

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
        intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

        row_4x32b = _mm_set_epi32(4, 3, 2, 1);

        for(row = 0; row < nt; row += 4)
        {

            WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;
            WORD8  ai1_src_temp0_val[16], ai1_src_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b, res_temp5_4x32b;
            __m128i src_values0, src_values1, src_values2, src_values3, src_values13;
            __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;
            __m128i src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            src_values12 = _mm_add_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));

            row_4x32b= _mm_add_epi32(row_4x32b, const_temp4_4x32b);

            /* idx = pos >> 5; */
            src_values11= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi32(const_temp3_4x32b , src_values11);

            _mm_storeu_si128((__m128i *)(ai1_src_temp1_val),  src_values11);
            _mm_storeu_si128((__m128i *)(ai1_src_temp0_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_src_temp1_val[0]);  /* row=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_src_temp1_val[4]);  /* row=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_src_temp1_val[8]);  /* row=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_src_temp1_val[12]);  /* row=3 */

            temp1 = _mm_set1_epi16(ai1_src_temp0_val[0]);  /* row=0 */
            temp2 = _mm_set1_epi16(ai1_src_temp0_val[4]);  /* row=1 */
            temp3 = _mm_set1_epi16(ai1_src_temp0_val[8]);  /* row=2 */
            temp4 = _mm_set1_epi16(ai1_src_temp0_val[12]);  /* row=3 */

            temp1 = _mm_unpacklo_epi16(temp1, fract1_8x16b);
            temp2 = _mm_unpacklo_epi16(temp2, fract2_8x16b);
            temp3 = _mm_unpacklo_epi16(temp3, fract3_8x16b);
            temp4 = _mm_unpacklo_epi16(temp4, fract4_8x16b);

            ref_main_temp0 = _mm_srli_si128(src_values12 , 4);  /* next 32 bit values */
            ref_main_temp1 = _mm_srli_si128(src_values12 , 8);  /* next 32 bit values */
            ref_main_temp2 = _mm_srli_si128(src_values12 , 12); /* next 32 bit values */
            ref_main_idx1  = _mm_cvtsi128_si32(src_values12);    /* row=0 */
            ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* row=1 */
            ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* row=2 */
            ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* row=3 */

            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx2));   /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx3));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx4));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+1));
            src_values11 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx2+1));
            src_values12 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx3+1));
            src_values13 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx4+1));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp1);
            src_values1 = _mm_madd_epi16(src_values1, temp2);
            src_values2 = _mm_madd_epi16(src_values2, temp3);
            src_values3 = _mm_madd_epi16(src_values3, temp4);

            src_values10 = _mm_madd_epi16(src_values10, temp1);
            src_values11 = _mm_madd_epi16(src_values11, temp2);
            src_values12 = _mm_madd_epi16(src_values12, temp3);
            src_values13 = _mm_madd_epi16(src_values13, temp4);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 16-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd)), src_values0);       /* row=0 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)), src_values1);   /* row=1 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)), src_values2);   /* row=2 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)), src_values3);   /* row=3 */
        }
    }
    else /* if nt =4 */
    {
        __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
        __m128i src_values10, src_values11, zero_8x16b, intra_pred_ang_4x32b;
        __m128i row_4x32b, two_nt_4x32b, src_values12;

        const_temp_4x32b  = _mm_set1_epi32(16);
        const_temp2_4x32b = _mm_set1_epi32(31);
        const_temp3_4x32b = _mm_set1_epi32(32);
        const_temp4_4x32b = _mm_set1_epi32(4);

        two_nt_4x32b = _mm_set1_epi32(two_nt+1);

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
        intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

        row_4x32b = _mm_set_epi32(4, 3, 2, 1);
        {
            WORD8  ai1_src_temp0_val[16], ai1_src_temp1_val[16];

            WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b, res_temp5_4x32b;
            __m128i src_values0, src_values1, src_values2, src_values3, src_values13;
            __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;
            __m128i src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            src_values12 = _mm_add_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));

            ref_main_temp0 = _mm_srli_si128(src_values12 , 4);  /* next 32 bit values */
            ref_main_temp1 = _mm_srli_si128(src_values12 , 8);  /* next 32 bit values */
            ref_main_temp2 = _mm_srli_si128(src_values12 , 12); /* next 32 bit values */
            ref_main_idx1  = _mm_cvtsi128_si32(src_values12);    /* row=0 */
            ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* row=1 */
            ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* row=2 */
            ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* row=3 */

            /* idx = pos >> 5; */
            src_values11= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi32(const_temp3_4x32b , src_values11);

            _mm_storeu_si128((__m128i *)(ai1_src_temp1_val),  src_values11);
            _mm_storeu_si128((__m128i *)(ai1_src_temp0_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_src_temp1_val[0]);  /* row=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_src_temp1_val[4]);  /* row=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_src_temp1_val[8]);  /* row=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_src_temp1_val[12]);  /* row=3 */

            temp1 = _mm_set1_epi16(ai1_src_temp0_val[0]);  /* row=0 */
            temp2 = _mm_set1_epi16(ai1_src_temp0_val[4]);  /* row=1 */
            temp3 = _mm_set1_epi16(ai1_src_temp0_val[8]);  /* row=2 */
            temp4 = _mm_set1_epi16(ai1_src_temp0_val[12]);  /* row=3 */

            temp1 = _mm_unpacklo_epi16(temp1, fract1_8x16b);
            temp2 = _mm_unpacklo_epi16(temp2, fract2_8x16b);
            temp3 = _mm_unpacklo_epi16(temp3, fract3_8x16b);
            temp4 = _mm_unpacklo_epi16(temp4, fract4_8x16b);

            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx2));   /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx3));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx4));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+1));
            src_values11 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx2+1));
            src_values12 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx3+1));
            src_values13 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx4+1));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp1);
            src_values1 = _mm_madd_epi16(src_values1, temp2);
            src_values2 = _mm_madd_epi16(src_values2, temp3);
            src_values3 = _mm_madd_epi16(src_values3, temp4);

            src_values10 = _mm_madd_epi16(src_values10, temp1);
            src_values11 = _mm_madd_epi16(src_values11, temp2);
            src_values12 = _mm_madd_epi16(src_values12, temp3);
            src_values13 = _mm_madd_epi16(src_values13, temp4);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu2_ref[ref_main_idx]+ fract * pu2_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 16-bit 4 pixels values */
            _mm_storel_epi64((__m128i *)(pu2_dst+(0 * dst_strd)), src_values0);   /* row=0 */
            _mm_storel_epi64((__m128i *)(pu2_dst+(1 * dst_strd)), src_values1);   /* row=1 */
            _mm_storel_epi64((__m128i *)(pu2_dst+(2 * dst_strd)), src_values2);   /* row=2 */
            _mm_storel_epi64((__m128i *)(pu2_dst+(3 * dst_strd)), src_values3);   /* row=3 */
        }
    }
}
