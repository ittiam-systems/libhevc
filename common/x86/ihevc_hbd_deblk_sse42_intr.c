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
*  ihevc_hbd_deblk_sse42_intr.c
*
* @brief
*  Contains function definitions for high bit depth deblocking filters using
*  SSE4.2 intrinsics.
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_deblk_luma_vert_sse42()
*  - ihevc_hbd_deblk_luma_horz_sse42()
*  - ihevc_hbd_deblk_chroma_vert_sse42()
*  - ihevc_hbd_deblk_chroma_horz_sse42()
*
* @remarks
*  None
*
*******************************************************************************
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>

#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_deblk.h"
#include "ihevc_deblk_tables.h"
#include "ihevc_debug.h"
#include "ihevc_defs.h"
#include "ihevc_hbd_tables_x86_intr.h"

/**
*******************************************************************************
*
* @brief
*  Decision process and filtering for the luma block vertical edge (HBD SSE4.2).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_luma_vert_sse42(UWORD16 *pu2_src,
                                     WORD32 src_strd,
                                     WORD32 bs,
                                     WORD32 quant_param_p,
                                     WORD32 quant_param_q,
                                     WORD32 beta_offset_div2,
                                     WORD32 tc_offset_div2,
                                     WORD32 filter_flag_p,
                                     WORD32 filter_flag_q,
                                     UWORD8 bit_depth)
{
    WORD32 qp_luma, beta_indx, tc_indx;
    WORD32 beta, tc;
    WORD32 d, dp, dq, d_sam0, d_sam3;

    WORD32 d3, d0, de_0, de_1, de_2, de_3;
    WORD32 de, dep, deq;
    __m128i src_row0_8x16b, src_row1_8x16b, src_row2_8x16b, src_row3_8x16b;
    __m128i clip_tmp_8x16b;

    {
        __m128i src_tmp_8x16b, coef_8x16b, mask_d_result_4x32b, mask_de_result_8x16b, mask_de1_result_8x16b;
        __m128i mask_16x8b, temp_coef0_8x16b, temp_coef1_8x16b;

        ASSERT((bs > 0) && (bs <= 3));
        ASSERT(filter_flag_p || filter_flag_q);

        qp_luma = (quant_param_p + quant_param_q + 1) >> 1;
        beta_indx = CLIP3(qp_luma + (beta_offset_div2 << 1), 0, 51);

        tc_indx = CLIP3(qp_luma + (2 * (bs >> 1)) + (tc_offset_div2 << 1), 0, 53);

        beta = gai4_ihevc_beta_table[beta_indx] * (1 << (bit_depth - 8));
        tc = gai4_ihevc_tc_table[tc_indx] * (1 << (bit_depth - 8));
        if(0 == tc)
        {
            return;
        }
        src_row0_8x16b = _mm_loadu_si128((__m128i *)(pu2_src - 4));
        src_row3_8x16b = _mm_loadu_si128((__m128i *)((pu2_src - 4) + 3 * src_strd));

        coef_8x16b = _mm_loadu_si128((__m128i *)(coef_hbd_d));
        mask_16x8b = _mm_loadu_si128((__m128i *)(shuffle_hbd_d));

        src_tmp_8x16b = _mm_shuffle_epi8(src_row0_8x16b, mask_16x8b);
        mask_de1_result_8x16b = _mm_shuffle_epi8(src_row3_8x16b, mask_16x8b);

        mask_de_result_8x16b = _mm_unpacklo_epi64(src_tmp_8x16b, mask_de1_result_8x16b);
        mask_de1_result_8x16b = _mm_unpackhi_epi64(src_tmp_8x16b, mask_de1_result_8x16b);

        src_tmp_8x16b = _mm_madd_epi16(src_row3_8x16b, coef_8x16b);
        mask_d_result_4x32b = _mm_madd_epi16(src_row0_8x16b, coef_8x16b);

        mask_d_result_4x32b = _mm_packs_epi32(mask_d_result_4x32b, src_tmp_8x16b);

        temp_coef0_8x16b = _mm_cmpeq_epi16(src_tmp_8x16b, src_tmp_8x16b);
        temp_coef1_8x16b = _mm_srli_epi16(temp_coef0_8x16b, 15);

        mask_d_result_4x32b = _mm_madd_epi16(mask_d_result_4x32b, temp_coef1_8x16b);

        temp_coef0_8x16b = _mm_unpacklo_epi16(temp_coef0_8x16b, temp_coef1_8x16b);

        mask_de_result_8x16b = _mm_madd_epi16(mask_de_result_8x16b, temp_coef0_8x16b);
        mask_de1_result_8x16b = _mm_madd_epi16(mask_de1_result_8x16b, temp_coef0_8x16b);

        mask_d_result_4x32b = _mm_abs_epi32(mask_d_result_4x32b);
        mask_16x8b = _mm_shuffle_epi32(mask_d_result_4x32b, 0xec);
        mask_d_result_4x32b = _mm_shuffle_epi32(mask_d_result_4x32b, 0x49);

        mask_d_result_4x32b = _mm_add_epi32(mask_d_result_4x32b, mask_16x8b);

        mask_de_result_8x16b = _mm_packs_epi32(mask_de_result_8x16b, mask_de1_result_8x16b);
        mask_de_result_8x16b = _mm_abs_epi16(mask_de_result_8x16b);
        mask_de_result_8x16b = _mm_madd_epi16(mask_de_result_8x16b, temp_coef1_8x16b);

        temp_coef0_8x16b = _mm_srli_si128(mask_d_result_4x32b, 4);
        temp_coef1_8x16b = _mm_srli_si128(mask_d_result_4x32b, 8);
        mask_16x8b = _mm_srli_si128(mask_d_result_4x32b, 12);

        d0 = _mm_cvtsi128_si32(mask_d_result_4x32b);
        d3 = _mm_cvtsi128_si32(temp_coef0_8x16b);
        dp = _mm_cvtsi128_si32(temp_coef1_8x16b);
        dq = _mm_cvtsi128_si32(mask_16x8b);

        d = d0 + d3;

        temp_coef0_8x16b = _mm_srli_si128(mask_de_result_8x16b, 4);
        temp_coef1_8x16b = _mm_srli_si128(mask_de_result_8x16b, 8);
        mask_16x8b = _mm_srli_si128(mask_de_result_8x16b, 12);

        de_0 = _mm_cvtsi128_si32(mask_de_result_8x16b);
        de_1 = _mm_cvtsi128_si32(temp_coef0_8x16b);
        de_2 = _mm_cvtsi128_si32(temp_coef1_8x16b);
        de_3 = _mm_cvtsi128_si32(mask_16x8b);

        de = 0;
        dep = 0;
        deq = 0;
        if(d < beta)
        {
            d_sam0 = 0;
            if((2 * d0 < (beta >> 2))
                && (de_2 < (beta >> 3))
                && (de_0 < ((5 * tc + 1) >> 1)))
            {
                d_sam0 = 1;
            }

            d_sam3 = 0;
            if((2 * d3 < (beta >> 2))
                && (de_3 < (beta >> 3))
                && de_1 < ((5 * tc + 1) >> 1))
            {
                d_sam3 = 1;
            }

            de = (d_sam0 & d_sam3) + 1;
            dep = (dp < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
            deq = (dq < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
            if(tc <= 1)
            {
                dep = 0;
                deq = 0;
            }
        }
    }

    if(de != 0)
    {
        src_row1_8x16b = _mm_loadu_si128((__m128i *)((pu2_src - 4) + src_strd));
        src_row2_8x16b = _mm_loadu_si128((__m128i *)((pu2_src - 4) + 2 * src_strd));

        if(de == 2)
        {
            __m128i temp_pq_str0_16x8b, temp_pq_str1_16x8b, temp_pq_str2_16x8b, temp_pq_str3_16x8b;
            __m128i temp_pq1_str0_16x8b, temp_pq1_str1_16x8b, temp_pq1_str2_16x8b, temp_pq1_str3_16x8b;
            __m128i temp_pq2_str0_16x8b;
            __m128i temp_str0_16x8b, temp_str1_16x8b, temp_str2_16x8b, temp_str3_16x8b;
            __m128i temp_max0_16x8b, temp_max1_16x8b, temp_max2_16x8b, temp_max3_16x8b;
            __m128i temp_min0_16x8b, temp_min1_16x8b, temp_min2_16x8b, temp_min3_16x8b;
            __m128i const2_8x16b, const2tc_8x16b;
            ULWORD64 mask, tc2;
            tc = tc << 1;
            mask = (((ULWORD64)filter_flag_q) << 63) | (((ULWORD64)filter_flag_p) << 31);
            tc2 = ((ULWORD64)tc);

            const2_8x16b = _mm_cmpeq_epi16(src_row0_8x16b, src_row0_8x16b);
            const2_8x16b = _mm_srli_epi16(const2_8x16b, 15);

            temp_pq_str0_16x8b = _mm_srli_si128(src_row0_8x16b, 4);
            temp_pq_str1_16x8b = _mm_srli_si128(src_row1_8x16b, 4);
            temp_pq_str2_16x8b = _mm_srli_si128(src_row2_8x16b, 4);
            temp_pq_str3_16x8b = _mm_srli_si128(src_row3_8x16b, 4);

            temp_pq_str0_16x8b = _mm_unpacklo_epi32(temp_pq_str0_16x8b, temp_pq_str1_16x8b);
            temp_pq_str1_16x8b = _mm_unpacklo_epi32(temp_pq_str2_16x8b, temp_pq_str3_16x8b);

            temp_pq_str0_16x8b = _mm_madd_epi16(temp_pq_str0_16x8b, const2_8x16b);
            temp_pq_str1_16x8b = _mm_madd_epi16(temp_pq_str1_16x8b, const2_8x16b);

            temp_pq_str0_16x8b = _mm_packus_epi32(temp_pq_str0_16x8b, temp_pq_str1_16x8b);
            temp_pq_str0_16x8b = _mm_shuffle_epi32(temp_pq_str0_16x8b, 0xd8);

            temp_pq1_str0_16x8b = _mm_srli_si128(src_row0_8x16b, 2);
            temp_pq1_str1_16x8b = _mm_srli_si128(src_row1_8x16b, 2);
            temp_pq1_str2_16x8b = _mm_srli_si128(src_row2_8x16b, 2);
            temp_pq1_str3_16x8b = _mm_srli_si128(src_row3_8x16b, 2);

            temp_str0_16x8b = _mm_unpacklo_epi32(temp_pq1_str0_16x8b, temp_pq1_str1_16x8b);
            temp_str1_16x8b = _mm_unpacklo_epi32(temp_pq1_str2_16x8b, temp_pq1_str3_16x8b);
            temp_str2_16x8b = _mm_unpackhi_epi32(temp_pq1_str0_16x8b, temp_pq1_str1_16x8b);
            temp_str3_16x8b = _mm_unpackhi_epi32(temp_pq1_str2_16x8b, temp_pq1_str3_16x8b);
            temp_str2_16x8b = _mm_unpacklo_epi64(temp_str2_16x8b, temp_str3_16x8b);

            temp_pq1_str0_16x8b = _mm_madd_epi16(temp_str0_16x8b, const2_8x16b);
            temp_pq1_str1_16x8b = _mm_madd_epi16(temp_str1_16x8b, const2_8x16b);
            temp_pq1_str2_16x8b = _mm_unpacklo_epi64(temp_pq1_str0_16x8b, temp_pq1_str1_16x8b);
            temp_pq1_str3_16x8b = _mm_unpackhi_epi64(temp_pq1_str0_16x8b, temp_pq1_str1_16x8b);
            temp_pq1_str1_16x8b = _mm_madd_epi16(temp_str2_16x8b, const2_8x16b);

            temp_pq1_str0_16x8b = _mm_packus_epi32(temp_pq1_str3_16x8b, temp_pq1_str3_16x8b);
            temp_pq1_str1_16x8b = _mm_packus_epi32(temp_pq1_str2_16x8b, temp_pq1_str1_16x8b);

            temp_str1_16x8b = _mm_slli_epi16(const2_8x16b, 8);
            temp_str0_16x8b = _mm_loadl_epi64((__m128i *)(&mask));
            const2tc_8x16b  = _mm_loadl_epi64((__m128i *)(&tc2));
            temp_str0_16x8b = _mm_shuffle_epi32(temp_str0_16x8b, 0x50);
            const2tc_8x16b  = _mm_shuffle_epi8(const2tc_8x16b, temp_str1_16x8b);

            temp_str0_16x8b = _mm_srai_epi32(temp_str0_16x8b, 31);
            const2tc_8x16b = _mm_and_si128(const2tc_8x16b, temp_str0_16x8b);

            temp_max0_16x8b = _mm_adds_epu16(src_row0_8x16b, const2tc_8x16b);
            temp_max1_16x8b = _mm_adds_epu16(src_row1_8x16b, const2tc_8x16b);
            temp_max2_16x8b = _mm_adds_epu16(src_row2_8x16b, const2tc_8x16b);
            temp_max3_16x8b = _mm_adds_epu16(src_row3_8x16b, const2tc_8x16b);

            temp_pq2_str0_16x8b = _mm_unpacklo_epi32(src_row0_8x16b, src_row1_8x16b);
            temp_str3_16x8b     = _mm_unpacklo_epi32(src_row2_8x16b, src_row3_8x16b);
            temp_str1_16x8b     = _mm_unpacklo_epi64(temp_pq2_str0_16x8b, temp_str3_16x8b);

            const2_8x16b = _mm_slli_epi16(const2_8x16b, 1);
            temp_pq2_str0_16x8b = _mm_madd_epi16(temp_str1_16x8b, const2_8x16b);

            temp_str2_16x8b     = _mm_unpackhi_epi32(src_row0_8x16b, src_row1_8x16b);
            temp_str3_16x8b     = _mm_unpackhi_epi32(src_row2_8x16b, src_row3_8x16b);
            temp_str3_16x8b     = _mm_unpackhi_epi64(temp_str2_16x8b, temp_str3_16x8b);
            temp_str2_16x8b     = _mm_madd_epi16(temp_str3_16x8b, const2_8x16b);

            temp_pq2_str0_16x8b = _mm_packus_epi32(temp_pq2_str0_16x8b, temp_str2_16x8b);

            temp_min0_16x8b = _mm_subs_epu16(src_row0_8x16b, const2tc_8x16b);
            temp_min1_16x8b = _mm_subs_epu16(src_row1_8x16b, const2tc_8x16b);
            temp_min2_16x8b = _mm_subs_epu16(src_row2_8x16b, const2tc_8x16b);
            temp_min3_16x8b = _mm_subs_epu16(src_row3_8x16b, const2tc_8x16b);

            temp_pq_str1_16x8b = _mm_srli_si128(temp_pq_str0_16x8b, 8);
            temp_pq_str0_16x8b = _mm_add_epi16(temp_pq_str0_16x8b, temp_pq_str1_16x8b);
            temp_pq_str0_16x8b = _mm_unpacklo_epi64(temp_pq_str0_16x8b, temp_pq_str0_16x8b);

            temp_pq1_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, temp_pq1_str1_16x8b);

            temp_str1_16x8b = _mm_slli_epi32(temp_str1_16x8b, 16);
            temp_str3_16x8b = _mm_srli_epi32(temp_str3_16x8b, 16);
            temp_str1_16x8b = _mm_srli_epi32(temp_str1_16x8b, 16);
            temp_str1_16x8b = _mm_packus_epi32(temp_str1_16x8b, temp_str3_16x8b);

            temp_pq1_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, const2_8x16b);
            temp_pq_str0_16x8b = _mm_add_epi16(temp_pq_str0_16x8b, const2_8x16b);
            temp_pq2_str0_16x8b = _mm_add_epi16(temp_pq2_str0_16x8b, const2_8x16b);
            temp_pq_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, temp_pq_str0_16x8b);
            temp_pq2_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, temp_pq2_str0_16x8b);

            temp_pq_str0_16x8b  = _mm_srai_epi16(temp_pq_str0_16x8b, 3);
            temp_pq1_str0_16x8b = _mm_srai_epi16(temp_pq1_str0_16x8b, 2);
            temp_pq2_str0_16x8b = _mm_srai_epi16(temp_pq2_str0_16x8b, 3);

            temp_str0_16x8b = _mm_unpacklo_epi16(temp_pq1_str0_16x8b, temp_pq_str0_16x8b);
            temp_str2_16x8b = _mm_unpacklo_epi16(temp_str1_16x8b, temp_pq2_str0_16x8b);
            temp_pq_str0_16x8b = _mm_unpackhi_epi16(temp_pq_str0_16x8b, temp_pq1_str0_16x8b);
            temp_pq2_str0_16x8b = _mm_unpackhi_epi16(temp_pq2_str0_16x8b, temp_str1_16x8b);
            temp_pq_str1_16x8b = _mm_unpacklo_epi32(temp_str2_16x8b, temp_str0_16x8b);
            temp_str2_16x8b    = _mm_unpackhi_epi32(temp_str2_16x8b, temp_str0_16x8b);
            temp_str0_16x8b    = _mm_unpacklo_epi32(temp_pq_str0_16x8b, temp_pq2_str0_16x8b);
            temp_pq_str0_16x8b = _mm_unpackhi_epi32(temp_pq_str0_16x8b, temp_pq2_str0_16x8b);

            src_row0_8x16b = _mm_unpacklo_epi64(temp_pq_str1_16x8b, temp_str0_16x8b);
            src_row1_8x16b = _mm_unpackhi_epi64(temp_pq_str1_16x8b, temp_str0_16x8b);
            src_row2_8x16b = _mm_unpacklo_epi64(temp_str2_16x8b, temp_pq_str0_16x8b);
            src_row3_8x16b = _mm_unpackhi_epi64(temp_str2_16x8b, temp_pq_str0_16x8b);

            src_row0_8x16b = _mm_min_epu16(src_row0_8x16b, temp_max0_16x8b);
            src_row1_8x16b = _mm_min_epu16(src_row1_8x16b, temp_max1_16x8b);
            src_row2_8x16b = _mm_min_epu16(src_row2_8x16b, temp_max2_16x8b);
            src_row3_8x16b = _mm_min_epu16(src_row3_8x16b, temp_max3_16x8b);

            src_row0_8x16b = _mm_max_epu16(src_row0_8x16b, temp_min0_16x8b);
            src_row1_8x16b = _mm_max_epu16(src_row1_8x16b, temp_min1_16x8b);
            src_row2_8x16b = _mm_max_epu16(src_row2_8x16b, temp_min2_16x8b);
            src_row3_8x16b = _mm_max_epu16(src_row3_8x16b, temp_min3_16x8b);
        }
        else
        {
            __m128i tmp_delta0_8x16b, tmp_delta1_8x16b, tmp_delta2_8x16b, tmp_delta3_8x16b;
            __m128i tmp0_const_8x16b, tmp1_const_8x16b, tmp2_const_8x16b, tmp3_const_8x16b;
            __m128i coefdelta_0_8x16b, mask_pq_8x16b;
            __m128i const2_8x16b, consttc_8x16b;

            ULWORD64 mask1;
            mask1 = (((ULWORD64)(filter_flag_q & deq)) << 63) | (((ULWORD64)filter_flag_q) << 47) |
                    (((ULWORD64)filter_flag_p) << 31) | (((ULWORD64)(filter_flag_p & dep)) << 15);

            consttc_8x16b = _mm_set1_epi32(tc);

            tmp_delta0_8x16b = _mm_srli_si128(src_row0_8x16b, 4);
            tmp_delta1_8x16b = _mm_srli_si128(src_row1_8x16b, 4);
            tmp_delta2_8x16b = _mm_srli_si128(src_row2_8x16b, 4);
            tmp_delta3_8x16b = _mm_srli_si128(src_row3_8x16b, 4);

            tmp_delta0_8x16b = _mm_unpacklo_epi32(tmp_delta0_8x16b, tmp_delta1_8x16b);
            tmp_delta3_8x16b = _mm_unpacklo_epi32(tmp_delta2_8x16b, tmp_delta3_8x16b);

            tmp_delta1_8x16b = _mm_unpacklo_epi64(tmp_delta0_8x16b, tmp_delta3_8x16b);
            tmp_delta2_8x16b = _mm_unpackhi_epi64(tmp_delta0_8x16b, tmp_delta3_8x16b);

            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_de1_1);
            tmp_delta3_8x16b = _mm_madd_epi16(tmp_delta2_8x16b, coefdelta_0_8x16b);

            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_de1_2);
            tmp_delta0_8x16b = _mm_madd_epi16(tmp_delta1_8x16b, coefdelta_0_8x16b);

            consttc_8x16b = _mm_packs_epi32(consttc_8x16b, consttc_8x16b);
            tmp1_const_8x16b = _mm_cmpeq_epi32(consttc_8x16b, consttc_8x16b);
            tmp2_const_8x16b = _mm_slli_epi16(consttc_8x16b, 1);
            tmp0_const_8x16b = _mm_slli_epi16(consttc_8x16b, 3);
            tmp3_const_8x16b = _mm_sign_epi16(consttc_8x16b, tmp1_const_8x16b);
            tmp2_const_8x16b = _mm_add_epi16(tmp2_const_8x16b, tmp0_const_8x16b);

            tmp_delta0_8x16b = _mm_add_epi32(tmp_delta0_8x16b, tmp_delta3_8x16b);
            const2_8x16b = _mm_srli_epi32(tmp1_const_8x16b, 31);

            mask_pq_8x16b = _mm_loadl_epi64((__m128i *)(&mask1));
            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_dep1_1);
            tmp_delta1_8x16b = _mm_madd_epi16(tmp_delta1_8x16b, coefdelta_0_8x16b);

            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_dep1_2);
            tmp_delta2_8x16b = _mm_madd_epi16(tmp_delta2_8x16b, coefdelta_0_8x16b);

            tmp_delta3_8x16b = _mm_packs_epi32(tmp_delta1_8x16b, tmp_delta2_8x16b);
            const2_8x16b = _mm_slli_epi32(const2_8x16b, 3);
            mask_pq_8x16b = _mm_unpacklo_epi64(mask_pq_8x16b, mask_pq_8x16b);

            tmp_delta0_8x16b = _mm_add_epi32(tmp_delta0_8x16b, const2_8x16b);
            tmp_delta0_8x16b = _mm_srai_epi32(tmp_delta0_8x16b, 4);

            tmp_delta2_8x16b = _mm_sign_epi32(tmp_delta0_8x16b, tmp1_const_8x16b);
            tmp_delta0_8x16b = _mm_packs_epi32(tmp_delta0_8x16b, tmp_delta2_8x16b);
            tmp_delta2_8x16b = _mm_abs_epi16(tmp_delta0_8x16b);
            tmp_delta0_8x16b = _mm_min_epi16(tmp_delta0_8x16b, consttc_8x16b);
            tmp0_const_8x16b = _mm_cmpgt_epi16(tmp2_const_8x16b, tmp_delta2_8x16b);
            tmp_delta0_8x16b = _mm_max_epi16(tmp_delta0_8x16b, tmp3_const_8x16b);

            tmp2_const_8x16b = _mm_loadl_epi64((__m128i *)(shuffle0_hbd));
            tmp_delta1_8x16b = _mm_shuffle_epi8(src_row0_8x16b, tmp2_const_8x16b);
            tmp_delta2_8x16b = _mm_shuffle_epi8(src_row1_8x16b, tmp2_const_8x16b);
            tmp3_const_8x16b = _mm_unpacklo_epi32(tmp_delta1_8x16b, tmp_delta2_8x16b);

            tmp_delta1_8x16b = _mm_shuffle_epi8(src_row2_8x16b, tmp2_const_8x16b);
            tmp_delta2_8x16b = _mm_shuffle_epi8(src_row3_8x16b, tmp2_const_8x16b);
            tmp_delta1_8x16b = _mm_unpacklo_epi32(tmp_delta1_8x16b, tmp_delta2_8x16b);
            tmp_delta2_8x16b = _mm_unpacklo_epi64(tmp3_const_8x16b, tmp_delta1_8x16b);

            tmp3_const_8x16b = _mm_loadu_si128((__m128i *)(shuffle1_hbd));
            tmp_delta1_8x16b = _mm_shuffle_epi8(tmp_delta2_8x16b, tmp3_const_8x16b);
            const2_8x16b = _mm_srli_epi16(tmp1_const_8x16b, 15);
            consttc_8x16b = _mm_srai_epi16(consttc_8x16b, 1);

            tmp1_const_8x16b = _mm_sign_epi16(consttc_8x16b, tmp1_const_8x16b);
            tmp2_const_8x16b = _mm_add_epi16(tmp_delta0_8x16b, tmp_delta0_8x16b);

            tmp_delta3_8x16b = _mm_add_epi16(tmp_delta3_8x16b, const2_8x16b);
            tmp_delta1_8x16b = _mm_add_epi16(tmp_delta1_8x16b, tmp2_const_8x16b);
            tmp_delta1_8x16b = _mm_add_epi16(tmp_delta1_8x16b, tmp_delta3_8x16b);
            tmp_delta1_8x16b = _mm_srai_epi16(tmp_delta1_8x16b, 2);

            tmp_delta1_8x16b = _mm_min_epi16(tmp_delta1_8x16b, consttc_8x16b);
            tmp_delta1_8x16b = _mm_max_epi16(tmp_delta1_8x16b, tmp1_const_8x16b);

            mask_pq_8x16b = _mm_srai_epi16(mask_pq_8x16b, 15);
            tmp_delta1_8x16b = _mm_and_si128(tmp_delta1_8x16b, tmp0_const_8x16b);
            tmp_delta0_8x16b = _mm_and_si128(tmp_delta0_8x16b, tmp0_const_8x16b);

            tmp1_const_8x16b = _mm_unpacklo_epi16(tmp_delta1_8x16b, tmp_delta0_8x16b);
            tmp_delta0_8x16b = _mm_unpackhi_epi16(tmp_delta0_8x16b, tmp_delta1_8x16b);

            tmp_delta1_8x16b = _mm_unpackhi_epi32(tmp1_const_8x16b, tmp_delta0_8x16b);
            tmp_delta0_8x16b = _mm_unpacklo_epi32(tmp1_const_8x16b, tmp_delta0_8x16b);

            tmp_delta0_8x16b = _mm_and_si128(tmp_delta0_8x16b, mask_pq_8x16b);
            tmp_delta1_8x16b = _mm_and_si128(tmp_delta1_8x16b, mask_pq_8x16b);

            tmp0_const_8x16b = _mm_loadu_si128((__m128i *)shuffle2_hbd);
            tmp1_const_8x16b = _mm_loadu_si128((__m128i *)shuffle3_hbd);

            tmp_delta3_8x16b = _mm_shuffle_epi8(tmp_delta1_8x16b, tmp1_const_8x16b);
            tmp_delta2_8x16b = _mm_shuffle_epi8(tmp_delta1_8x16b, tmp0_const_8x16b);
            tmp_delta1_8x16b = _mm_shuffle_epi8(tmp_delta0_8x16b, tmp1_const_8x16b);
            tmp_delta0_8x16b = _mm_shuffle_epi8(tmp_delta0_8x16b, tmp0_const_8x16b);

            src_row3_8x16b = _mm_add_epi16(tmp_delta3_8x16b, src_row3_8x16b);
            src_row2_8x16b = _mm_add_epi16(tmp_delta2_8x16b, src_row2_8x16b);
            src_row1_8x16b = _mm_add_epi16(tmp_delta1_8x16b, src_row1_8x16b);
            src_row0_8x16b = _mm_add_epi16(tmp_delta0_8x16b, src_row0_8x16b);
        }

        clip_tmp_8x16b = _mm_cmpeq_epi16(src_row0_8x16b, src_row0_8x16b);
        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16 - bit_depth);
        src_row0_8x16b = _mm_min_epi16(src_row0_8x16b, clip_tmp_8x16b);
        src_row1_8x16b = _mm_min_epi16(src_row1_8x16b, clip_tmp_8x16b);
        src_row2_8x16b = _mm_min_epi16(src_row2_8x16b, clip_tmp_8x16b);
        src_row3_8x16b = _mm_min_epi16(src_row3_8x16b, clip_tmp_8x16b);

        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
        src_row0_8x16b = _mm_max_epi16(src_row0_8x16b, clip_tmp_8x16b);
        src_row1_8x16b = _mm_max_epi16(src_row1_8x16b, clip_tmp_8x16b);
        src_row2_8x16b = _mm_max_epi16(src_row2_8x16b, clip_tmp_8x16b);
        src_row3_8x16b = _mm_max_epi16(src_row3_8x16b, clip_tmp_8x16b);

        _mm_storeu_si128((__m128i *)(pu2_src - 4), src_row0_8x16b);
        _mm_storeu_si128((__m128i *)((pu2_src - 4) + src_strd), src_row1_8x16b);
        _mm_storeu_si128((__m128i *)((pu2_src - 4) + 2 * src_strd), src_row2_8x16b);
        _mm_storeu_si128((__m128i *)((pu2_src - 4) + 3 * src_strd), src_row3_8x16b);
    }
}

/**
*******************************************************************************
*
* @brief
*  Decision process and filtering for the luma block horizontal edge (HBD SSE4.2).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_luma_horz_sse42(UWORD16 *pu2_src,
                                     WORD32 src_strd,
                                     WORD32 bs,
                                     WORD32 quant_param_p,
                                     WORD32 quant_param_q,
                                     WORD32 beta_offset_div2,
                                     WORD32 tc_offset_div2,
                                     WORD32 filter_flag_p,
                                     WORD32 filter_flag_q,
                                     UWORD8 bit_depth)
{
    WORD32 qp_luma, beta_indx, tc_indx;
    WORD32 beta, tc;

    WORD32 d0, d3, dp, dq, d;
    WORD32 de_0, de_1, de_2, de_3;
    WORD32 d_sam0, d_sam3;
    WORD32 de, dep, deq;

    __m128i src_q0_8x16b, src_q1_8x16b, src_p0_8x16b, src_p1_8x16b, src_q2_8x16b;
    __m128i tmp_pq_str1_8x16b, src_p2_8x16b, tmp_pq_str0_8x16b;
    __m128i clip_tmp_8x16b;

    {
        __m128i src_tmp_p_0_8x16b, src_tmp_p_1_8x16b, src_tmp_q_0_8x16b, src_tmp_q_1_8x16b;
        __m128i coef_8x16b, mask_d_result_4x32b, mask_de_result_8x16b, mask_de1_result_8x16b;
        __m128i mask_16x8b, temp_coef0_8x16b, temp_coef1_8x16b;

        ASSERT((bs > 0));
        ASSERT(filter_flag_p || filter_flag_q);

        qp_luma = (quant_param_p + quant_param_q + 1) >> 1;
        beta_indx = CLIP3(qp_luma + (beta_offset_div2 << 1), 0, 51);

        tc_indx = CLIP3(qp_luma + 2 * (bs >> 1) + (tc_offset_div2 << 1), 0, 53);

        beta = gai4_ihevc_beta_table[beta_indx] * (1 << (bit_depth - 8));
        tc = gai4_ihevc_tc_table[tc_indx] * (1 << (bit_depth - 8));
        if(0 == tc)
        {
            return;
        }
        src_q0_8x16b = _mm_loadu_si128((__m128i *)(pu2_src));
        src_q1_8x16b = _mm_loadu_si128((__m128i *)(pu2_src + src_strd));
        src_p0_8x16b = _mm_loadu_si128((__m128i *)(pu2_src - src_strd));
        src_p1_8x16b = _mm_loadu_si128((__m128i *)(pu2_src - 2 * src_strd));
        src_q2_8x16b = _mm_loadu_si128((__m128i *)(pu2_src + 2 * src_strd));
        tmp_pq_str1_8x16b = _mm_loadu_si128((__m128i *)(pu2_src + 3 * src_strd));
        src_p2_8x16b = _mm_loadu_si128((__m128i *)(pu2_src - 3 * src_strd));
        tmp_pq_str0_8x16b = _mm_loadu_si128((__m128i *)(pu2_src - 4 * src_strd));

        src_tmp_p_0_8x16b = _mm_unpacklo_epi16(src_p1_8x16b, src_p0_8x16b);
        src_tmp_p_1_8x16b = _mm_unpacklo_epi16(tmp_pq_str0_8x16b, src_p2_8x16b);

        src_tmp_q_0_8x16b = _mm_unpacklo_epi16(src_q0_8x16b, src_q1_8x16b);
        src_tmp_q_1_8x16b = _mm_unpacklo_epi16(src_q2_8x16b, tmp_pq_str1_8x16b);

        src_tmp_p_0_8x16b = _mm_shuffle_epi32(src_tmp_p_0_8x16b, 0x6c);
        src_tmp_p_1_8x16b = _mm_shuffle_epi32(src_tmp_p_1_8x16b, 0x6c);
        src_tmp_q_0_8x16b = _mm_shuffle_epi32(src_tmp_q_0_8x16b, 0x6c);
        src_tmp_q_1_8x16b = _mm_shuffle_epi32(src_tmp_q_1_8x16b, 0x6c);

        src_tmp_p_1_8x16b = _mm_unpacklo_epi32(src_tmp_p_1_8x16b, src_tmp_p_0_8x16b);
        src_tmp_q_0_8x16b = _mm_unpacklo_epi32(src_tmp_q_0_8x16b, src_tmp_q_1_8x16b);

        src_tmp_p_0_8x16b = _mm_unpacklo_epi64(src_tmp_p_1_8x16b, src_tmp_q_0_8x16b);
        src_tmp_q_0_8x16b = _mm_unpackhi_epi64(src_tmp_p_1_8x16b, src_tmp_q_0_8x16b);

        coef_8x16b = _mm_loadu_si128((__m128i *)(coef_hbd_d));
        mask_16x8b = _mm_loadu_si128((__m128i *)(shuffle_hbd_d));

        src_tmp_p_1_8x16b     = _mm_shuffle_epi8(src_tmp_p_0_8x16b, mask_16x8b);
        mask_de1_result_8x16b = _mm_shuffle_epi8(src_tmp_q_0_8x16b, mask_16x8b);

        mask_de_result_8x16b = _mm_unpacklo_epi64(src_tmp_p_1_8x16b, mask_de1_result_8x16b);
        mask_de1_result_8x16b = _mm_unpackhi_epi64(src_tmp_p_1_8x16b, mask_de1_result_8x16b);

        mask_d_result_4x32b = _mm_madd_epi16(src_tmp_p_0_8x16b, coef_8x16b);
        src_tmp_q_0_8x16b = _mm_madd_epi16(src_tmp_q_0_8x16b, coef_8x16b);
        mask_d_result_4x32b = _mm_packs_epi32(mask_d_result_4x32b, src_tmp_q_0_8x16b);

        temp_coef0_8x16b = _mm_cmpeq_epi16(src_tmp_p_0_8x16b, src_tmp_p_0_8x16b);
        temp_coef1_8x16b = _mm_srli_epi16(temp_coef0_8x16b, 15);

        mask_d_result_4x32b = _mm_madd_epi16(mask_d_result_4x32b, temp_coef1_8x16b);

        temp_coef0_8x16b = _mm_unpacklo_epi16(temp_coef0_8x16b, temp_coef1_8x16b);

        mask_de_result_8x16b = _mm_madd_epi16(mask_de_result_8x16b, temp_coef0_8x16b);
        mask_de1_result_8x16b = _mm_madd_epi16(mask_de1_result_8x16b, temp_coef0_8x16b);

        temp_coef0_8x16b = _mm_srli_epi16(temp_coef1_8x16b, 8);

        mask_d_result_4x32b = _mm_abs_epi32(mask_d_result_4x32b);
        mask_16x8b = _mm_shuffle_epi32(mask_d_result_4x32b, 0xec);
        mask_d_result_4x32b = _mm_shuffle_epi32(mask_d_result_4x32b, 0x49);

        mask_d_result_4x32b = _mm_add_epi32(mask_d_result_4x32b, mask_16x8b);

        mask_de_result_8x16b = _mm_packs_epi32(mask_de_result_8x16b, mask_de1_result_8x16b);
        mask_de_result_8x16b = _mm_abs_epi16(mask_de_result_8x16b);
        mask_de_result_8x16b = _mm_madd_epi16(mask_de_result_8x16b, temp_coef1_8x16b);

        temp_coef0_8x16b = _mm_srli_si128(mask_d_result_4x32b, 4);
        temp_coef1_8x16b = _mm_srli_si128(mask_d_result_4x32b, 8);
        mask_16x8b = _mm_srli_si128(mask_d_result_4x32b, 12);

        d0 = _mm_cvtsi128_si32(mask_d_result_4x32b);
        d3 = _mm_cvtsi128_si32(temp_coef0_8x16b);
        dp = _mm_cvtsi128_si32(temp_coef1_8x16b);
        dq = _mm_cvtsi128_si32(mask_16x8b);

        d = d0 + d3;

        temp_coef0_8x16b = _mm_srli_si128(mask_de_result_8x16b, 4);
        temp_coef1_8x16b = _mm_srli_si128(mask_de_result_8x16b, 8);
        mask_16x8b = _mm_srli_si128(mask_de_result_8x16b, 12);

        de_0 = _mm_cvtsi128_si32(mask_de_result_8x16b);
        de_1 = _mm_cvtsi128_si32(temp_coef0_8x16b);
        de_2 = _mm_cvtsi128_si32(temp_coef1_8x16b);
        de_3 = _mm_cvtsi128_si32(mask_16x8b);

        de = 0;
        dep = 0;
        deq = 0;
        if(d < beta)
        {
            d_sam0 = 0;
            if((2 * d0 < (beta >> 2))
                && (de_2 < (beta >> 3))
                && (de_0 < ((5 * tc + 1) >> 1)))
            {
                d_sam0 = 1;
            }

            d_sam3 = 0;
            if((2 * d3 < (beta >> 2))
                && (de_3 < (beta >> 3))
                && de_1 < ((5 * tc + 1) >> 1))
            {
                d_sam3 = 1;
            }

            de = (d_sam0 & d_sam3) + 1;
            dep = (dp < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
            deq = (dq < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
            if(tc <= 1)
            {
                dep = 0;
                deq = 0;
            }
        }
    }

    if(de != 0)
    {
        if(2 == de)
        {
            __m128i temp_pq0_str0_16x8b;
            __m128i temp_pq1_str0_16x8b, temp_pq1_str1_16x8b;
            __m128i temp_pq2_str0_16x8b;
            __m128i temp_str0_16x8b, temp_str1_16x8b, temp_str2_16x8b, temp_str3_16x8b, temp_str4_16x8b, temp_str5_16x8b;
            __m128i const2_8x16b, const2tc_8x16b;

            ULWORD64 mask, tc2;
            tc = tc << 1;
            mask = (((ULWORD64)filter_flag_q) << 63) | (((ULWORD64)filter_flag_p) << 31);
            tc2 = ((ULWORD64)tc);

            const2_8x16b = _mm_cmpeq_epi16(src_p1_8x16b, src_p1_8x16b);
            const2_8x16b = _mm_srli_epi16(const2_8x16b, 15);

            temp_pq0_str0_16x8b = _mm_add_epi16(src_p0_8x16b, src_p1_8x16b);
            temp_str0_16x8b     = _mm_add_epi16(src_q0_8x16b, src_q1_8x16b);
            temp_pq0_str0_16x8b = _mm_unpacklo_epi64(temp_pq0_str0_16x8b, temp_str0_16x8b);

            temp_pq1_str1_16x8b = _mm_add_epi16(src_q1_8x16b, src_q2_8x16b);
            temp_str1_16x8b     = _mm_add_epi16(src_p1_8x16b, src_p2_8x16b);
            temp_pq1_str1_16x8b = _mm_unpacklo_epi64(temp_str1_16x8b, temp_pq1_str1_16x8b);

            temp_pq1_str0_16x8b = _mm_add_epi16(src_q0_8x16b, src_p0_8x16b);
            temp_pq1_str0_16x8b = _mm_unpacklo_epi64(temp_pq1_str0_16x8b, temp_pq1_str0_16x8b);

            temp_str1_16x8b = _mm_setzero_si128();
            temp_str0_16x8b = _mm_loadl_epi64((__m128i *)(&mask));
            const2tc_8x16b  = _mm_loadl_epi64((__m128i *)(&tc2));
            temp_str0_16x8b = _mm_shuffle_epi32(temp_str0_16x8b, 0x50);
            const2tc_8x16b  = _mm_shuffle_epi8(const2tc_8x16b, temp_str1_16x8b);

            const2tc_8x16b = _mm_srli_epi16(const2tc_8x16b, 8);
            temp_str0_16x8b = _mm_srai_epi32(temp_str0_16x8b, 31);
            const2tc_8x16b = _mm_and_si128(const2tc_8x16b, temp_str0_16x8b);

            temp_str0_16x8b = _mm_unpacklo_epi64(src_p0_8x16b, src_q0_8x16b);
            temp_str1_16x8b = _mm_unpacklo_epi64(src_p1_8x16b, src_q1_8x16b);

            temp_str2_16x8b = _mm_adds_epu16(temp_str0_16x8b, const2tc_8x16b);
            temp_str3_16x8b = _mm_adds_epu16(temp_str1_16x8b, const2tc_8x16b);
            temp_str0_16x8b = _mm_subs_epu16(temp_str0_16x8b, const2tc_8x16b);
            temp_str1_16x8b = _mm_subs_epu16(temp_str1_16x8b, const2tc_8x16b);

            const2_8x16b = _mm_slli_epi16(const2_8x16b, 1);
            tmp_pq_str0_8x16b = _mm_add_epi16(src_p2_8x16b, tmp_pq_str0_8x16b);
            temp_pq2_str0_16x8b = _mm_add_epi16(src_q2_8x16b, tmp_pq_str1_8x16b);
            temp_pq2_str0_16x8b = _mm_unpacklo_epi64(tmp_pq_str0_8x16b, temp_pq2_str0_16x8b);
            temp_pq2_str0_16x8b = _mm_slli_epi16(temp_pq2_str0_16x8b, 1);

            temp_str4_16x8b = _mm_unpacklo_epi64(src_p2_8x16b, src_q2_8x16b);

            temp_str5_16x8b = _mm_adds_epu16(temp_str4_16x8b, const2tc_8x16b);
            temp_str4_16x8b = _mm_subs_epu16(temp_str4_16x8b, const2tc_8x16b);

            tmp_pq_str0_8x16b = _mm_shuffle_epi32(temp_pq0_str0_16x8b, 0x4e);
            temp_pq0_str0_16x8b = _mm_add_epi16(temp_pq0_str0_16x8b, tmp_pq_str0_8x16b);
            temp_pq1_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, temp_pq1_str1_16x8b);

            temp_pq1_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, const2_8x16b);
            temp_pq0_str0_16x8b = _mm_add_epi16(temp_pq0_str0_16x8b, const2_8x16b);
            temp_pq2_str0_16x8b = _mm_add_epi16(temp_pq2_str0_16x8b, const2_8x16b);
            temp_pq0_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, temp_pq0_str0_16x8b);
            temp_pq2_str0_16x8b = _mm_add_epi16(temp_pq1_str0_16x8b, temp_pq2_str0_16x8b);

            temp_pq0_str0_16x8b = _mm_srai_epi16(temp_pq0_str0_16x8b, 3);
            temp_pq1_str0_16x8b = _mm_srai_epi16(temp_pq1_str0_16x8b, 2);
            temp_pq2_str0_16x8b = _mm_srai_epi16(temp_pq2_str0_16x8b, 3);

            temp_pq0_str0_16x8b = _mm_min_epu16(temp_pq0_str0_16x8b, temp_str2_16x8b);
            temp_pq1_str0_16x8b = _mm_min_epu16(temp_pq1_str0_16x8b, temp_str3_16x8b);
            temp_pq2_str0_16x8b = _mm_min_epu16(temp_pq2_str0_16x8b, temp_str5_16x8b);
            temp_pq0_str0_16x8b = _mm_max_epu16(temp_pq0_str0_16x8b, temp_str0_16x8b);
            temp_pq1_str0_16x8b = _mm_max_epu16(temp_pq1_str0_16x8b, temp_str1_16x8b);
            temp_pq2_str0_16x8b = _mm_max_epu16(temp_pq2_str0_16x8b, temp_str4_16x8b);

            clip_tmp_8x16b = _mm_cmpeq_epi16(temp_pq0_str0_16x8b, temp_pq0_str0_16x8b);
            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16 - bit_depth);
            temp_pq0_str0_16x8b = _mm_min_epi16(temp_pq0_str0_16x8b, clip_tmp_8x16b);
            temp_pq1_str0_16x8b = _mm_min_epi16(temp_pq1_str0_16x8b, clip_tmp_8x16b);
            temp_pq2_str0_16x8b = _mm_min_epi16(temp_pq2_str0_16x8b, clip_tmp_8x16b);

            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
            temp_pq0_str0_16x8b = _mm_max_epi16(temp_pq0_str0_16x8b, clip_tmp_8x16b);
            temp_pq1_str0_16x8b = _mm_max_epi16(temp_pq1_str0_16x8b, clip_tmp_8x16b);
            temp_pq2_str0_16x8b = _mm_max_epi16(temp_pq2_str0_16x8b, clip_tmp_8x16b);

            src_q0_8x16b = _mm_srli_si128(temp_pq0_str0_16x8b, 8);
            src_q1_8x16b = _mm_srli_si128(temp_pq1_str0_16x8b, 8);
            src_q2_8x16b = _mm_srli_si128(temp_pq2_str0_16x8b, 8);

            _mm_storel_epi64((__m128i *)(pu2_src - 3 * src_strd), temp_pq2_str0_16x8b);
            _mm_storel_epi64((__m128i *)(pu2_src - 2 * src_strd), temp_pq1_str0_16x8b);
            _mm_storel_epi64((__m128i *)(pu2_src - src_strd), temp_pq0_str0_16x8b);
            _mm_storel_epi64((__m128i *)(pu2_src), src_q0_8x16b);
            _mm_storel_epi64((__m128i *)(pu2_src + src_strd), src_q1_8x16b);
            _mm_storel_epi64((__m128i *)(pu2_src + 2 * src_strd), src_q2_8x16b);
        }
        else
        {
            __m128i tmp_delta0_8x16b, tmp_delta1_8x16b, tmp_delta2_8x16b, tmp_delta3_8x16b;
            __m128i tmp0_const_8x16b, tmp1_const_8x16b, tmp2_const_8x16b;
            __m128i coefdelta_0_8x16b;
            __m128i const2_8x16b, consttc_8x16b;

            LWORD64 maskp0, maskp1, maskq0, maskq1;
            maskp0 = (LWORD64)filter_flag_p;
            maskq0 = (LWORD64)filter_flag_q;
            maskp1 = (LWORD64)dep;
            maskq1 = (LWORD64)deq;
            consttc_8x16b = _mm_set1_epi32(tc);

            tmp_delta2_8x16b = _mm_unpacklo_epi16(src_p1_8x16b, src_p0_8x16b);
            tmp_delta3_8x16b = _mm_unpacklo_epi16(src_q0_8x16b, src_q1_8x16b);

            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_de1_2);
            tmp_delta0_8x16b = _mm_madd_epi16(tmp_delta2_8x16b, coefdelta_0_8x16b);
            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_de1_1);
            tmp_delta1_8x16b = _mm_madd_epi16(tmp_delta3_8x16b, coefdelta_0_8x16b);

            tmp2_const_8x16b = _mm_cmpeq_epi32(consttc_8x16b, consttc_8x16b);

            consttc_8x16b = _mm_packs_epi32(consttc_8x16b, consttc_8x16b);
            tmp_pq_str0_8x16b = _mm_slli_epi16(consttc_8x16b, 1);
            tmp_pq_str1_8x16b = _mm_slli_epi16(consttc_8x16b, 3);

            const2_8x16b = _mm_srli_epi16(tmp2_const_8x16b, 15);
            tmp_pq_str0_8x16b = _mm_add_epi16(tmp_pq_str0_8x16b, tmp_pq_str1_8x16b);
            tmp_delta0_8x16b = _mm_add_epi32(tmp_delta0_8x16b, tmp_delta1_8x16b);

            const2_8x16b = _mm_srli_epi32(tmp2_const_8x16b, 31);

            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_dep1_1);
            tmp_delta2_8x16b = _mm_madd_epi16(tmp_delta2_8x16b, coefdelta_0_8x16b);
            coefdelta_0_8x16b = _mm_loadu_si128((__m128i *)coef_hbd_dep1_2);
            tmp_delta1_8x16b = _mm_madd_epi16(tmp_delta3_8x16b, coefdelta_0_8x16b);
            tmp_delta1_8x16b = _mm_packs_epi32(tmp_delta2_8x16b, tmp_delta1_8x16b);

            const2_8x16b = _mm_slli_epi32(const2_8x16b, 3);

            tmp_delta0_8x16b = _mm_add_epi32(tmp_delta0_8x16b, const2_8x16b);
            tmp_delta0_8x16b = _mm_srai_epi32(tmp_delta0_8x16b, 4);

            tmp_pq_str1_8x16b = _mm_sign_epi32(tmp_delta0_8x16b, tmp2_const_8x16b);
            tmp1_const_8x16b = _mm_sign_epi16(consttc_8x16b, tmp2_const_8x16b);
            tmp_delta0_8x16b = _mm_packs_epi32(tmp_delta0_8x16b, tmp_pq_str1_8x16b);
            tmp_pq_str1_8x16b = _mm_abs_epi16(tmp_delta0_8x16b);

            tmp_delta0_8x16b = _mm_min_epi16(tmp_delta0_8x16b, consttc_8x16b);
            consttc_8x16b = _mm_srai_epi16(consttc_8x16b, 1);
            tmp_delta0_8x16b = _mm_max_epi16(tmp_delta0_8x16b, tmp1_const_8x16b);

            tmp1_const_8x16b = _mm_sign_epi16(consttc_8x16b, tmp2_const_8x16b);
            tmp_pq_str0_8x16b = _mm_cmpgt_epi16(tmp_pq_str0_8x16b, tmp_pq_str1_8x16b);

            tmp0_const_8x16b = _mm_setzero_si128();
            src_p2_8x16b = _mm_unpacklo_epi64(src_p2_8x16b, src_q2_8x16b);
            const2_8x16b = _mm_srli_epi16(tmp2_const_8x16b, 15);
            tmp2_const_8x16b = _mm_add_epi16(tmp_delta0_8x16b, tmp_delta0_8x16b);
            tmp_delta1_8x16b = _mm_add_epi16(tmp_delta1_8x16b, const2_8x16b);
            src_p2_8x16b = _mm_add_epi16(src_p2_8x16b, tmp2_const_8x16b);
            tmp_delta1_8x16b = _mm_add_epi16(tmp_delta1_8x16b, src_p2_8x16b);
            tmp_delta1_8x16b = _mm_srai_epi16(tmp_delta1_8x16b, 2);

            tmp_pq_str1_8x16b = _mm_loadl_epi64((__m128i *)(&(maskq0)));
            src_p2_8x16b = _mm_loadl_epi64((__m128i *)(&(maskp0)));

            src_q2_8x16b = _mm_loadl_epi64((__m128i *)(&(maskq1)));
            coefdelta_0_8x16b = _mm_loadl_epi64((__m128i *)(&(maskp1)));

            src_p2_8x16b = _mm_unpacklo_epi32(src_p2_8x16b, tmp_pq_str1_8x16b);
            src_q2_8x16b = _mm_unpacklo_epi32(coefdelta_0_8x16b, src_q2_8x16b);
            src_q2_8x16b = _mm_and_si128(src_q2_8x16b, src_p2_8x16b);

            src_q2_8x16b = _mm_shuffle_epi32(src_q2_8x16b, 0x50);
            src_p2_8x16b = _mm_shuffle_epi32(src_p2_8x16b, 0x50);

            src_q2_8x16b = _mm_slli_epi32(src_q2_8x16b, 31);
            src_p2_8x16b = _mm_slli_epi32(src_p2_8x16b, 31);
            src_q2_8x16b = _mm_srai_epi32(src_q2_8x16b, 31);
            src_p2_8x16b = _mm_srai_epi32(src_p2_8x16b, 31);

            tmp_pq_str1_8x16b = _mm_and_si128(tmp_pq_str0_8x16b, src_q2_8x16b);
            tmp_delta1_8x16b = _mm_min_epi16(tmp_delta1_8x16b, consttc_8x16b);
            tmp_pq_str0_8x16b = _mm_and_si128(tmp_pq_str0_8x16b, src_p2_8x16b);
            tmp_delta1_8x16b = _mm_max_epi16(tmp_delta1_8x16b, tmp1_const_8x16b);

            tmp_delta1_8x16b = _mm_and_si128(tmp_delta1_8x16b, tmp_pq_str1_8x16b);
            tmp_delta0_8x16b = _mm_and_si128(tmp_delta0_8x16b, tmp_pq_str0_8x16b);

            tmp_pq_str0_8x16b = _mm_unpacklo_epi64(tmp_delta0_8x16b, tmp0_const_8x16b);
            tmp_pq_str1_8x16b = _mm_unpackhi_epi64(tmp_delta0_8x16b, tmp0_const_8x16b);

            src_p0_8x16b = _mm_add_epi16(src_p0_8x16b, tmp_pq_str0_8x16b);
            src_q0_8x16b = _mm_add_epi16(src_q0_8x16b, tmp_pq_str1_8x16b);

            tmp_pq_str0_8x16b = _mm_unpacklo_epi64(tmp_delta1_8x16b, tmp0_const_8x16b);
            tmp_pq_str1_8x16b = _mm_unpackhi_epi64(tmp_delta1_8x16b, tmp0_const_8x16b);

            src_p1_8x16b = _mm_add_epi16(src_p1_8x16b, tmp_pq_str0_8x16b);
            src_q1_8x16b = _mm_add_epi16(src_q1_8x16b, tmp_pq_str1_8x16b);

            clip_tmp_8x16b = _mm_cmpeq_epi16(src_p0_8x16b, src_p0_8x16b);
            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16 - bit_depth);
            src_p0_8x16b = _mm_min_epi16(src_p0_8x16b, clip_tmp_8x16b);
            src_q0_8x16b = _mm_min_epi16(src_q0_8x16b, clip_tmp_8x16b);
            src_p1_8x16b = _mm_min_epi16(src_p1_8x16b, clip_tmp_8x16b);
            src_q1_8x16b = _mm_min_epi16(src_q1_8x16b, clip_tmp_8x16b);

            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
            src_p0_8x16b = _mm_max_epi16(src_p0_8x16b, clip_tmp_8x16b);
            src_q0_8x16b = _mm_max_epi16(src_q0_8x16b, clip_tmp_8x16b);
            src_p1_8x16b = _mm_max_epi16(src_p1_8x16b, clip_tmp_8x16b);
            src_q1_8x16b = _mm_max_epi16(src_q1_8x16b, clip_tmp_8x16b);

            _mm_storeu_si128((__m128i *)(pu2_src - 2 * src_strd), src_p1_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_src - src_strd), src_p0_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_src), src_q0_8x16b);
            _mm_storeu_si128((__m128i *)(pu2_src + src_strd), src_q1_8x16b);
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Decision process and filtering for the chroma block vertical edge (HBD SSE4.2).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_chroma_vert_sse42(UWORD16 *pu2_src,
                                       WORD32 src_strd,
                                       WORD32 quant_param_p,
                                       WORD32 quant_param_q,
                                       WORD32 qp_offset_u,
                                       WORD32 qp_offset_v,
                                       WORD32 tc_offset_div2,
                                       WORD32 filter_flag_p,
                                       WORD32 filter_flag_q,
                                       UWORD8 bit_depth,
                                       WORD8 chroma_fmt_idc)
{
    WORD32 qp_indx_u, qp_chroma_u;
    WORD32 qp_indx_v, qp_chroma_v;
    WORD32 tc_indx_u, tc_u;
    WORD32 tc_indx_v, tc_v;

    __m128i src_row_0_16x8b, tmp_pxl_0_16x8b, src_row_2_16x8b, tmp_pxl_1_16x8b;
    __m128i tmp_pxl_2_16x8b, tmp_pxl_3_16x8b, tmp_pxl_4_16x8b, tmp_pxl_5_16x8b;
    __m128i clip_tmp_8x16b;
    ASSERT(filter_flag_p || filter_flag_q);

    /* chroma processing is done only if BS is 2 */
    /* this function is assumed to be called only if BS is 2 */
    qp_indx_u = qp_offset_u + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_indx_v = qp_offset_v + ((quant_param_p + quant_param_q + 1) >> 1);

    if(chroma_fmt_idc == CHROMA_FMT_IDC_YUV444 || chroma_fmt_idc == CHROMA_FMT_IDC_YUV422)
    {
        qp_chroma_u = MIN(qp_indx_u, 51);
        qp_chroma_v = MIN(qp_indx_v, 51);
    }
    else
    {
        qp_chroma_u = qp_indx_u < 0 ? qp_indx_u : (qp_indx_u > 57 ? qp_indx_u - 6 : gai4_ihevc_qp_table[qp_indx_u]);
        qp_chroma_v = qp_indx_v < 0 ? qp_indx_v : (qp_indx_v > 57 ? qp_indx_v - 6 : gai4_ihevc_qp_table[qp_indx_v]);
    }

    tc_indx_u = CLIP3(qp_chroma_u + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_u = gai4_ihevc_tc_table[tc_indx_u] * (1 << (bit_depth - 8));

    tc_indx_v = CLIP3(qp_chroma_v + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_v = gai4_ihevc_tc_table[tc_indx_v] * (1 << (bit_depth - 8));

    if(0 == tc_u && 0 == tc_v)
    {
        return;
    }
    src_row_0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 4));
    tmp_pxl_0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + src_strd - 4));
    src_row_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + 2 * src_strd - 4));
    tmp_pxl_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + 3 * src_strd - 4));

    {
        ULWORD64 mask_tc, mask_flag, mask;
        __m128i delta_vu0_16x8b, delta_vu1_16x8b, delta_vu2_16x8b, delta_vu3_16x8b;
        __m128i mask_tc_16x8, mask_16x8b, mask_flag_p_16x8b, mask_flag_q_16x8b;
        __m128i min_0_16x8b;
        __m128i const_16x8b;
        mask_flag = (((ULWORD64)filter_flag_p) << 31) | (((ULWORD64)filter_flag_q) << 63);
        mask_tc = (((ULWORD64)tc_v) << 16) | ((ULWORD64)tc_u);
        mask = 0x00000000ffffffffULL;

        delta_vu0_16x8b = _mm_unpacklo_epi64(src_row_0_16x8b, tmp_pxl_0_16x8b);
        delta_vu1_16x8b = _mm_unpacklo_epi64(src_row_2_16x8b, tmp_pxl_1_16x8b);
        delta_vu2_16x8b = _mm_unpackhi_epi64(src_row_0_16x8b, tmp_pxl_0_16x8b);
        delta_vu3_16x8b = _mm_unpackhi_epi64(src_row_2_16x8b, tmp_pxl_1_16x8b);

        mask_16x8b = _mm_loadu_si128((__m128i *)shuffle_uv_hbd);
        delta_vu0_16x8b = _mm_shuffle_epi8(delta_vu0_16x8b, mask_16x8b);
        delta_vu1_16x8b = _mm_shuffle_epi8(delta_vu1_16x8b, mask_16x8b);
        delta_vu2_16x8b = _mm_shuffle_epi8(delta_vu2_16x8b, mask_16x8b);
        delta_vu3_16x8b = _mm_shuffle_epi8(delta_vu3_16x8b, mask_16x8b);

        tmp_pxl_2_16x8b = _mm_loadu_si128((__m128i *)delta0_hbd);
        tmp_pxl_3_16x8b = _mm_loadu_si128((__m128i *)delta1_hbd);

        delta_vu0_16x8b = _mm_madd_epi16(delta_vu0_16x8b, tmp_pxl_2_16x8b);
        delta_vu1_16x8b = _mm_madd_epi16(delta_vu1_16x8b, tmp_pxl_2_16x8b);
        delta_vu2_16x8b = _mm_madd_epi16(delta_vu2_16x8b, tmp_pxl_3_16x8b);
        delta_vu3_16x8b = _mm_madd_epi16(delta_vu3_16x8b, tmp_pxl_3_16x8b);

        delta_vu0_16x8b = _mm_packs_epi32(delta_vu0_16x8b, delta_vu1_16x8b);
        delta_vu1_16x8b = _mm_packs_epi32(delta_vu2_16x8b, delta_vu3_16x8b);

        const_16x8b = _mm_cmpeq_epi16(tmp_pxl_0_16x8b, tmp_pxl_0_16x8b);
        mask_tc_16x8 = _mm_loadl_epi64((__m128i *)(&mask_tc));
        mask_flag_q_16x8b = _mm_loadl_epi64((__m128i *)(&mask_flag));

        mask_tc_16x8 = _mm_shuffle_epi32(mask_tc_16x8, 0x00);
        mask_flag_q_16x8b = _mm_srai_epi32(mask_flag_q_16x8b, 31);
        min_0_16x8b = _mm_sign_epi16(mask_tc_16x8, const_16x8b);
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);

        mask_flag_p_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x00);
        mask_flag_q_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x55);

        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, delta_vu1_16x8b);
        const_16x8b = _mm_slli_epi16(const_16x8b, 2);

        mask_16x8b = _mm_loadl_epi64((__m128i *)(&mask));

        tmp_pxl_2_16x8b = _mm_srli_si128(src_row_0_16x8b, 4);
        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, const_16x8b);

        tmp_pxl_3_16x8b = _mm_srli_si128(tmp_pxl_0_16x8b, 4);
        tmp_pxl_2_16x8b = _mm_unpacklo_epi32(tmp_pxl_2_16x8b, tmp_pxl_3_16x8b);

        const_16x8b = _mm_setzero_si128();
        delta_vu0_16x8b = _mm_srai_epi16(delta_vu0_16x8b, 3);
        mask_16x8b = _mm_shuffle_epi32(mask_16x8b, 0x14);

        delta_vu0_16x8b = _mm_min_epi16(delta_vu0_16x8b, mask_tc_16x8);
        tmp_pxl_4_16x8b = _mm_srli_si128(src_row_2_16x8b, 4);
        delta_vu0_16x8b = _mm_max_epi16(delta_vu0_16x8b, min_0_16x8b);

        tmp_pxl_5_16x8b = _mm_srli_si128(tmp_pxl_1_16x8b, 4);
        tmp_pxl_4_16x8b = _mm_unpacklo_epi32(tmp_pxl_4_16x8b, tmp_pxl_5_16x8b);

        delta_vu1_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_q_16x8b);
        delta_vu0_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_p_16x8b);

        tmp_pxl_3_16x8b = _mm_unpackhi_epi64(tmp_pxl_2_16x8b, tmp_pxl_4_16x8b);
        tmp_pxl_2_16x8b = _mm_unpacklo_epi64(tmp_pxl_2_16x8b, tmp_pxl_4_16x8b);

        tmp_pxl_3_16x8b = _mm_sub_epi16(tmp_pxl_3_16x8b, delta_vu1_16x8b);
        tmp_pxl_2_16x8b = _mm_add_epi16(tmp_pxl_2_16x8b, delta_vu0_16x8b);
        delta_vu1_16x8b = _mm_unpackhi_epi32(tmp_pxl_2_16x8b, tmp_pxl_3_16x8b);
        delta_vu0_16x8b = _mm_unpacklo_epi32(tmp_pxl_2_16x8b, tmp_pxl_3_16x8b);

        mask_flag_q_16x8b = _mm_loadu_si128((__m128i *)shuffle_uv_hbd1);
        mask_flag_p_16x8b = _mm_loadu_si128((__m128i *)shuffle_uv_hbd2);

        tmp_pxl_2_16x8b = _mm_shuffle_epi8(delta_vu0_16x8b, mask_flag_q_16x8b);
        tmp_pxl_3_16x8b = _mm_shuffle_epi8(delta_vu0_16x8b, mask_flag_p_16x8b);
        tmp_pxl_4_16x8b = _mm_shuffle_epi8(delta_vu1_16x8b, mask_flag_q_16x8b);
        tmp_pxl_5_16x8b = _mm_shuffle_epi8(delta_vu1_16x8b, mask_flag_p_16x8b);

        src_row_0_16x8b = _mm_and_si128(src_row_0_16x8b, mask_16x8b);
        tmp_pxl_0_16x8b = _mm_and_si128(tmp_pxl_0_16x8b, mask_16x8b);
        src_row_2_16x8b = _mm_and_si128(src_row_2_16x8b, mask_16x8b);
        tmp_pxl_1_16x8b = _mm_and_si128(tmp_pxl_1_16x8b, mask_16x8b);

        src_row_0_16x8b = _mm_or_si128(src_row_0_16x8b, tmp_pxl_2_16x8b);
        tmp_pxl_0_16x8b = _mm_or_si128(tmp_pxl_0_16x8b, tmp_pxl_3_16x8b);
        src_row_2_16x8b = _mm_or_si128(src_row_2_16x8b, tmp_pxl_4_16x8b);
        tmp_pxl_1_16x8b = _mm_or_si128(tmp_pxl_1_16x8b, tmp_pxl_5_16x8b);

        clip_tmp_8x16b = _mm_cmpeq_epi16(src_row_0_16x8b, src_row_0_16x8b);
        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16 - bit_depth);
        src_row_0_16x8b = _mm_min_epi16(src_row_0_16x8b, clip_tmp_8x16b);
        tmp_pxl_0_16x8b = _mm_min_epi16(tmp_pxl_0_16x8b, clip_tmp_8x16b);
        src_row_2_16x8b = _mm_min_epi16(src_row_2_16x8b, clip_tmp_8x16b);
        tmp_pxl_1_16x8b = _mm_min_epi16(tmp_pxl_1_16x8b, clip_tmp_8x16b);

        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
        src_row_0_16x8b = _mm_max_epi16(src_row_0_16x8b, clip_tmp_8x16b);
        tmp_pxl_0_16x8b = _mm_max_epi16(tmp_pxl_0_16x8b, clip_tmp_8x16b);
        src_row_2_16x8b = _mm_max_epi16(src_row_2_16x8b, clip_tmp_8x16b);
        tmp_pxl_1_16x8b = _mm_max_epi16(tmp_pxl_1_16x8b, clip_tmp_8x16b);

        _mm_storeu_si128((__m128i *)(pu2_src - 4), src_row_0_16x8b);
        _mm_storeu_si128((__m128i *)((pu2_src - 4) + src_strd), tmp_pxl_0_16x8b);
        _mm_storeu_si128((__m128i *)((pu2_src - 4) + 2 * src_strd), src_row_2_16x8b);
        _mm_storeu_si128((__m128i *)((pu2_src - 4) + 3 * src_strd), tmp_pxl_1_16x8b);
    }
}

/**
*******************************************************************************
*
* @brief
*  Decision process and filtering for the chroma block horizontal edge (HBD SSE4.2).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_chroma_horz_sse42(UWORD16 *pu2_src,
                                       WORD32 src_strd,
                                       WORD32 quant_param_p,
                                       WORD32 quant_param_q,
                                       WORD32 qp_offset_u,
                                       WORD32 qp_offset_v,
                                       WORD32 tc_offset_div2,
                                       WORD32 filter_flag_p,
                                       WORD32 filter_flag_q,
                                       UWORD8 bit_depth,
                                       WORD8 chroma_fmt_idc)
{
    WORD32 qp_indx_u, qp_chroma_u;
    WORD32 qp_indx_v, qp_chroma_v;
    WORD32 tc_indx_u, tc_u;
    WORD32 tc_indx_v, tc_v;

    __m128i tmp_p0_16x8b, src_p0_16x8b, src_q0_16x8b, tmp_q0_16x8b;
    __m128i clip_tmp_8x16b;

    ASSERT(filter_flag_p || filter_flag_q);

    /* chroma processing is done only if BS is 2 */
    /* this function is assumed to be called only if BS is 2 */
    qp_indx_u = qp_offset_u + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_indx_v = qp_offset_v + ((quant_param_p + quant_param_q + 1) >> 1);

    if(chroma_fmt_idc == CHROMA_FMT_IDC_YUV444 || chroma_fmt_idc == CHROMA_FMT_IDC_YUV422)
    {
        qp_chroma_u = MIN(qp_indx_u, 51);
        qp_chroma_v = MIN(qp_indx_v, 51);
    }
    else
    {
        qp_chroma_u = qp_indx_u < 0 ? qp_indx_u : (qp_indx_u > 57 ? qp_indx_u - 6 : gai4_ihevc_qp_table[qp_indx_u]);
        qp_chroma_v = qp_indx_v < 0 ? qp_indx_v : (qp_indx_v > 57 ? qp_indx_v - 6 : gai4_ihevc_qp_table[qp_indx_v]);
    }

    tc_indx_u = CLIP3(qp_chroma_u + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_u = gai4_ihevc_tc_table[tc_indx_u] * (1 << (bit_depth - 8));

    tc_indx_v = CLIP3(qp_chroma_v + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_v = gai4_ihevc_tc_table[tc_indx_v] * (1 << (bit_depth - 8));

    if(0 == tc_u && 0 == tc_v)
    {
        return;
    }
    tmp_p0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 2 * src_strd));
    src_p0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - src_strd));
    src_q0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));
    tmp_q0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + src_strd));

    {
        ULWORD64 mask_tc, mask_flag;
        __m128i delta_vu0_16x8b, delta_vu1_16x8b;
        __m128i mask_tc_16x8, mask_flag_p_16x8b, mask_flag_q_16x8b;
        __m128i min_0_16x8b;
        __m128i const_16x8b;
        mask_flag = (((ULWORD64)filter_flag_p) << 31) | (((ULWORD64)filter_flag_q) << 63);
        mask_tc = (((ULWORD64)tc_v) << 16) | ((ULWORD64)tc_u);

        delta_vu0_16x8b = _mm_sub_epi16(tmp_p0_16x8b, tmp_q0_16x8b);
        delta_vu1_16x8b = _mm_sub_epi16(src_q0_16x8b, src_p0_16x8b);
        delta_vu1_16x8b = _mm_slli_epi16(delta_vu1_16x8b, 2);

        mask_tc_16x8 = _mm_loadl_epi64((__m128i *)(&mask_tc));
        mask_flag_q_16x8b = _mm_loadl_epi64((__m128i *)(&mask_flag));

        const_16x8b = _mm_cmpeq_epi16(tmp_p0_16x8b, tmp_p0_16x8b);
        mask_tc_16x8 = _mm_shuffle_epi32(mask_tc_16x8, 0x00);
        mask_flag_q_16x8b = _mm_srai_epi32(mask_flag_q_16x8b, 31);
        min_0_16x8b = _mm_sign_epi16(mask_tc_16x8, const_16x8b);
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);

        mask_flag_p_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x00);

        const_16x8b = _mm_slli_epi16(const_16x8b, 2);
        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, delta_vu1_16x8b);

        mask_flag_q_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x55);
        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, const_16x8b);
        delta_vu0_16x8b = _mm_srai_epi16(delta_vu0_16x8b, 3);

        delta_vu0_16x8b = _mm_min_epi16(delta_vu0_16x8b, mask_tc_16x8);
        delta_vu0_16x8b = _mm_max_epi16(delta_vu0_16x8b, min_0_16x8b);

        delta_vu1_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_q_16x8b);
        delta_vu0_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_p_16x8b);

        src_q0_16x8b = _mm_sub_epi16(src_q0_16x8b, delta_vu1_16x8b);
        src_p0_16x8b = _mm_add_epi16(src_p0_16x8b, delta_vu0_16x8b);

        clip_tmp_8x16b = _mm_cmpeq_epi16(src_q0_16x8b, src_q0_16x8b);
        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16 - bit_depth);
        src_q0_16x8b = _mm_min_epi16(src_q0_16x8b, clip_tmp_8x16b);
        src_p0_16x8b = _mm_min_epi16(src_p0_16x8b, clip_tmp_8x16b);

        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
        src_q0_16x8b = _mm_max_epi16(src_q0_16x8b, clip_tmp_8x16b);
        src_p0_16x8b = _mm_max_epi16(src_p0_16x8b, clip_tmp_8x16b);

        _mm_storeu_si128((__m128i *)(pu2_src - src_strd), src_p0_16x8b);
        _mm_storeu_si128((__m128i *)(pu2_src), src_q0_16x8b);
    }
}
