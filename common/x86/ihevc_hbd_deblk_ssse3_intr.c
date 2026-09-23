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
*  ihevc_hbd_deblk_ssse3_intr.c
*
* @brief
*  Contains function definitions for high bit depth deblocking filters using
*  SSSE3 intrinsics.
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_deblk_chroma_vert_ssse3()
*  - ihevc_hbd_deblk_chroma_horz_ssse3()
*  - ihevc_hbd_deblk_luma_vert_ssse3()
*  - ihevc_hbd_deblk_luma_horz_ssse3()
*
* @remarks
*  None
*
*******************************************************************************
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <tmmintrin.h>

#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_deblk.h"
#include "ihevc_deblk_tables.h"
#include "ihevc_debug.h"
#include "ihevc_defs.h"

static inline void store_u32(void *dst, int val)
{
    memcpy(dst, &val, sizeof(UWORD32));
}

static inline __m128i select_vec(__m128i a, __m128i b, __m128i mask)
{
    /* returns b where mask is all 1s, a where mask is 0 */
    return _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a));
}

/**
*******************************************************************************
*
* @brief
*  SSSE3 implementation of chroma vertical edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_chroma_vert_ssse3(UWORD16 *pu2_src,
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

    ASSERT(filter_flag_p || filter_flag_q);

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

    /* Load 4 rows of 8 samples [-4..3] */
    __m128i r0 = _mm_loadu_si128((__m128i const *)(pu2_src - 4));
    __m128i r1 = _mm_loadu_si128((__m128i const *)(pu2_src + src_strd - 4));
    __m128i r2 = _mm_loadu_si128((__m128i const *)(pu2_src + 2 * src_strd - 4));
    __m128i r3 = _mm_loadu_si128((__m128i const *)(pu2_src + 3 * src_strd - 4));

    /* 4x4 32-bit transpose: separates p1, p0, q0, q1 for 4 rows */
    __m128 f0 = _mm_castsi128_ps(r0);
    __m128 f1 = _mm_castsi128_ps(r1);
    __m128 f2 = _mm_castsi128_ps(r2);
    __m128 f3 = _mm_castsi128_ps(r3);
    _MM_TRANSPOSE4_PS(f0, f1, f2, f3);
    __m128i vec_p1 = _mm_castps_si128(f0);
    __m128i vec_p0 = _mm_castps_si128(f1);
    __m128i vec_q0 = _mm_castps_si128(f2);
    __m128i vec_q1 = _mm_castps_si128(f3);

    __m128i vec_tc = _mm_setr_epi16(tc_u, tc_v, tc_u, tc_v, tc_u, tc_v, tc_u, tc_v);
    __m128i vec_neg_tc = _mm_sub_epi16(_mm_setzero_si128(), vec_tc);

    /* delta = (((q0 - p0) << 2) + p1 - q1 + 4) >> 3 */
    __m128i diff_q0_p0 = _mm_sub_epi16(vec_q0, vec_p0);
    __m128i diff_x4 = _mm_slli_epi16(diff_q0_p0, 2);
    __m128i diff_p1_q1 = _mm_sub_epi16(vec_p1, vec_q1);
    __m128i sum = _mm_add_epi16(diff_x4, diff_p1_q1);
    __m128i sum_plus_4 = _mm_add_epi16(sum, _mm_set1_epi16(4));
    __m128i delta = _mm_srai_epi16(sum_plus_4, 3);
    __m128i delta_clipped = _mm_min_epi16(_mm_max_epi16(delta, vec_neg_tc), vec_tc);

    __m128i max_val = _mm_set1_epi16((1 << bit_depth) - 1);
    __m128i new_p0 = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vec_p0, delta_clipped), _mm_setzero_si128()), max_val);
    __m128i new_q0 = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(vec_q0, delta_clipped), _mm_setzero_si128()), max_val);

    if(filter_flag_p && filter_flag_q)
    {
        __m128i lo = _mm_unpacklo_epi32(new_p0, new_q0);
        __m128i hi = _mm_unpackhi_epi32(new_p0, new_q0);
        _mm_storel_epi64((__m128i *)(pu2_src - 2), lo);
        _mm_storel_epi64((__m128i *)(pu2_src + src_strd - 2), _mm_srli_si128(lo, 8));
        _mm_storel_epi64((__m128i *)(pu2_src + 2 * src_strd - 2), hi);
        _mm_storel_epi64((__m128i *)(pu2_src + 3 * src_strd - 2), _mm_srli_si128(hi, 8));
    }
    else if(filter_flag_p)
    {
        store_u32(pu2_src - 2, _mm_cvtsi128_si32(new_p0));
        store_u32(pu2_src + src_strd - 2, _mm_cvtsi128_si32(_mm_srli_si128(new_p0, 4)));
        store_u32(pu2_src + 2 * src_strd - 2, _mm_cvtsi128_si32(_mm_srli_si128(new_p0, 8)));
        store_u32(pu2_src + 3 * src_strd - 2, _mm_cvtsi128_si32(_mm_srli_si128(new_p0, 12)));
    }
    else if(filter_flag_q)
    {
        store_u32(pu2_src, _mm_cvtsi128_si32(new_q0));
        store_u32(pu2_src + src_strd, _mm_cvtsi128_si32(_mm_srli_si128(new_q0, 4)));
        store_u32(pu2_src + 2 * src_strd, _mm_cvtsi128_si32(_mm_srli_si128(new_q0, 8)));
        store_u32(pu2_src + 3 * src_strd, _mm_cvtsi128_si32(_mm_srli_si128(new_q0, 12)));
    }
}

/**
*******************************************************************************
*
* @brief
*  SSSE3 implementation of chroma horizontal edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_chroma_horz_ssse3(UWORD16 *pu2_src,
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

    ASSERT(filter_flag_p || filter_flag_q);

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

    __m128i vec_p1 = _mm_loadu_si128((__m128i const *)(pu2_src - 2 * src_strd));
    __m128i vec_p0 = _mm_loadu_si128((__m128i const *)(pu2_src - 1 * src_strd));
    __m128i vec_q0 = _mm_loadu_si128((__m128i const *)(pu2_src));
    __m128i vec_q1 = _mm_loadu_si128((__m128i const *)(pu2_src + 1 * src_strd));

    __m128i vec_tc = _mm_setr_epi16(tc_u, tc_v, tc_u, tc_v, tc_u, tc_v, tc_u, tc_v);
    __m128i vec_neg_tc = _mm_sub_epi16(_mm_setzero_si128(), vec_tc);

    __m128i diff_q0_p0 = _mm_sub_epi16(vec_q0, vec_p0);
    __m128i diff_x4 = _mm_slli_epi16(diff_q0_p0, 2);
    __m128i diff_p1_q1 = _mm_sub_epi16(vec_p1, vec_q1);
    __m128i sum = _mm_add_epi16(diff_x4, diff_p1_q1);
    __m128i sum_plus_4 = _mm_add_epi16(sum, _mm_set1_epi16(4));
    __m128i delta = _mm_srai_epi16(sum_plus_4, 3);
    __m128i delta_clipped = _mm_min_epi16(_mm_max_epi16(delta, vec_neg_tc), vec_tc);

    __m128i max_val = _mm_set1_epi16((1 << bit_depth) - 1);
    __m128i new_p0 = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vec_p0, delta_clipped), _mm_setzero_si128()), max_val);
    __m128i new_q0 = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(vec_q0, delta_clipped), _mm_setzero_si128()), max_val);

    if(filter_flag_p)
    {
        _mm_storeu_si128((__m128i *)(pu2_src - src_strd), new_p0);
    }
    if(filter_flag_q)
    {
        _mm_storeu_si128((__m128i *)(pu2_src), new_q0);
    }
}


/**
*******************************************************************************
*
* @brief
*  Core decision process and filtering for luma edge (HBD).
*  Operates on 4 parallel columns or rows contained in lanes 0..3 of each register.
*
*******************************************************************************
*/
static inline WORD32 ihevc_hbd_deblk_luma_core(
                                               __m128i vec_p3, __m128i vec_p2, __m128i vec_p1, __m128i vec_p0,
                                               __m128i vec_q0, __m128i vec_q1, __m128i vec_q2, __m128i vec_q3,
                                               WORD32 beta, WORD32 tc,
                                               UWORD8 bit_depth,
                                               __m128i *p_new_p2, __m128i *p_new_p1, __m128i *p_new_p0,
                                               __m128i *p_new_q0, __m128i *p_new_q1, __m128i *p_new_q2)
{
    __m128i q2_minus_2q1_plus_q0 = _mm_sub_epi16(_mm_add_epi16(vec_q2, vec_q0), _mm_slli_epi16(vec_q1, 1));
    __m128i vec_dq = _mm_abs_epi16(q2_minus_2q1_plus_q0);

    __m128i p2_minus_2p1_plus_p0 = _mm_sub_epi16(_mm_add_epi16(vec_p2, vec_p0), _mm_slli_epi16(vec_p1, 1));
    __m128i vec_dp = _mm_abs_epi16(p2_minus_2p1_plus_p0);

    int dp0 = _mm_extract_epi16(vec_dp, 0);
    int dp3 = _mm_extract_epi16(vec_dp, 3);
    int dq0 = _mm_extract_epi16(vec_dq, 0);
    int dq3 = _mm_extract_epi16(vec_dq, 3);
    int d0 = dp0 + dq0;
    int d3 = dp3 + dq3;
    int dp = dp0 + dp3;
    int dq = dq0 + dq3;
    int d = d0 + d3;

    if(d >= beta)
    {
        return 0;
    }

    __m128i q3_minus_q0 = _mm_abs_epi16(_mm_sub_epi16(vec_q3, vec_q0));
    __m128i p0_minus_p3 = _mm_abs_epi16(_mm_sub_epi16(vec_p0, vec_p3));
    __m128i q0_minus_p0 = _mm_abs_epi16(_mm_sub_epi16(vec_q0, vec_p0));
    __m128i end_sum = _mm_add_epi16(q3_minus_q0, p0_minus_p3);

    int end_sum0 = _mm_extract_epi16(end_sum, 0);
    int end_sum3 = _mm_extract_epi16(end_sum, 3);
    int q0_p0_diff0 = _mm_extract_epi16(q0_minus_p0, 0);
    int q0_p0_diff3 = _mm_extract_epi16(q0_minus_p0, 3);

    int d_sam0 = 0;
    if((2 * d0 < (beta >> 2)) && (end_sum0 < (beta >> 3)) && (q0_p0_diff0 < ((5 * tc + 1) >> 1)))
    {
        d_sam0 = 1;
    }

    int d_sam3 = 0;
    if((2 * d3 < (beta >> 2)) && (end_sum3 < (beta >> 3)) && (q0_p0_diff3 < ((5 * tc + 1) >> 1)))
    {
        d_sam3 = 1;
    }

    int de = (d_sam0 == 1 && d_sam3 == 1) ? 2 : 1;
    int dep = (dp < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
    int deq = (dq < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
    if(tc <= 1)
    {
        dep = 0;
        deq = 0;
    }

    if(de == 2)
    {
        __m128i two_tc = _mm_set1_epi16(2 * tc);
        __m128i q1_q0_p0 = _mm_add_epi16(vec_q1, _mm_add_epi16(vec_q0, vec_p0));

        /* q0 = CLIP3((q2 + 2*q1 + 2*q0 + 2*p0 + p1 + 4) >> 3, q0 - 2*tc, q0 + 2*tc) */
        __m128i q0_sum = _mm_add_epi16(_mm_add_epi16(vec_q2, vec_p1),
                                       _mm_add_epi16(_mm_slli_epi16(q1_q0_p0, 1), _mm_set1_epi16(4)));
        __m128i q0_val = _mm_srai_epi16(q0_sum, 3);
        __m128i q0_min = _mm_sub_epi16(vec_q0, two_tc);
        __m128i q0_max = _mm_add_epi16(vec_q0, two_tc);
        *p_new_q0 = _mm_min_epi16(_mm_max_epi16(q0_val, q0_min), q0_max);

        /* q1 = CLIP3((q2 + q1 + q0 + p0 + 2) >> 2, q1 - 2*tc, q1 + 2*tc) */
        __m128i q1_sum = _mm_add_epi16(vec_q2, _mm_add_epi16(q1_q0_p0, _mm_set1_epi16(2)));
        __m128i q1_val = _mm_srai_epi16(q1_sum, 2);
        __m128i q1_min = _mm_sub_epi16(vec_q1, two_tc);
        __m128i q1_max = _mm_add_epi16(vec_q1, two_tc);
        *p_new_q1 = _mm_min_epi16(_mm_max_epi16(q1_val, q1_min), q1_max);

        /* q2 = CLIP3((2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3, q2 - 2*tc, q2 + 2*tc) */
        __m128i q3_q2_x2 = _mm_slli_epi16(_mm_add_epi16(vec_q3, vec_q2), 1);
        __m128i q2_sum = _mm_add_epi16(q3_q2_x2,
                                       _mm_add_epi16(vec_q2,
                                                     _mm_add_epi16(q1_q0_p0, _mm_set1_epi16(4))));
        __m128i q2_val = _mm_srai_epi16(q2_sum, 3);
        __m128i q2_min = _mm_sub_epi16(vec_q2, two_tc);
        __m128i q2_max = _mm_add_epi16(vec_q2, two_tc);
        *p_new_q2 = _mm_min_epi16(_mm_max_epi16(q2_val, q2_min), q2_max);

        /* p side */
        __m128i p1_p0_q0 = _mm_add_epi16(vec_p1, _mm_add_epi16(vec_p0, vec_q0));

        /* p0 = CLIP3((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3, p0 - 2*tc, p0 + 2*tc) */
        __m128i p0_sum = _mm_add_epi16(_mm_add_epi16(vec_p2, vec_q1),
                                       _mm_add_epi16(_mm_slli_epi16(p1_p0_q0, 1), _mm_set1_epi16(4)));
        __m128i p0_val = _mm_srai_epi16(p0_sum, 3);
        __m128i p0_min = _mm_sub_epi16(vec_p0, two_tc);
        __m128i p0_max = _mm_add_epi16(vec_p0, two_tc);
        *p_new_p0 = _mm_min_epi16(_mm_max_epi16(p0_val, p0_min), p0_max);

        /* p1 = CLIP3((p2 + p1 + p0 + q0 + 2) >> 2, p1 - 2*tc, p1 + 2*tc) */
        __m128i p1_sum = _mm_add_epi16(vec_p2, _mm_add_epi16(p1_p0_q0, _mm_set1_epi16(2)));
        __m128i p1_val = _mm_srai_epi16(p1_sum, 2);
        __m128i p1_min = _mm_sub_epi16(vec_p1, two_tc);
        __m128i p1_max = _mm_add_epi16(vec_p1, two_tc);
        *p_new_p1 = _mm_min_epi16(_mm_max_epi16(p1_val, p1_min), p1_max);

        /* p2 = CLIP3((2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3, p2 - 2*tc, p2 + 2*tc) */
        __m128i p3_p2_x2 = _mm_slli_epi16(_mm_add_epi16(vec_p3, vec_p2), 1);
        __m128i p2_sum = _mm_add_epi16(p3_p2_x2,
                                       _mm_add_epi16(vec_p2,
                                                     _mm_add_epi16(p1_p0_q0, _mm_set1_epi16(4))));
        __m128i p2_val = _mm_srai_epi16(p2_sum, 3);
        __m128i p2_min = _mm_sub_epi16(vec_p2, two_tc);
        __m128i p2_max = _mm_add_epi16(vec_p2, two_tc);
        *p_new_p2 = _mm_min_epi16(_mm_max_epi16(p2_val, p2_min), p2_max);
    }
    else
    {
        __m128i d_q0_p0 = _mm_sub_epi16(vec_q0, vec_p0);
        __m128i d_q1_p1 = _mm_sub_epi16(vec_q1, vec_p1);

        /* delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4 */
        __m128i term1 = _mm_add_epi16(_mm_slli_epi16(d_q0_p0, 3), d_q0_p0);
        __m128i term2 = _mm_add_epi16(_mm_slli_epi16(d_q1_p1, 1), d_q1_p1);
        __m128i delta = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(term1, term2), _mm_set1_epi16(8)), 4);

        __m128i abs_delta = _mm_abs_epi16(delta);
        __m128i ten_tc = _mm_set1_epi16(10 * tc);
        __m128i mask_filter = _mm_cmplt_epi16(abs_delta, ten_tc);

        __m128i vec_tc = _mm_set1_epi16(tc);
        __m128i vec_neg_tc = _mm_sub_epi16(_mm_setzero_si128(), vec_tc);
        __m128i delta_clipped = _mm_min_epi16(_mm_max_epi16(delta, vec_neg_tc), vec_tc);

        __m128i max_val = _mm_set1_epi16((1 << bit_depth) - 1);

        __m128i calc_p0 = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vec_p0, delta_clipped), _mm_setzero_si128()), max_val);
        __m128i calc_q0 = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(vec_q0, delta_clipped), _mm_setzero_si128()), max_val);

        *p_new_p0 = select_vec(vec_p0, calc_p0, mask_filter);
        *p_new_q0 = select_vec(vec_q0, calc_q0, mask_filter);

        *p_new_p1 = vec_p1;
        if(dep == 1)
        {
            __m128i avg_p = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(vec_p2, vec_p0), _mm_set1_epi16(1)), 1);
            __m128i delta_p = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(avg_p, vec_p1), delta_clipped), 1);
            __m128i tc_div2 = _mm_set1_epi16(tc >> 1);
            __m128i neg_tc_div2 = _mm_sub_epi16(_mm_setzero_si128(), tc_div2);
            delta_p = _mm_min_epi16(_mm_max_epi16(delta_p, neg_tc_div2), tc_div2);
            __m128i calc_p1 = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vec_p1, delta_p), _mm_setzero_si128()), max_val);
            *p_new_p1 = select_vec(vec_p1, calc_p1, mask_filter);
        }

        *p_new_q1 = vec_q1;
        if(deq == 1)
        {
            __m128i avg_q = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(vec_q2, vec_q0), _mm_set1_epi16(1)), 1);
            __m128i delta_q = _mm_srai_epi16(_mm_sub_epi16(_mm_sub_epi16(avg_q, vec_q1), delta_clipped), 1);
            __m128i tc_div2 = _mm_set1_epi16(tc >> 1);
            __m128i neg_tc_div2 = _mm_sub_epi16(_mm_setzero_si128(), tc_div2);
            delta_q = _mm_min_epi16(_mm_max_epi16(delta_q, neg_tc_div2), tc_div2);
            __m128i calc_q1 = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vec_q1, delta_q), _mm_setzero_si128()), max_val);
            *p_new_q1 = select_vec(vec_q1, calc_q1, mask_filter);
        }

        *p_new_p2 = vec_p2;
        *p_new_q2 = vec_q2;
    }

    return 1;
}

/**
*******************************************************************************
*
* @brief
*  SSSE3 implementation of luma vertical edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_luma_vert_ssse3(UWORD16 *pu2_src,
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

    /* Load 4 rows of 8 samples [-4..3] */
    __m128i r0 = _mm_loadu_si128((__m128i const *)(pu2_src - 4));
    __m128i r1 = _mm_loadu_si128((__m128i const *)(pu2_src + src_strd - 4));
    __m128i r2 = _mm_loadu_si128((__m128i const *)(pu2_src + 2 * src_strd - 4));
    __m128i r3 = _mm_loadu_si128((__m128i const *)(pu2_src + 3 * src_strd - 4));

    /* Transpose low 64 bits of 4 rows: p3, p2, p1, p0 */
    __m128i t0 = _mm_unpacklo_epi16(r0, r1);
    __m128i t1 = _mm_unpacklo_epi16(r2, r3);
    __m128i u0 = _mm_unpacklo_epi32(t0, t1);
    __m128i u1 = _mm_unpackhi_epi32(t0, t1);

    __m128i vec_p3 = u0;
    __m128i vec_p2 = _mm_srli_si128(u0, 8);
    __m128i vec_p1 = u1;
    __m128i vec_p0 = _mm_srli_si128(u1, 8);

    /* Transpose high 64 bits of 4 rows: q0, q1, q2, q3 */
    __m128i t2 = _mm_unpackhi_epi16(r0, r1);
    __m128i t3 = _mm_unpackhi_epi16(r2, r3);
    __m128i u2 = _mm_unpacklo_epi32(t2, t3);
    __m128i u3 = _mm_unpackhi_epi32(t2, t3);

    __m128i vec_q0 = u2;
    __m128i vec_q1 = _mm_srli_si128(u2, 8);
    __m128i vec_q2 = u3;
    __m128i vec_q3 = _mm_srli_si128(u3, 8);

    __m128i new_p2, new_p1, new_p0;
    __m128i new_q0, new_q1, new_q2;

    WORD32 filtered = ihevc_hbd_deblk_luma_core(
        vec_p3, vec_p2, vec_p1, vec_p0,
        vec_q0, vec_q1, vec_q2, vec_q3,
        beta, tc, bit_depth,
        &new_p2, &new_p1, &new_p0,
        &new_q0, &new_q1, &new_q2);

    if(!filtered)
    {
        return;
    }

    if(filter_flag_p)
    {
        pu2_src[0 * src_strd - 3] = (UWORD16)_mm_extract_epi16(new_p2, 0);
        pu2_src[0 * src_strd - 2] = (UWORD16)_mm_extract_epi16(new_p1, 0);
        pu2_src[0 * src_strd - 1] = (UWORD16)_mm_extract_epi16(new_p0, 0);

        pu2_src[1 * src_strd - 3] = (UWORD16)_mm_extract_epi16(new_p2, 1);
        pu2_src[1 * src_strd - 2] = (UWORD16)_mm_extract_epi16(new_p1, 1);
        pu2_src[1 * src_strd - 1] = (UWORD16)_mm_extract_epi16(new_p0, 1);

        pu2_src[2 * src_strd - 3] = (UWORD16)_mm_extract_epi16(new_p2, 2);
        pu2_src[2 * src_strd - 2] = (UWORD16)_mm_extract_epi16(new_p1, 2);
        pu2_src[2 * src_strd - 1] = (UWORD16)_mm_extract_epi16(new_p0, 2);

        pu2_src[3 * src_strd - 3] = (UWORD16)_mm_extract_epi16(new_p2, 3);
        pu2_src[3 * src_strd - 2] = (UWORD16)_mm_extract_epi16(new_p1, 3);
        pu2_src[3 * src_strd - 1] = (UWORD16)_mm_extract_epi16(new_p0, 3);
    }

    if(filter_flag_q)
    {
        pu2_src[0 * src_strd + 0] = (UWORD16)_mm_extract_epi16(new_q0, 0);
        pu2_src[0 * src_strd + 1] = (UWORD16)_mm_extract_epi16(new_q1, 0);
        pu2_src[0 * src_strd + 2] = (UWORD16)_mm_extract_epi16(new_q2, 0);

        pu2_src[1 * src_strd + 0] = (UWORD16)_mm_extract_epi16(new_q0, 1);
        pu2_src[1 * src_strd + 1] = (UWORD16)_mm_extract_epi16(new_q1, 1);
        pu2_src[1 * src_strd + 2] = (UWORD16)_mm_extract_epi16(new_q2, 1);

        pu2_src[2 * src_strd + 0] = (UWORD16)_mm_extract_epi16(new_q0, 2);
        pu2_src[2 * src_strd + 1] = (UWORD16)_mm_extract_epi16(new_q1, 2);
        pu2_src[2 * src_strd + 2] = (UWORD16)_mm_extract_epi16(new_q2, 2);

        pu2_src[3 * src_strd + 0] = (UWORD16)_mm_extract_epi16(new_q0, 3);
        pu2_src[3 * src_strd + 1] = (UWORD16)_mm_extract_epi16(new_q1, 3);
        pu2_src[3 * src_strd + 2] = (UWORD16)_mm_extract_epi16(new_q2, 3);
    }
}

/**
*******************************************************************************
*
* @brief
*  SSSE3 implementation of luma horizontal edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_luma_horz_ssse3(UWORD16 *pu2_src,
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

    /* Load 4 columns for each of the 8 lines [-4..3] */
    __m128i vec_p3 = _mm_loadl_epi64((__m128i const *)(pu2_src - 4 * src_strd));
    __m128i vec_p2 = _mm_loadl_epi64((__m128i const *)(pu2_src - 3 * src_strd));
    __m128i vec_p1 = _mm_loadl_epi64((__m128i const *)(pu2_src - 2 * src_strd));
    __m128i vec_p0 = _mm_loadl_epi64((__m128i const *)(pu2_src - 1 * src_strd));
    __m128i vec_q0 = _mm_loadl_epi64((__m128i const *)(pu2_src + 0 * src_strd));
    __m128i vec_q1 = _mm_loadl_epi64((__m128i const *)(pu2_src + 1 * src_strd));
    __m128i vec_q2 = _mm_loadl_epi64((__m128i const *)(pu2_src + 2 * src_strd));
    __m128i vec_q3 = _mm_loadl_epi64((__m128i const *)(pu2_src + 3 * src_strd));

    __m128i new_p2, new_p1, new_p0;
    __m128i new_q0, new_q1, new_q2;

    WORD32 filtered = ihevc_hbd_deblk_luma_core(
        vec_p3, vec_p2, vec_p1, vec_p0,
        vec_q0, vec_q1, vec_q2, vec_q3,
        beta, tc, bit_depth,
        &new_p2, &new_p1, &new_p0,
        &new_q0, &new_q1, &new_q2);

    if(!filtered)
    {
        return;
    }

    if(filter_flag_p)
    {
        _mm_storel_epi64((__m128i *)(pu2_src - 3 * src_strd), new_p2);
        _mm_storel_epi64((__m128i *)(pu2_src - 2 * src_strd), new_p1);
        _mm_storel_epi64((__m128i *)(pu2_src - 1 * src_strd), new_p0);
    }

    if(filter_flag_q)
    {
        _mm_storel_epi64((__m128i *)(pu2_src + 0 * src_strd), new_q0);
        _mm_storel_epi64((__m128i *)(pu2_src + 1 * src_strd), new_q1);
        _mm_storel_epi64((__m128i *)(pu2_src + 2 * src_strd), new_q2);
    }
}
