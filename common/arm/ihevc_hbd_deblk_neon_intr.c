/******************************************************************************
*
* Copyright (C) 2026 The Android Open Source Project
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
*****************************************************************************/
/**
*******************************************************************************
* @file
*  ihevc_hbd_deblk_neon_intr.c
*
* @brief
*  Contains function definitions for high bit depth deblocking filters using
*  ARM NEON intrinsics.
*
* @author
*  Ittiam
*
* @par List of Functions:
*   - ihevc_hbd_deblk_luma_vert_neonintr()
*   - ihevc_hbd_deblk_luma_horz_neonintr()
*   - ihevc_hbd_deblk_chroma_vert_neonintr()
*   - ihevc_hbd_deblk_chroma_horz_neonintr()
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
#include <arm_neon.h>

#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_deblk.h"
#include "ihevc_deblk_tables.h"
#include "ihevc_debug.h"
#include "ihevc_defs.h"

/**
*******************************************************************************
*
* @brief
*  ARM NEON implementation of luma vertical edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_luma_vert_neonintr(UWORD16 *pu2_src,
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
    WORD32 dp0, dp3, dq0, dq3, d0, d3, dp, dq, d;
    WORD32 d_sam0, d_sam3, de, dep, deq;

    ASSERT((bs > 0) && (bs <= 3));
    ASSERT(filter_flag_p || filter_flag_q);

    qp_luma = (quant_param_p + quant_param_q + 1) >> 1;
    beta_indx = CLIP3(qp_luma + (beta_offset_div2 << 1), 0, 51);

    /* BS based on implementation can take value 3 if it is intra/inter egde          */
    /* based on BS, tc index is calcuated by adding 2 * ( bs - 1) to QP and tc_offset */
    /* for BS = 1 adding factor is (0*2), BS = 2 or 3 adding factor is (1*2)          */
    /* the above desired functionallity is achieved by doing (2*(bs>>1))              */

    tc_indx = CLIP3(qp_luma + (2 * (bs >> 1)) + (tc_offset_div2 << 1), 0, 53);

    beta = gai4_ihevc_beta_table[beta_indx] * (1 << (bit_depth - 8));
    tc = gai4_ihevc_tc_table[tc_indx] * (1 << (bit_depth - 8));
    if(0 == tc)
    {
        return;
    }

    dq0 = ABS(pu2_src[2] - 2 * pu2_src[1] + pu2_src[0]);
    dq3 = ABS(pu2_src[3 * src_strd + 2] - 2 * pu2_src[3 * src_strd + 1]
                    + pu2_src[3 * src_strd + 0]);
    dp0 = ABS(pu2_src[-3] - 2 * pu2_src[-2] + pu2_src[-1]);
    dp3 = ABS(pu2_src[3 * src_strd - 3] - 2 * pu2_src[3 * src_strd - 2]
                    + pu2_src[3 * src_strd - 1]);

    d0 = dp0 + dq0;
    d3 = dp3 + dq3;
    dp = dp0 + dp3;
    dq = dq0 + dq3;
    d = d0 + d3;

    de = 0;
    dep = 0;
    deq = 0;

    if(d < beta)
    {
        d_sam0 = 0;
        if((2 * d0 < (beta >> 2))
                        && (ABS(pu2_src[3] - pu2_src[0]) + ABS(pu2_src[-1] - pu2_src[-4])
                                        < (beta >> 3))
                        && ABS(pu2_src[0] - pu2_src[-1]) < ((5 * tc + 1) >> 1))
        {
            d_sam0 = 1;
        }

        UWORD16 *pu2_src_r3 = pu2_src + 3 * src_strd;
        d_sam3 = 0;
        if((2 * d3 < (beta >> 2))
                        && (ABS(pu2_src_r3[3] - pu2_src_r3[0]) + ABS(pu2_src_r3[-1] - pu2_src_r3[-4])
                                        < (beta >> 3))
                        && ABS(pu2_src_r3[0] - pu2_src_r3[-1]) < ((5 * tc + 1) >> 1))
        {
            d_sam3 = 1;
        }

        de = (d_sam0 == 1 && d_sam3 == 1) ? 2 : 1;
        dep = (dp < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
        deq = (dq < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
        if(tc <= 1)
        {
            dep = 0;
            deq = 0;
        }
    }

    if(de != 0)
    {
        /* Load 4 rows of 8 samples [-4..3] */
        uint16x8_t r0 = vld1q_u16(pu2_src - 4);
        uint16x8_t r1 = vld1q_u16(pu2_src + src_strd - 4);
        uint16x8_t r2 = vld1q_u16(pu2_src + 2 * src_strd - 4);
        uint16x8_t r3 = vld1q_u16(pu2_src + 3 * src_strd - 4);

        /* Transpose low 64 bits of 4 rows: p3, p2, p1, p0 */
        uint16x4_t lo0 = vget_low_u16(r0);
        uint16x4_t lo1 = vget_low_u16(r1);
        uint16x4_t lo2 = vget_low_u16(r2);
        uint16x4_t lo3 = vget_low_u16(r3);

        uint16x4x2_t trn01_lo = vtrn_u16(lo0, lo1);
        uint16x4x2_t trn23_lo = vtrn_u16(lo2, lo3);
        uint32x2x2_t trn_p_0 = vtrn_u32(vreinterpret_u32_u16(trn01_lo.val[0]), vreinterpret_u32_u16(trn23_lo.val[0]));
        uint32x2x2_t trn_p_1 = vtrn_u32(vreinterpret_u32_u16(trn01_lo.val[1]), vreinterpret_u32_u16(trn23_lo.val[1]));

        int16x4_t vec_p3 = vreinterpret_s16_u32(trn_p_0.val[0]);
        int16x4_t vec_p2 = vreinterpret_s16_u32(trn_p_1.val[0]);
        int16x4_t vec_p1 = vreinterpret_s16_u32(trn_p_0.val[1]);
        int16x4_t vec_p0 = vreinterpret_s16_u32(trn_p_1.val[1]);

        /* Transpose high 64 bits of 4 rows: q0, q1, q2, q3 */
        uint16x4_t hi0 = vget_high_u16(r0);
        uint16x4_t hi1 = vget_high_u16(r1);
        uint16x4_t hi2 = vget_high_u16(r2);
        uint16x4_t hi3 = vget_high_u16(r3);

        uint16x4x2_t trn01_hi = vtrn_u16(hi0, hi1);
        uint16x4x2_t trn23_hi = vtrn_u16(hi2, hi3);
        uint32x2x2_t trn_q_0 = vtrn_u32(vreinterpret_u32_u16(trn01_hi.val[0]), vreinterpret_u32_u16(trn23_hi.val[0]));
        uint32x2x2_t trn_q_1 = vtrn_u32(vreinterpret_u32_u16(trn01_hi.val[1]), vreinterpret_u32_u16(trn23_hi.val[1]));

        int16x4_t vec_q0 = vreinterpret_s16_u32(trn_q_0.val[0]);
        int16x4_t vec_q1 = vreinterpret_s16_u32(trn_q_1.val[0]);
        int16x4_t vec_q2 = vreinterpret_s16_u32(trn_q_0.val[1]);
        int16x4_t vec_q3 = vreinterpret_s16_u32(trn_q_1.val[1]);

        int16x4_t new_p2, new_p1, new_p0;
        int16x4_t new_q0, new_q1, new_q2;

        if(de == 2)
        {
            int16x4_t two_tc = vdup_n_s16(2 * tc);
            int16x4_t q1_q0_p0 = vadd_s16(vec_q1, vadd_s16(vec_q0, vec_p0));

            /* q0 = CLIP3((q2 + 2*q1 + 2*q0 + 2*p0 + p1 + 4) >> 3, q0 - 2*tc, q0 + 2*tc) */
            int16x4_t q0_sum = vadd_s16(vadd_s16(vec_q2, vec_p1), vshl_n_s16(q1_q0_p0, 1));
            int16x4_t q0_val = vrshr_n_s16(q0_sum, 3);
            new_q0 = vmin_s16(vmax_s16(q0_val, vsub_s16(vec_q0, two_tc)), vadd_s16(vec_q0, two_tc));

            /* q1 = CLIP3((q2 + q1 + q0 + p0 + 2) >> 2, q1 - 2*tc, q1 + 2*tc) */
            int16x4_t q1_sum = vadd_s16(vec_q2, q1_q0_p0);
            int16x4_t q1_val = vrshr_n_s16(q1_sum, 2);
            new_q1 = vmin_s16(vmax_s16(q1_val, vsub_s16(vec_q1, two_tc)), vadd_s16(vec_q1, two_tc));

            /* q2 = CLIP3((2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3, q2 - 2*tc, q2 + 2*tc) */
            int16x4_t q3_q2_x2 = vshl_n_s16(vadd_s16(vec_q3, vec_q2), 1);
            int16x4_t q2_sum = vadd_s16(q3_q2_x2, vadd_s16(vec_q2, q1_q0_p0));
            int16x4_t q2_val = vrshr_n_s16(q2_sum, 3);
            new_q2 = vmin_s16(vmax_s16(q2_val, vsub_s16(vec_q2, two_tc)), vadd_s16(vec_q2, two_tc));

            /* p side */
            int16x4_t p1_p0_q0 = vadd_s16(vec_p1, vadd_s16(vec_p0, vec_q0));

            /* p0 = CLIP3((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3, p0 - 2*tc, p0 + 2*tc) */
            int16x4_t p0_sum = vadd_s16(vadd_s16(vec_p2, vec_q1), vshl_n_s16(p1_p0_q0, 1));
            int16x4_t p0_val = vrshr_n_s16(p0_sum, 3);
            new_p0 = vmin_s16(vmax_s16(p0_val, vsub_s16(vec_p0, two_tc)), vadd_s16(vec_p0, two_tc));

            /* p1 = CLIP3((p2 + p1 + p0 + q0 + 2) >> 2, p1 - 2*tc, p1 + 2*tc) */
            int16x4_t p1_sum = vadd_s16(vec_p2, p1_p0_q0);
            int16x4_t p1_val = vrshr_n_s16(p1_sum, 2);
            new_p1 = vmin_s16(vmax_s16(p1_val, vsub_s16(vec_p1, two_tc)), vadd_s16(vec_p1, two_tc));

            /* p2 = CLIP3((2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3, p2 - 2*tc, p2 + 2*tc) */
            int16x4_t p3_p2_x2 = vshl_n_s16(vadd_s16(vec_p3, vec_p2), 1);
            int16x4_t p2_sum = vadd_s16(p3_p2_x2, vadd_s16(vec_p2, p1_p0_q0));
            int16x4_t p2_val = vrshr_n_s16(p2_sum, 3);
            new_p2 = vmin_s16(vmax_s16(p2_val, vsub_s16(vec_p2, two_tc)), vadd_s16(vec_p2, two_tc));
        }
        else
        {
            int16x4_t d_q0_p0 = vsub_s16(vec_q0, vec_p0);
            int16x4_t d_q1_p1 = vsub_s16(vec_q1, vec_p1);

            /* delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4 */
            int16x4_t term1 = vadd_s16(vshl_n_s16(d_q0_p0, 3), d_q0_p0);
            int16x4_t term2 = vadd_s16(vshl_n_s16(d_q1_p1, 1), d_q1_p1);
            int16x4_t delta = vrshr_n_s16(vsub_s16(term1, term2), 4);

            int16x4_t abs_delta = vabs_s16(delta);
            uint16x4_t mask_filter = vclt_s16(abs_delta, vdup_n_s16(10 * tc));

            int16x4_t vec_tc = vdup_n_s16(tc);
            int16x4_t vec_neg_tc = vneg_s16(vec_tc);
            int16x4_t delta_clipped = vmin_s16(vmax_s16(delta, vec_neg_tc), vec_tc);

            int16x4_t max_val = vdup_n_s16((1 << bit_depth) - 1);
            int16x4_t zero = vdup_n_s16(0);

            int16x4_t calc_p0 = vmin_s16(vmax_s16(vadd_s16(vec_p0, delta_clipped), zero), max_val);
            int16x4_t calc_q0 = vmin_s16(vmax_s16(vsub_s16(vec_q0, delta_clipped), zero), max_val);

            new_p0 = vbsl_s16(mask_filter, calc_p0, vec_p0);
            new_q0 = vbsl_s16(mask_filter, calc_q0, vec_q0);

            new_p1 = vec_p1;
            if(dep == 1)
            {
                int16x4_t avg_p = vrhadd_s16(vec_p2, vec_p0);
                int16x4_t delta_p = vshr_n_s16(vadd_s16(vsub_s16(avg_p, vec_p1), delta_clipped), 1);
                int16x4_t tc_div2 = vdup_n_s16(tc >> 1);
                int16x4_t neg_tc_div2 = vneg_s16(tc_div2);
                delta_p = vmin_s16(vmax_s16(delta_p, neg_tc_div2), tc_div2);
                int16x4_t calc_p1 = vmin_s16(vmax_s16(vadd_s16(vec_p1, delta_p), zero), max_val);
                new_p1 = vbsl_s16(mask_filter, calc_p1, vec_p1);
            }

            new_q1 = vec_q1;
            if(deq == 1)
            {
                int16x4_t avg_q = vrhadd_s16(vec_q2, vec_q0);
                int16x4_t delta_q = vshr_n_s16(vsub_s16(vsub_s16(avg_q, vec_q1), delta_clipped), 1);
                int16x4_t tc_div2 = vdup_n_s16(tc >> 1);
                int16x4_t neg_tc_div2 = vneg_s16(tc_div2);
                delta_q = vmin_s16(vmax_s16(delta_q, neg_tc_div2), tc_div2);
                int16x4_t calc_q1 = vmin_s16(vmax_s16(vadd_s16(vec_q1, delta_q), zero), max_val);
                new_q1 = vbsl_s16(mask_filter, calc_q1, vec_q1);
            }

            new_p2 = vec_p2;
            new_q2 = vec_q2;
        }

        if(filter_flag_p)
        {
            /* Transpose (vec_p3, new_p2, new_p1, new_p0) back to 4 rows */
            int16x4x2_t out_p32 = vtrn_s16(vec_p3, new_p2);
            int16x4x2_t out_p10 = vtrn_s16(new_p1, new_p0);
            int32x2x2_t row_p02 = vtrn_s32(vreinterpret_s32_s16(out_p32.val[0]), vreinterpret_s32_s16(out_p10.val[0]));
            int32x2x2_t row_p13 = vtrn_s32(vreinterpret_s32_s16(out_p32.val[1]), vreinterpret_s32_s16(out_p10.val[1]));

            vst1_u16(pu2_src + 0 * src_strd - 4, vreinterpret_u16_s32(row_p02.val[0]));
            vst1_u16(pu2_src + 1 * src_strd - 4, vreinterpret_u16_s32(row_p13.val[0]));
            vst1_u16(pu2_src + 2 * src_strd - 4, vreinterpret_u16_s32(row_p02.val[1]));
            vst1_u16(pu2_src + 3 * src_strd - 4, vreinterpret_u16_s32(row_p13.val[1]));
        }

        if(filter_flag_q)
        {
            /* Transpose (new_q0, new_q1, new_q2, vec_q3) back to 4 rows */
            int16x4x2_t out_q01 = vtrn_s16(new_q0, new_q1);
            int16x4x2_t out_q23 = vtrn_s16(new_q2, vec_q3);
            int32x2x2_t row_q02 = vtrn_s32(vreinterpret_s32_s16(out_q01.val[0]), vreinterpret_s32_s16(out_q23.val[0]));
            int32x2x2_t row_q13 = vtrn_s32(vreinterpret_s32_s16(out_q01.val[1]), vreinterpret_s32_s16(out_q23.val[1]));

            vst1_u16(pu2_src + 0 * src_strd, vreinterpret_u16_s32(row_q02.val[0]));
            vst1_u16(pu2_src + 1 * src_strd, vreinterpret_u16_s32(row_q13.val[0]));
            vst1_u16(pu2_src + 2 * src_strd, vreinterpret_u16_s32(row_q02.val[1]));
            vst1_u16(pu2_src + 3 * src_strd, vreinterpret_u16_s32(row_q13.val[1]));
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  ARM NEON implementation of luma horizontal edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_luma_horz_neonintr(UWORD16 *pu2_src,
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
    WORD32 dp0, dp3, dq0, dq3, d0, d3, dp, dq, d;
    WORD32 d_sam0, d_sam3, de, dep, deq;

    ASSERT((bs > 0));
    ASSERT(filter_flag_p || filter_flag_q);

    qp_luma = (quant_param_p + quant_param_q + 1) >> 1;
    beta_indx = CLIP3(qp_luma + (beta_offset_div2 << 1), 0, 51);

    /* BS based on implementation can take value 3 if it is intra/inter egde          */
    /* based on BS, tc index is casslcuated by adding 2 * ( bs - 1) to QP and tc_offset */
    /* for BS = 1 adding factor is (0*2), BS = 2 or 3 adding factor is (1*2)          */
    /* the above desired functionallity is achieved by doing (2*(bs>>1))              */

    tc_indx = CLIP3(qp_luma + 2 * (bs >> 1) + (tc_offset_div2 << 1), 0, 53);

    beta = gai4_ihevc_beta_table[beta_indx] * (1 << (bit_depth - 8));
    tc = gai4_ihevc_tc_table[tc_indx] * (1 << (bit_depth - 8));
    if(0 == tc)
    {
        return;
    }

    dq0 = ABS(pu2_src[2 * src_strd] - 2 * pu2_src[1 * src_strd] +
                    pu2_src[0 * src_strd]);
    dq3 = ABS(pu2_src[3 + 2 * src_strd] - 2 * pu2_src[3 + 1 * src_strd] +
                    pu2_src[3 + 0 * src_strd]);
    dp0 = ABS(pu2_src[-3 * src_strd] - 2 * pu2_src[-2 * src_strd] +
                    pu2_src[-1 * src_strd]);
    dp3 = ABS(pu2_src[3 - 3 * src_strd] - 2 * pu2_src[3 - 2 * src_strd] +
                    pu2_src[3 - 1 * src_strd]);

    d0 = dp0 + dq0;
    d3 = dp3 + dq3;
    dp = dp0 + dp3;
    dq = dq0 + dq3;
    d = d0 + d3;

    de = 0;
    dep = 0;
    deq = 0;

    if(d < beta)
    {
        d_sam0 = 0;
        if((2 * d0 < (beta >> 2))
                        && (ABS(pu2_src[3 * src_strd] - pu2_src[0])
                                        + ABS(pu2_src[-1 * src_strd] - pu2_src[-4 * src_strd])
                                        < (beta >> 3))
                        && ABS(pu2_src[0] - pu2_src[-1 * src_strd]) < ((5 * tc + 1) >> 1))
        {
            d_sam0 = 1;
        }

        UWORD16 *pu2_src_c3 = pu2_src + 3;
        d_sam3 = 0;
        if((2 * d3 < (beta >> 2))
                        && (ABS(pu2_src_c3[3 * src_strd] - pu2_src_c3[0])
                                        + ABS(pu2_src_c3[-1 * src_strd] - pu2_src_c3[-4 * src_strd])
                                        < (beta >> 3))
                        && ABS(pu2_src_c3[0] - pu2_src_c3[-1 * src_strd]) < ((5 * tc + 1) >> 1))
        {
            d_sam3 = 1;
        }

        de = (d_sam0 == 1 && d_sam3 == 1) ? 2 : 1;
        dep = (dp < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
        deq = (dq < ((beta + (beta >> 1)) >> 3)) ? 1 : 0;
        if(tc <= 1)
        {
            dep = 0;
            deq = 0;
        }
    }

    if(de != 0)
    {
        int16x4_t vec_p3 = vreinterpret_s16_u16(vld1_u16(pu2_src - 4 * src_strd));
        int16x4_t vec_p2 = vreinterpret_s16_u16(vld1_u16(pu2_src - 3 * src_strd));
        int16x4_t vec_p1 = vreinterpret_s16_u16(vld1_u16(pu2_src - 2 * src_strd));
        int16x4_t vec_p0 = vreinterpret_s16_u16(vld1_u16(pu2_src - 1 * src_strd));
        int16x4_t vec_q0 = vreinterpret_s16_u16(vld1_u16(pu2_src + 0 * src_strd));
        int16x4_t vec_q1 = vreinterpret_s16_u16(vld1_u16(pu2_src + 1 * src_strd));
        int16x4_t vec_q2 = vreinterpret_s16_u16(vld1_u16(pu2_src + 2 * src_strd));
        int16x4_t vec_q3 = vreinterpret_s16_u16(vld1_u16(pu2_src + 3 * src_strd));

        int16x4_t new_p2, new_p1, new_p0;
        int16x4_t new_q0, new_q1, new_q2;

        if(de == 2)
        {
            int16x4_t two_tc = vdup_n_s16(2 * tc);
            int16x4_t q1_q0_p0 = vadd_s16(vec_q1, vadd_s16(vec_q0, vec_p0));

            /* q0 = CLIP3((q2 + 2*q1 + 2*q0 + 2*p0 + p1 + 4) >> 3, q0 - 2*tc, q0 + 2*tc) */
            int16x4_t q0_sum = vadd_s16(vadd_s16(vec_q2, vec_p1), vshl_n_s16(q1_q0_p0, 1));
            int16x4_t q0_val = vrshr_n_s16(q0_sum, 3);
            new_q0 = vmin_s16(vmax_s16(q0_val, vsub_s16(vec_q0, two_tc)), vadd_s16(vec_q0, two_tc));

            /* q1 = CLIP3((q2 + q1 + q0 + p0 + 2) >> 2, q1 - 2*tc, q1 + 2*tc) */
            int16x4_t q1_sum = vadd_s16(vec_q2, q1_q0_p0);
            int16x4_t q1_val = vrshr_n_s16(q1_sum, 2);
            new_q1 = vmin_s16(vmax_s16(q1_val, vsub_s16(vec_q1, two_tc)), vadd_s16(vec_q1, two_tc));

            /* q2 = CLIP3((2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3, q2 - 2*tc, q2 + 2*tc) */
            int16x4_t q3_q2_x2 = vshl_n_s16(vadd_s16(vec_q3, vec_q2), 1);
            int16x4_t q2_sum = vadd_s16(q3_q2_x2, vadd_s16(vec_q2, q1_q0_p0));
            int16x4_t q2_val = vrshr_n_s16(q2_sum, 3);
            new_q2 = vmin_s16(vmax_s16(q2_val, vsub_s16(vec_q2, two_tc)), vadd_s16(vec_q2, two_tc));

            /* p side */
            int16x4_t p1_p0_q0 = vadd_s16(vec_p1, vadd_s16(vec_p0, vec_q0));

            /* p0 = CLIP3((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3, p0 - 2*tc, p0 + 2*tc) */
            int16x4_t p0_sum = vadd_s16(vadd_s16(vec_p2, vec_q1), vshl_n_s16(p1_p0_q0, 1));
            int16x4_t p0_val = vrshr_n_s16(p0_sum, 3);
            new_p0 = vmin_s16(vmax_s16(p0_val, vsub_s16(vec_p0, two_tc)), vadd_s16(vec_p0, two_tc));

            /* p1 = CLIP3((p2 + p1 + p0 + q0 + 2) >> 2, p1 - 2*tc, p1 + 2*tc) */
            int16x4_t p1_sum = vadd_s16(vec_p2, p1_p0_q0);
            int16x4_t p1_val = vrshr_n_s16(p1_sum, 2);
            new_p1 = vmin_s16(vmax_s16(p1_val, vsub_s16(vec_p1, two_tc)), vadd_s16(vec_p1, two_tc));

            /* p2 = CLIP3((2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3, p2 - 2*tc, p2 + 2*tc) */
            int16x4_t p3_p2_x2 = vshl_n_s16(vadd_s16(vec_p3, vec_p2), 1);
            int16x4_t p2_sum = vadd_s16(p3_p2_x2, vadd_s16(vec_p2, p1_p0_q0));
            int16x4_t p2_val = vrshr_n_s16(p2_sum, 3);
            new_p2 = vmin_s16(vmax_s16(p2_val, vsub_s16(vec_p2, two_tc)), vadd_s16(vec_p2, two_tc));
        }
        else
        {
            int16x4_t d_q0_p0 = vsub_s16(vec_q0, vec_p0);
            int16x4_t d_q1_p1 = vsub_s16(vec_q1, vec_p1);

            /* delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4 */
            int16x4_t term1 = vadd_s16(vshl_n_s16(d_q0_p0, 3), d_q0_p0);
            int16x4_t term2 = vadd_s16(vshl_n_s16(d_q1_p1, 1), d_q1_p1);
            int16x4_t delta = vrshr_n_s16(vsub_s16(term1, term2), 4);

            int16x4_t abs_delta = vabs_s16(delta);
            uint16x4_t mask_filter = vclt_s16(abs_delta, vdup_n_s16(10 * tc));

            int16x4_t vec_tc = vdup_n_s16(tc);
            int16x4_t vec_neg_tc = vneg_s16(vec_tc);
            int16x4_t delta_clipped = vmin_s16(vmax_s16(delta, vec_neg_tc), vec_tc);

            int16x4_t max_val = vdup_n_s16((1 << bit_depth) - 1);
            int16x4_t zero = vdup_n_s16(0);

            int16x4_t calc_p0 = vmin_s16(vmax_s16(vadd_s16(vec_p0, delta_clipped), zero), max_val);
            int16x4_t calc_q0 = vmin_s16(vmax_s16(vsub_s16(vec_q0, delta_clipped), zero), max_val);

            new_p0 = vbsl_s16(mask_filter, calc_p0, vec_p0);
            new_q0 = vbsl_s16(mask_filter, calc_q0, vec_q0);

            new_p1 = vec_p1;
            if(dep == 1)
            {
                int16x4_t avg_p = vrhadd_s16(vec_p2, vec_p0);
                int16x4_t delta_p = vshr_n_s16(vadd_s16(vsub_s16(avg_p, vec_p1), delta_clipped), 1);
                int16x4_t tc_div2 = vdup_n_s16(tc >> 1);
                int16x4_t neg_tc_div2 = vneg_s16(tc_div2);
                delta_p = vmin_s16(vmax_s16(delta_p, neg_tc_div2), tc_div2);
                int16x4_t calc_p1 = vmin_s16(vmax_s16(vadd_s16(vec_p1, delta_p), zero), max_val);
                new_p1 = vbsl_s16(mask_filter, calc_p1, vec_p1);
            }

            new_q1 = vec_q1;
            if(deq == 1)
            {
                int16x4_t avg_q = vrhadd_s16(vec_q2, vec_q0);
                int16x4_t delta_q = vshr_n_s16(vsub_s16(vsub_s16(avg_q, vec_q1), delta_clipped), 1);
                int16x4_t tc_div2 = vdup_n_s16(tc >> 1);
                int16x4_t neg_tc_div2 = vneg_s16(tc_div2);
                delta_q = vmin_s16(vmax_s16(delta_q, neg_tc_div2), tc_div2);
                int16x4_t calc_q1 = vmin_s16(vmax_s16(vadd_s16(vec_q1, delta_q), zero), max_val);
                new_q1 = vbsl_s16(mask_filter, calc_q1, vec_q1);
            }

            new_p2 = vec_p2;
            new_q2 = vec_q2;
        }

        if(filter_flag_p)
        {
            vst1_u16(pu2_src - 3 * src_strd, vreinterpret_u16_s16(new_p2));
            vst1_u16(pu2_src - 2 * src_strd, vreinterpret_u16_s16(new_p1));
            vst1_u16(pu2_src - 1 * src_strd, vreinterpret_u16_s16(new_p0));
        }

        if(filter_flag_q)
        {
            vst1_u16(pu2_src + 0 * src_strd, vreinterpret_u16_s16(new_q0));
            vst1_u16(pu2_src + 1 * src_strd, vreinterpret_u16_s16(new_q1));
            vst1_u16(pu2_src + 2 * src_strd, vreinterpret_u16_s16(new_q2));
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  ARM NEON implementation of chroma vertical edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_chroma_vert_neonintr(UWORD16 *pu2_src,
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

    /* chroma processing is done only if BS is 2             */
    /* this function is assumed to be called only if BS is 2 */
    qp_indx_u = qp_offset_u + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_indx_v = qp_offset_v + ((quant_param_p + quant_param_q + 1) >> 1);

    /* 8.7.2.5.5 Filtering process for chroma block edges */
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

    /* Load 4 rows of 8 samples [-4..3] as 32-bit UV pairs: (p1_u, p1_v), (p0_u, p0_v), (q0_u, q0_v), (q1_u, q1_v) */
    uint32x4_t r0 = vld1q_u32((uint32_t *)(pu2_src - 4));
    uint32x4_t r1 = vld1q_u32((uint32_t *)(pu2_src + src_strd - 4));
    uint32x4_t r2 = vld1q_u32((uint32_t *)(pu2_src + 2 * src_strd - 4));
    uint32x4_t r3 = vld1q_u32((uint32_t *)(pu2_src + 3 * src_strd - 4));

    /* 4x4 32-bit transpose: separates p1, p0, q0, q1 (each is a UV pair) for all 4 rows */
    uint32x4x2_t trn01 = vtrnq_u32(r0, r1);
    uint32x4x2_t trn23 = vtrnq_u32(r2, r3);

    uint32x4_t u32_p1 = vcombine_u32(vget_low_u32(trn01.val[0]), vget_low_u32(trn23.val[0]));
    uint32x4_t u32_p0 = vcombine_u32(vget_low_u32(trn01.val[1]), vget_low_u32(trn23.val[1]));
    uint32x4_t u32_q0 = vcombine_u32(vget_high_u32(trn01.val[0]), vget_high_u32(trn23.val[0]));
    uint32x4_t u32_q1 = vcombine_u32(vget_high_u32(trn01.val[1]), vget_high_u32(trn23.val[1]));

    int16x8_t vec_p1 = vreinterpretq_s16_u32(u32_p1);
    int16x8_t vec_p0 = vreinterpretq_s16_u32(u32_p0);
    int16x8_t vec_q0 = vreinterpretq_s16_u32(u32_q0);
    int16x8_t vec_q1 = vreinterpretq_s16_u32(u32_q1);

    int16x8_t vec_tc = vzipq_s16(vdupq_n_s16(tc_u), vdupq_n_s16(tc_v)).val[0];
    int16x8_t vec_neg_tc = vnegq_s16(vec_tc);

    /* delta = (((q0 - p0) << 2) + p1 - q1 + 4) >> 3 */
    int16x8_t diff_q0_p0 = vsubq_s16(vec_q0, vec_p0);
    int16x8_t diff_x4 = vshlq_n_s16(diff_q0_p0, 2);
    int16x8_t diff_p1_q1 = vsubq_s16(vec_p1, vec_q1);
    int16x8_t sum = vaddq_s16(diff_x4, diff_p1_q1);
    int16x8_t delta = vrshrq_n_s16(sum, 3);
    int16x8_t delta_clipped = vminq_s16(vmaxq_s16(delta, vec_neg_tc), vec_tc);

    int16x8_t max_val = vdupq_n_s16((1 << bit_depth) - 1);
    int16x8_t zero = vdupq_n_s16(0);

    int16x8_t new_p0 = vminq_s16(vmaxq_s16(vaddq_s16(vec_p0, delta_clipped), zero), max_val);
    int16x8_t new_q0 = vminq_s16(vmaxq_s16(vsubq_s16(vec_q0, delta_clipped), zero), max_val);

    uint32x4_t u32_new_p0 = vreinterpretq_u32_s16(new_p0);
    uint32x4_t u32_new_q0 = vreinterpretq_u32_s16(new_q0);

    if(filter_flag_p && filter_flag_q)
    {
        uint32x2_t p0_lo = vget_low_u32(u32_new_p0);
        uint32x2_t p0_hi = vget_high_u32(u32_new_p0);
        uint32x2_t q0_lo = vget_low_u32(u32_new_q0);
        uint32x2_t q0_hi = vget_high_u32(u32_new_q0);

        uint32x2x2_t zip_row01 = vzip_u32(p0_lo, q0_lo);
        uint32x2x2_t zip_row23 = vzip_u32(p0_hi, q0_hi);

        vst1_u32((uint32_t *)(pu2_src - 2), zip_row01.val[0]);
        vst1_u32((uint32_t *)(pu2_src + src_strd - 2), zip_row01.val[1]);
        vst1_u32((uint32_t *)(pu2_src + 2 * src_strd - 2), zip_row23.val[0]);
        vst1_u32((uint32_t *)(pu2_src + 3 * src_strd - 2), zip_row23.val[1]);
    }
    else if(filter_flag_p)
    {
        vst1q_lane_u32((uint32_t *)(pu2_src - 2), u32_new_p0, 0);
        vst1q_lane_u32((uint32_t *)(pu2_src + src_strd - 2), u32_new_p0, 1);
        vst1q_lane_u32((uint32_t *)(pu2_src + 2 * src_strd - 2), u32_new_p0, 2);
        vst1q_lane_u32((uint32_t *)(pu2_src + 3 * src_strd - 2), u32_new_p0, 3);
    }
    else if(filter_flag_q)
    {
        vst1q_lane_u32((uint32_t *)(pu2_src), u32_new_q0, 0);
        vst1q_lane_u32((uint32_t *)(pu2_src + src_strd), u32_new_q0, 1);
        vst1q_lane_u32((uint32_t *)(pu2_src + 2 * src_strd), u32_new_q0, 2);
        vst1q_lane_u32((uint32_t *)(pu2_src + 3 * src_strd), u32_new_q0, 3);
    }
}

/**
*******************************************************************************
*
* @brief
*  ARM NEON implementation of chroma horizontal edge deblocking filter (HBD).
*
*******************************************************************************
*/
void ihevc_hbd_deblk_chroma_horz_neonintr(UWORD16 *pu2_src,
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

    /* chroma processing is done only if BS is 2             */
    /* this function is assumed to be called only if BS is 2 */
    qp_indx_u = qp_offset_u + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_indx_v = qp_offset_v + ((quant_param_p + quant_param_q + 1) >> 1);

    /* 8.7.2.5.5 Filtering process for chroma block edges */
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

    int16x8_t vec_p1 = vreinterpretq_s16_u16(vld1q_u16(pu2_src - 2 * src_strd));
    int16x8_t vec_p0 = vreinterpretq_s16_u16(vld1q_u16(pu2_src - 1 * src_strd));
    int16x8_t vec_q0 = vreinterpretq_s16_u16(vld1q_u16(pu2_src));
    int16x8_t vec_q1 = vreinterpretq_s16_u16(vld1q_u16(pu2_src + 1 * src_strd));

    int16x8_t vec_tc = vzipq_s16(vdupq_n_s16(tc_u), vdupq_n_s16(tc_v)).val[0];
    int16x8_t vec_neg_tc = vnegq_s16(vec_tc);

    /* delta = (((q0 - p0) << 2) + p1 - q1 + 4) >> 3 */
    int16x8_t diff_q0_p0 = vsubq_s16(vec_q0, vec_p0);
    int16x8_t diff_x4 = vshlq_n_s16(diff_q0_p0, 2);
    int16x8_t diff_p1_q1 = vsubq_s16(vec_p1, vec_q1);
    int16x8_t sum = vaddq_s16(diff_x4, diff_p1_q1);
    int16x8_t delta = vrshrq_n_s16(sum, 3);
    int16x8_t delta_clipped = vminq_s16(vmaxq_s16(delta, vec_neg_tc), vec_tc);

    int16x8_t max_val = vdupq_n_s16((1 << bit_depth) - 1);
    int16x8_t zero = vdupq_n_s16(0);

    int16x8_t new_p0 = vminq_s16(vmaxq_s16(vaddq_s16(vec_p0, delta_clipped), zero), max_val);
    int16x8_t new_q0 = vminq_s16(vmaxq_s16(vsubq_s16(vec_q0, delta_clipped), zero), max_val);

    if(filter_flag_p)
    {
        vst1q_u16(pu2_src - 1 * src_strd, vreinterpretq_u16_s16(new_p0));
    }
    if(filter_flag_q)
    {
        vst1q_u16(pu2_src, vreinterpretq_u16_s16(new_q0));
    }
}
