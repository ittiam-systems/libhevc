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
 ******************************************************************************
 * @file
 *  ihevc_hbd_chroma_itrans_recon_8x8_neon_intr.c
 *
 * @brief
 *  Contains optimized function definitions for HBD 8x8 inverse transform and
 *  reconstruction of chroma interleaved data using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_hbd_chroma_itrans_recon_8x8_neonintr()
 *
 * @remarks
 *  None
 *
 ******************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_function_selector.h"
#include "ihevc_trans_macros.h"
#include "arm_neon.h"

/* Optimized 8x4 transpose using 128-bit vector primitives */
static inline void transpose8x4(int16x4_t s0, int16x4_t s1, int16x4_t s2, int16x4_t s3,
                                int16x4_t s4, int16x4_t s5, int16x4_t s6, int16x4_t s7,
                                int16x8_t *r0, int16x8_t *r1, int16x8_t *r2, int16x8_t *r3)
{
    int16x8_t s04 = vcombine_s16(s0, s4);
    int16x8_t s15 = vcombine_s16(s1, s5);
    int16x8_t s26 = vcombine_s16(s2, s6);
    int16x8_t s37 = vcombine_s16(s3, s7);

    int16x8x2_t t0 = vtrnq_s16(s04, s15);
    int16x8x2_t t1 = vtrnq_s16(s26, s37);
    int32x4x2_t u0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]), vreinterpretq_s32_s16(t1.val[0]));
    int32x4x2_t u1 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]), vreinterpretq_s32_s16(t1.val[1]));

    *r0 = vreinterpretq_s16_s32(u0.val[0]);
    *r1 = vreinterpretq_s16_s32(u1.val[0]);
    *r2 = vreinterpretq_s16_s32(u0.val[1]);
    *r3 = vreinterpretq_s16_s32(u1.val[1]);
}

static inline void idct8_4col_zero4(int16x4_t r0, int16x4_t r1, int16x4_t r2, int16x4_t r3,
                                    int16x4_t c_even, int16x4_t c_odd,
                                    int32x4_t *y0, int32x4_t *y1, int32x4_t *y2, int32x4_t *y3,
                                    int32x4_t *y4, int32x4_t *y5, int32x4_t *y6, int32x4_t *y7)
{
    int32x4_t ee0 = vmull_lane_s16(r0, c_even, 0);
    int32x4_t ee1 = ee0;
    int32x4_t eo0 = vmull_lane_s16(r2, c_even, 1);
    int32x4_t eo1 = vmull_lane_s16(r2, c_even, 2);

    int32x4_t e0 = vaddq_s32(ee0, eo0);
    int32x4_t e1 = vaddq_s32(ee1, eo1);
    int32x4_t e2 = vsubq_s32(ee1, eo1);
    int32x4_t e3 = vsubq_s32(ee0, eo0);

    /* Interleave odd branches to eliminate pipeline bubbles */
    int32x4_t o0 = vmull_lane_s16(r1, c_odd, 0);
    int32x4_t o1 = vmull_lane_s16(r1, c_odd, 1);
    int32x4_t o2 = vmull_lane_s16(r1, c_odd, 2);
    int32x4_t o3 = vmull_lane_s16(r1, c_odd, 3);

    o0 = vmlal_lane_s16(o0, r3, c_odd, 1);
    o1 = vmlsl_lane_s16(o1, r3, c_odd, 3);
    o2 = vmlsl_lane_s16(o2, r3, c_odd, 0);
    o3 = vmlsl_lane_s16(o3, r3, c_odd, 2);

    *y0 = vaddq_s32(e0, o0);
    *y1 = vaddq_s32(e1, o1);
    *y2 = vaddq_s32(e2, o2);
    *y3 = vaddq_s32(e3, o3);
    *y4 = vsubq_s32(e3, o3);
    *y5 = vsubq_s32(e2, o2);
    *y6 = vsubq_s32(e1, o1);
    *y7 = vsubq_s32(e0, o0);
}

static inline void idct8_4col_full(int16x4_t r0, int16x4_t r1, int16x4_t r2, int16x4_t r3,
                                   int16x4_t r4, int16x4_t r5, int16x4_t r6, int16x4_t r7,
                                   int16x4_t c_even, int16x4_t c_odd,
                                   int32x4_t *y0, int32x4_t *y1, int32x4_t *y2, int32x4_t *y3,
                                   int32x4_t *y4, int32x4_t *y5, int32x4_t *y6, int32x4_t *y7)
{
    /* Reuse common multiplication vmull_lane_s16(r0, c_even, 0) */
    int32x4_t r0_64 = vmull_lane_s16(r0, c_even, 0);
    int32x4_t eo0   = vmull_lane_s16(r2, c_even, 1);
    int32x4_t eo1   = vmull_lane_s16(r2, c_even, 2);

    int32x4_t ee0   = vmlal_lane_s16(r0_64, r4, c_even, 0);
    int32x4_t ee1   = vmlsl_lane_s16(r0_64, r4, c_even, 0);
    eo0             = vmlal_lane_s16(eo0, r6, c_even, 2);
    eo1             = vmlsl_lane_s16(eo1, r6, c_even, 1);

    int32x4_t e0 = vaddq_s32(ee0, eo0);
    int32x4_t e1 = vaddq_s32(ee1, eo1);
    int32x4_t e2 = vsubq_s32(ee1, eo1);
    int32x4_t e3 = vsubq_s32(ee0, eo0);

    /* Interleaved odd path across o0, o1, o2, o3 to hide pipeline latency */
    int32x4_t o0 = vmull_lane_s16(r1, c_odd, 0);
    int32x4_t o1 = vmull_lane_s16(r1, c_odd, 1);
    int32x4_t o2 = vmull_lane_s16(r1, c_odd, 2);
    int32x4_t o3 = vmull_lane_s16(r1, c_odd, 3);

    o0 = vmlal_lane_s16(o0, r3, c_odd, 1);
    o1 = vmlsl_lane_s16(o1, r3, c_odd, 3);
    o2 = vmlsl_lane_s16(o2, r3, c_odd, 0);
    o3 = vmlsl_lane_s16(o3, r3, c_odd, 2);

    o0 = vmlal_lane_s16(o0, r5, c_odd, 2);
    o1 = vmlsl_lane_s16(o1, r5, c_odd, 0);
    o2 = vmlal_lane_s16(o2, r5, c_odd, 3);
    o3 = vmlal_lane_s16(o3, r5, c_odd, 1);

    o0 = vmlal_lane_s16(o0, r7, c_odd, 3);
    o1 = vmlsl_lane_s16(o1, r7, c_odd, 2);
    o2 = vmlal_lane_s16(o2, r7, c_odd, 1);
    o3 = vmlsl_lane_s16(o3, r7, c_odd, 0);

    *y0 = vaddq_s32(e0, o0);
    *y1 = vaddq_s32(e1, o1);
    *y2 = vaddq_s32(e2, o2);
    *y3 = vaddq_s32(e3, o3);
    *y4 = vsubq_s32(e3, o3);
    *y5 = vsubq_s32(e2, o2);
    *y6 = vsubq_s32(e1, o1);
    *y7 = vsubq_s32(e0, o0);
}

/* Chroma direct reconstruction and interleaved storage for 8 pixels */
static inline void chroma_recon_store_8(int16x8_t recon_row,
                                        const UWORD16 *pu2_pred,
                                        UWORD16 *pu2_dst,
                                        int16x8_t clip_min,
                                        int16x8_t clip_max)
{
    uint16x8x2_t p = vld2q_u16(pu2_pred);
    int16x8_t a = vqaddq_s16(recon_row, vreinterpretq_s16_u16(p.val[0]));
    int16x8_t c = vminq_s16(vmaxq_s16(a, clip_min), clip_max);
    uint16x8x2_t d = vld2q_u16(pu2_dst);
    d.val[0] = vreinterpretq_u16_s16(c);
    vst2q_u16(pu2_dst, d);
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs HBD Inverse transform and reconstruction for 8x8
 *  chroma interleaved input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs 8x8 inverse DCT transform, adds the 16-bit interleaved chroma
 *  prediction data (U or V component at stride 2), and clips output to the
 *  configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 8x8 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 8x8 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 8x8 interleaved chroma block
 *
 * @param[out] pu2_dst
 *  Output 8x8 interleaved chroma block
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction stride
 *
 * @param[in] dst_strd
 *  Output Stride
 *
 * @param[in] zero_cols
 *  Zero columns in pi2_src
 *
 * @param[in] zero_rows
 *  Zero rows in pi2_src
 *
 * @param[in] u1_bit_depth
 *  Bit depth of the pixel data
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
void ihevc_hbd_chroma_itrans_recon_8x8_neonintr(WORD16 *pi2_src,
                                                WORD16 *pi2_tmp,
                                                UWORD16 *pu2_pred,
                                                UWORD16 *pu2_dst,
                                                WORD32 src_strd,
                                                WORD32 pred_strd,
                                                WORD32 dst_strd,
                                                WORD32 zero_cols,
                                                WORD32 zero_rows,
                                                UWORD8 u1_bit_depth)
{
    UNUSED(pi2_tmp);

    /* Fast Zero-block bypass: both zero_cols and zero_rows checked */
    if(((UWORD32)(zero_cols & 0xFF) == 0xFFU) || ((UWORD32)(zero_rows & 0xFF) == 0xFFU))
    {
        for(WORD32 i = 0; i < 8; i++)
        {
            uint16x8x2_t p = vld2q_u16(pu2_pred);
            uint16x8x2_t d = vld2q_u16(pu2_dst);
            d.val[0] = p.val[0];
            vst2q_u16(pu2_dst, d);
            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
        return;
    }

    static const int16_t a_even[4] = { 64, 83, 36, 0 };
    static const int16_t a_odd[4]  = { 89, 75, 50, 18 };
    int16x4_t c_even = vld1_s16(a_even);
    int16x4_t c_odd  = vld1_s16(a_odd);

    WORD32 check_row_stage_1 = ((zero_rows & 0xF0) != 0xF0) ? 1 : 0;
    WORD32 check_row_stage_2 = ((zero_cols & 0xF0) != 0xF0) ? 1 : 0;

    int16x8_t r0, r1, r2, r3;
    int16x8_t r4 = vdupq_n_s16(0);
    int16x8_t r5 = vdupq_n_s16(0);
    int16x8_t r6 = vdupq_n_s16(0);
    int16x8_t r7 = vdupq_n_s16(0);

    /* -------------------------------------------------------------------------
     * STAGE 1: Column transform on input coefficients
     * ------------------------------------------------------------------------- */
    int16x4_t s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo, s6_lo, s7_lo;
    int16x4_t s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi, s6_hi, s7_hi;

    if(!check_row_stage_2)
    {
        /* Only columns 0..3 have non-zero coefficients: load 64-bit vectors */
        s0_lo = vld1_s16(pi2_src + 0 * src_strd);
        s1_lo = vld1_s16(pi2_src + 1 * src_strd);
        s2_lo = vld1_s16(pi2_src + 2 * src_strd);
        s3_lo = vld1_s16(pi2_src + 3 * src_strd);

        if(check_row_stage_1)
        {
            s4_lo = vld1_s16(pi2_src + 4 * src_strd);
            s5_lo = vld1_s16(pi2_src + 5 * src_strd);
            s6_lo = vld1_s16(pi2_src + 6 * src_strd);
            s7_lo = vld1_s16(pi2_src + 7 * src_strd);
        }
    }
    else
    {
        /* All 8 columns active: load 128-bit vectors to halve memory load instructions */
        int16x8_t s0 = vld1q_s16(pi2_src + 0 * src_strd);
        int16x8_t s1 = vld1q_s16(pi2_src + 1 * src_strd);
        int16x8_t s2 = vld1q_s16(pi2_src + 2 * src_strd);
        int16x8_t s3 = vld1q_s16(pi2_src + 3 * src_strd);
        s0_lo = vget_low_s16(s0);  s0_hi = vget_high_s16(s0);
        s1_lo = vget_low_s16(s1);  s1_hi = vget_high_s16(s1);
        s2_lo = vget_low_s16(s2);  s2_hi = vget_high_s16(s2);
        s3_lo = vget_low_s16(s3);  s3_hi = vget_high_s16(s3);

        if(check_row_stage_1)
        {
            int16x8_t s4 = vld1q_s16(pi2_src + 4 * src_strd);
            int16x8_t s5 = vld1q_s16(pi2_src + 5 * src_strd);
            int16x8_t s6 = vld1q_s16(pi2_src + 6 * src_strd);
            int16x8_t s7 = vld1q_s16(pi2_src + 7 * src_strd);
            s4_lo = vget_low_s16(s4);  s4_hi = vget_high_s16(s4);
            s5_lo = vget_low_s16(s5);  s5_hi = vget_high_s16(s5);
            s6_lo = vget_low_s16(s6);  s6_hi = vget_high_s16(s6);
            s7_lo = vget_low_s16(s7);  s7_hi = vget_high_s16(s7);
        }
    }

    /* Transform columns 0..3 */
    int32x4_t y0, y1, y2, y3, y4, y5, y6, y7;
    if(!check_row_stage_1)
    {
        idct8_4col_zero4(s0_lo, s1_lo, s2_lo, s3_lo, c_even, c_odd,
                         &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7);
    }
    else
    {
        idct8_4col_full(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo, s6_lo, s7_lo, c_even, c_odd,
                        &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7);
    }

    int16x4_t q0 = vqrshrn_n_s32(y0, 7);
    int16x4_t q1 = vqrshrn_n_s32(y1, 7);
    int16x4_t q2 = vqrshrn_n_s32(y2, 7);
    int16x4_t q3 = vqrshrn_n_s32(y3, 7);
    int16x4_t q4 = vqrshrn_n_s32(y4, 7);
    int16x4_t q5 = vqrshrn_n_s32(y5, 7);
    int16x4_t q6 = vqrshrn_n_s32(y6, 7);
    int16x4_t q7 = vqrshrn_n_s32(y7, 7);

    transpose8x4(q0, q1, q2, q3, q4, q5, q6, q7, &r0, &r1, &r2, &r3);

    /* Transform columns 4..7 (if active) */
    if(check_row_stage_2)
    {
        if(!check_row_stage_1)
        {
            idct8_4col_zero4(s0_hi, s1_hi, s2_hi, s3_hi, c_even, c_odd,
                             &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7);
        }
        else
        {
            idct8_4col_full(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi, s6_hi, s7_hi, c_even, c_odd,
                            &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7);
        }

        q0 = vqrshrn_n_s32(y0, 7);
        q1 = vqrshrn_n_s32(y1, 7);
        q2 = vqrshrn_n_s32(y2, 7);
        q3 = vqrshrn_n_s32(y3, 7);
        q4 = vqrshrn_n_s32(y4, 7);
        q5 = vqrshrn_n_s32(y5, 7);
        q6 = vqrshrn_n_s32(y6, 7);
        q7 = vqrshrn_n_s32(y7, 7);

        transpose8x4(q0, q1, q2, q3, q4, q5, q6, q7, &r4, &r5, &r6, &r7);
    }

    /* -------------------------------------------------------------------------
     * STAGE 2 & RECONSTRUCTION: Decoupled into rows 0..3 and rows 4..7
     * ------------------------------------------------------------------------- */
    int32x4_t neg_shift = vdupq_n_s32(-(20 - u1_bit_depth));
    int16x8_t clip_min  = vdupq_n_s16(0);
    int16x8_t clip_max  = vdupq_n_s16((1 << u1_bit_depth) - 1);

    /* Stage 2 Pass 1: Rows 0..3 */
    int32x4_t z0, z1, z2, z3, z4, z5, z6, z7;
    if(!check_row_stage_2)
    {
        idct8_4col_zero4(vget_low_s16(r0), vget_low_s16(r1), vget_low_s16(r2), vget_low_s16(r3),
                         c_even, c_odd,
                         &z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    }
    else
    {
        idct8_4col_full(vget_low_s16(r0), vget_low_s16(r1), vget_low_s16(r2), vget_low_s16(r3),
                        vget_low_s16(r4), vget_low_s16(r5), vget_low_s16(r6), vget_low_s16(r7),
                        c_even, c_odd,
                        &z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    }

    q0 = vqmovn_s32(vrshlq_s32(z0, neg_shift));
    q1 = vqmovn_s32(vrshlq_s32(z1, neg_shift));
    q2 = vqmovn_s32(vrshlq_s32(z2, neg_shift));
    q3 = vqmovn_s32(vrshlq_s32(z3, neg_shift));
    q4 = vqmovn_s32(vrshlq_s32(z4, neg_shift));
    q5 = vqmovn_s32(vrshlq_s32(z5, neg_shift));
    q6 = vqmovn_s32(vrshlq_s32(z6, neg_shift));
    q7 = vqmovn_s32(vrshlq_s32(z7, neg_shift));

    int16x8_t out0, out1, out2, out3;
    transpose8x4(q0, q1, q2, q3, q4, q5, q6, q7, &out0, &out1, &out2, &out3);

    chroma_recon_store_8(out0, pu2_pred + 0 * pred_strd, pu2_dst + 0 * dst_strd, clip_min, clip_max);
    chroma_recon_store_8(out1, pu2_pred + 1 * pred_strd, pu2_dst + 1 * dst_strd, clip_min, clip_max);
    chroma_recon_store_8(out2, pu2_pred + 2 * pred_strd, pu2_dst + 2 * dst_strd, clip_min, clip_max);
    chroma_recon_store_8(out3, pu2_pred + 3 * pred_strd, pu2_dst + 3 * dst_strd, clip_min, clip_max);

    /* Stage 2 Pass 2: Rows 4..7 (reusing vector registers z0..z7 and out0..out3) */
    if(!check_row_stage_2)
    {
        idct8_4col_zero4(vget_high_s16(r0), vget_high_s16(r1), vget_high_s16(r2), vget_high_s16(r3),
                         c_even, c_odd,
                         &z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    }
    else
    {
        idct8_4col_full(vget_high_s16(r0), vget_high_s16(r1), vget_high_s16(r2), vget_high_s16(r3),
                        vget_high_s16(r4), vget_high_s16(r5), vget_high_s16(r6), vget_high_s16(r7),
                        c_even, c_odd,
                        &z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    }

    q0 = vqmovn_s32(vrshlq_s32(z0, neg_shift));
    q1 = vqmovn_s32(vrshlq_s32(z1, neg_shift));
    q2 = vqmovn_s32(vrshlq_s32(z2, neg_shift));
    q3 = vqmovn_s32(vrshlq_s32(z3, neg_shift));
    q4 = vqmovn_s32(vrshlq_s32(z4, neg_shift));
    q5 = vqmovn_s32(vrshlq_s32(z5, neg_shift));
    q6 = vqmovn_s32(vrshlq_s32(z6, neg_shift));
    q7 = vqmovn_s32(vrshlq_s32(z7, neg_shift));

    transpose8x4(q0, q1, q2, q3, q4, q5, q6, q7, &out0, &out1, &out2, &out3);

    chroma_recon_store_8(out0, pu2_pred + 4 * pred_strd, pu2_dst + 4 * dst_strd, clip_min, clip_max);
    chroma_recon_store_8(out1, pu2_pred + 5 * pred_strd, pu2_dst + 5 * dst_strd, clip_min, clip_max);
    chroma_recon_store_8(out2, pu2_pred + 6 * pred_strd, pu2_dst + 6 * dst_strd, clip_min, clip_max);
    chroma_recon_store_8(out3, pu2_pred + 7 * pred_strd, pu2_dst + 7 * dst_strd, clip_min, clip_max);
}
