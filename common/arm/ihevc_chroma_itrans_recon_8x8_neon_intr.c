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
 *  ihevc_chroma_itrans_recon_8x8_neon_intr.c
 *
 * @brief
 *  Contains optimized function definitions for 8-bit inverse transform
 *  and reconstruction of chroma interleaved data 8x8 using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_chroma_itrans_recon_8x8_neonintr()
 *
 * @remarks
 *  None
 *
 ******************************************************************************
 */
#include <arm_neon.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_trans_macros.h"

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs 8-bit Inverse transform and reconstruction for 8x8
 *  chroma interleaved input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs 8x8 inverse DCT transform, adds the 8-bit interleaved chroma
 *  prediction data (U or V component at stride 2), and clips output to [0, 255]
 *
 * @param[in] pi2_src
 *  Input 8x8 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 8x8 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu1_pred
 *  Prediction 8x8 interleaved chroma block
 *
 * @param[out] pu1_dst
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
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
void ihevc_chroma_itrans_recon_8x8_neonintr(WORD16 *pi2_src,
                                            WORD16 *pi2_tmp,
                                            UWORD8 *pu1_pred,
                                            UWORD8 *pu1_dst,
                                            WORD32 src_strd,
                                            WORD32 pred_strd,
                                            WORD32 dst_strd,
                                            WORD32 zero_cols,
                                            WORD32 zero_rows)
{
    WORD32 check_row_stage_1;
    WORD32 check_row_stage_2;
    int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    int16x8_t dst_row0, dst_row1, dst_row2, dst_row3;
    int16x8_t dst_row4, dst_row5, dst_row6, dst_row7;
    int16x4x2_t tr0, tr1;
    int32x2x2_t tr2, tr3;
    uint16x8_t mask_00ff = vdupq_n_u16(0x00FF);
    uint8x16_t blend_mask = vreinterpretq_u8_u16(mask_00ff);

    UNUSED(pi2_tmp);

    /* Fast zero-block bypass: avoids transform computation when block has zero columns or rows */
    if(((UWORD32)(zero_cols & 0xFF) == 0xFFU) || ((UWORD32)(zero_rows & 0xFF) == 0xFFU))
    {
        WORD32 i;
        for(i = 0; i < 8; i++)
        {
            uint8x16_t p = vld1q_u8(pu1_pred);
            uint8x16_t d = vld1q_u8(pu1_dst);
            vst1q_u8(pu1_dst, vbslq_u8(blend_mask, p, d));
            pu1_pred += pred_strd;
            pu1_dst += dst_strd;
        }
        return;
    }

    check_row_stage_1 = ((zero_rows & 0xF0) != 0xF0) ? 1 : 0;
    check_row_stage_2 = ((zero_cols & 0xF0) != 0xF0) ? 1 : 0;

    if(!check_row_stage_2)
    {
        int16x4_t row0_lo, row1_lo, row2_lo, row3_lo;
        int16x4_t row0_hi, row1_hi, row2_hi, row3_hi;
        int16x4_t t0, t1, t2, t3, t4, t5, t6, t7;
        int32x4_t ee0, ee1, eo0, eo1;
        int32x4_t e0, e1, e2, e3;
        int32x4_t o0, o1, o2, o3;

        /* ---------------- Stage 1 (4 columns) ---------------- */
        if(!check_row_stage_1)
        {
            int16x4_t r0 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r1 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r2 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r3 = vld1_s16(pi2_src);

            /* ee */
            ee0 = vshll_n_s16(r0, 6);

            /* eo */
            eo0 = vmull_n_s16(r2, 83);
            eo1 = vmull_n_s16(r2, 36);

            /* e */
            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee0, eo1);
            e2 = vsubq_s32(ee0, eo1);

            /* o */
            o0 = vmlal_n_s16(vmull_n_s16(r1, 89), r3, 75);
            o1 = vmlsl_n_s16(vmull_n_s16(r1, 75), r3, 18);
            o2 = vmlsl_n_s16(vmull_n_s16(r1, 50), r3, 89);
            o3 = vmlsl_n_s16(vmull_n_s16(r1, 18), r3, 50);
        }
        else
        {
            int16x4_t r0 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r1 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r2 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r3 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r4 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r5 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r6 = vld1_s16(pi2_src);
            pi2_src += src_strd;
            int16x4_t r7 = vld1_s16(pi2_src);

            /* ee */
            ee0 = vshlq_n_s32(vaddl_s16(r0, r4), 6);
            ee1 = vshlq_n_s32(vsubl_s16(r0, r4), 6);

            /* eo */
            eo0 = vmlal_n_s16(vmull_n_s16(r2, 83), r6, 36);
            eo1 = vmlsl_n_s16(vmull_n_s16(r2, 36), r6, 83);

            /* e */
            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee1, eo1);
            e2 = vsubq_s32(ee1, eo1);

            /* o */
            o0 = vmlal_n_s16(vmlal_n_s16(vmlal_n_s16(vmull_n_s16(r1, 89), r3, 75), r5, 50), r7, 18);
            o1 = vmlsl_n_s16(vmlsl_n_s16(vmlsl_n_s16(vmull_n_s16(r1, 75), r3, 18), r5, 89), r7, 50);
            o2 = vmlal_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(r1, 50), r3, 89), r5, 18), r7, 75);
            o3 = vmlsl_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(r1, 18), r3, 50), r5, 75), r7, 89);
        }

        t0 = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_1);
        t1 = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_1);
        t2 = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_1);
        t3 = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_1);
        t4 = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_1);
        t5 = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_1);
        t6 = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_1);
        t7 = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_1);

        /* Transpose 8x4 Stage 1 output into 4x8 (row0..row3) */
        tr0 = vtrn_s16(t0, t1);
        tr1 = vtrn_s16(t2, t3);
        tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
        tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
        row0_lo = vreinterpret_s16_s32(tr2.val[0]);
        row1_lo = vreinterpret_s16_s32(tr3.val[0]);
        row2_lo = vreinterpret_s16_s32(tr2.val[1]);
        row3_lo = vreinterpret_s16_s32(tr3.val[1]);

        tr0 = vtrn_s16(t4, t5);
        tr1 = vtrn_s16(t6, t7);
        tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
        tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
        row0_hi = vreinterpret_s16_s32(tr2.val[0]);
        row1_hi = vreinterpret_s16_s32(tr3.val[0]);
        row2_hi = vreinterpret_s16_s32(tr2.val[1]);
        row3_hi = vreinterpret_s16_s32(tr3.val[1]);

        /* ---------------- Stage 2 (4 active rows, 8 columns) ---------------- */
        {
            int16x4_t s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo, s6_lo, s7_lo;
            int16x4_t s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi, s6_hi, s7_hi;

            /* Low 4 columns */
            ee0 = vshll_n_s16(row0_lo, 6);
            eo0 = vmull_n_s16(row2_lo, 83);
            eo1 = vmull_n_s16(row2_lo, 36);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee0, eo1);
            e2 = vsubq_s32(ee0, eo1);

            o0 = vmlal_n_s16(vmull_n_s16(row1_lo, 89), row3_lo, 75);
            o1 = vmlsl_n_s16(vmull_n_s16(row1_lo, 75), row3_lo, 18);
            o2 = vmlsl_n_s16(vmull_n_s16(row1_lo, 50), row3_lo, 89);
            o3 = vmlsl_n_s16(vmull_n_s16(row1_lo, 18), row3_lo, 50);

            s0_lo = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_2);
            s1_lo = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s2_lo = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s3_lo = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s4_lo = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s5_lo = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s6_lo = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s7_lo = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_2);

            /* High 4 columns */
            ee0 = vshll_n_s16(row0_hi, 6);
            eo0 = vmull_n_s16(row2_hi, 83);
            eo1 = vmull_n_s16(row2_hi, 36);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee0, eo1);
            e2 = vsubq_s32(ee0, eo1);

            o0 = vmlal_n_s16(vmull_n_s16(row1_hi, 89), row3_hi, 75);
            o1 = vmlsl_n_s16(vmull_n_s16(row1_hi, 75), row3_hi, 18);
            o2 = vmlsl_n_s16(vmull_n_s16(row1_hi, 50), row3_hi, 89);
            o3 = vmlsl_n_s16(vmull_n_s16(row1_hi, 18), row3_hi, 50);

            s0_hi = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_2);
            s1_hi = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s2_hi = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s3_hi = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s4_hi = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s5_hi = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s6_hi = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s7_hi = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_2);

            s0 = vcombine_s16(s0_lo, s0_hi);
            s1 = vcombine_s16(s1_lo, s1_hi);
            s2 = vcombine_s16(s2_lo, s2_hi);
            s3 = vcombine_s16(s3_lo, s3_hi);
            s4 = vcombine_s16(s4_lo, s4_hi);
            s5 = vcombine_s16(s5_lo, s5_hi);
            s6 = vcombine_s16(s6_lo, s6_hi);
            s7 = vcombine_s16(s7_lo, s7_hi);
        }
    }
    else
    {
        int16x4_t row0_lo, row1_lo, row2_lo, row3_lo, row4_lo, row5_lo, row6_lo, row7_lo;
        int16x4_t row0_hi, row1_hi, row2_hi, row3_hi, row4_hi, row5_hi, row6_hi, row7_hi;
        int16x4_t t0_lo, t1_lo, t2_lo, t3_lo, t4_lo, t5_lo, t6_lo, t7_lo;
        int16x4_t t0_hi, t1_hi, t2_hi, t3_hi, t4_hi, t5_hi, t6_hi, t7_hi;
        int32x4_t ee0, ee1, eo0, eo1;
        int32x4_t e0, e1, e2, e3;
        int32x4_t o0, o1, o2, o3;

        /* ---------------- Stage 1 (8 columns) ---------------- */
        if(!check_row_stage_1)
        {
            int16x8_t r0 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r1 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r2 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r3 = vld1q_s16(pi2_src);

            int16x4_t r0_lo = vget_low_s16(r0);
            int16x4_t r1_lo = vget_low_s16(r1);
            int16x4_t r2_lo = vget_low_s16(r2);
            int16x4_t r3_lo = vget_low_s16(r3);

            int16x4_t r0_hi = vget_high_s16(r0);
            int16x4_t r1_hi = vget_high_s16(r1);
            int16x4_t r2_hi = vget_high_s16(r2);
            int16x4_t r3_hi = vget_high_s16(r3);

            /* Low 4 columns */
            ee0 = vshll_n_s16(r0_lo, 6);
            eo0 = vmull_n_s16(r2_lo, 83);
            eo1 = vmull_n_s16(r2_lo, 36);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee0, eo1);
            e2 = vsubq_s32(ee0, eo1);

            o0 = vmlal_n_s16(vmull_n_s16(r1_lo, 89), r3_lo, 75);
            o1 = vmlsl_n_s16(vmull_n_s16(r1_lo, 75), r3_lo, 18);
            o2 = vmlsl_n_s16(vmull_n_s16(r1_lo, 50), r3_lo, 89);
            o3 = vmlsl_n_s16(vmull_n_s16(r1_lo, 18), r3_lo, 50);

            t0_lo = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_1);
            t1_lo = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t2_lo = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t3_lo = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t4_lo = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t5_lo = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t6_lo = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t7_lo = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_1);

            /* High 4 columns */
            ee0 = vshll_n_s16(r0_hi, 6);
            eo0 = vmull_n_s16(r2_hi, 83);
            eo1 = vmull_n_s16(r2_hi, 36);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee0, eo1);
            e2 = vsubq_s32(ee0, eo1);

            o0 = vmlal_n_s16(vmull_n_s16(r1_hi, 89), r3_hi, 75);
            o1 = vmlsl_n_s16(vmull_n_s16(r1_hi, 75), r3_hi, 18);
            o2 = vmlsl_n_s16(vmull_n_s16(r1_hi, 50), r3_hi, 89);
            o3 = vmlsl_n_s16(vmull_n_s16(r1_hi, 18), r3_hi, 50);

            t0_hi = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_1);
            t1_hi = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t2_hi = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t3_hi = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t4_hi = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t5_hi = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t6_hi = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t7_hi = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_1);
        }
        else
        {
            int16x8_t r0 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r1 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r2 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r3 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r4 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r5 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r6 = vld1q_s16(pi2_src);
            pi2_src += src_strd;
            int16x8_t r7 = vld1q_s16(pi2_src);

            int16x4_t r0_lo = vget_low_s16(r0);
            int16x4_t r1_lo = vget_low_s16(r1);
            int16x4_t r2_lo = vget_low_s16(r2);
            int16x4_t r3_lo = vget_low_s16(r3);
            int16x4_t r4_lo = vget_low_s16(r4);
            int16x4_t r5_lo = vget_low_s16(r5);
            int16x4_t r6_lo = vget_low_s16(r6);
            int16x4_t r7_lo = vget_low_s16(r7);

            int16x4_t r0_hi = vget_high_s16(r0);
            int16x4_t r1_hi = vget_high_s16(r1);
            int16x4_t r2_hi = vget_high_s16(r2);
            int16x4_t r3_hi = vget_high_s16(r3);
            int16x4_t r4_hi = vget_high_s16(r4);
            int16x4_t r5_hi = vget_high_s16(r5);
            int16x4_t r6_hi = vget_high_s16(r6);
            int16x4_t r7_hi = vget_high_s16(r7);

            /* Low 4 columns */
            ee0 = vshlq_n_s32(vaddl_s16(r0_lo, r4_lo), 6);
            ee1 = vshlq_n_s32(vsubl_s16(r0_lo, r4_lo), 6);
            eo0 = vmlal_n_s16(vmull_n_s16(r2_lo, 83), r6_lo, 36);
            eo1 = vmlsl_n_s16(vmull_n_s16(r2_lo, 36), r6_lo, 83);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee1, eo1);
            e2 = vsubq_s32(ee1, eo1);

            o0 = vmlal_n_s16(vmlal_n_s16(vmlal_n_s16(vmull_n_s16(r1_lo, 89), r3_lo, 75), r5_lo, 50), r7_lo, 18);
            o1 = vmlsl_n_s16(vmlsl_n_s16(vmlsl_n_s16(vmull_n_s16(r1_lo, 75), r3_lo, 18), r5_lo, 89), r7_lo, 50);
            o2 = vmlal_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(r1_lo, 50), r3_lo, 89), r5_lo, 18), r7_lo, 75);
            o3 = vmlsl_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(r1_lo, 18), r3_lo, 50), r5_lo, 75), r7_lo, 89);

            t0_lo = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_1);
            t1_lo = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t2_lo = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t3_lo = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t4_lo = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t5_lo = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t6_lo = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t7_lo = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_1);

            /* High 4 columns */
            ee0 = vshlq_n_s32(vaddl_s16(r0_hi, r4_hi), 6);
            ee1 = vshlq_n_s32(vsubl_s16(r0_hi, r4_hi), 6);
            eo0 = vmlal_n_s16(vmull_n_s16(r2_hi, 83), r6_hi, 36);
            eo1 = vmlsl_n_s16(vmull_n_s16(r2_hi, 36), r6_hi, 83);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee1, eo1);
            e2 = vsubq_s32(ee1, eo1);

            o0 = vmlal_n_s16(vmlal_n_s16(vmlal_n_s16(vmull_n_s16(r1_hi, 89), r3_hi, 75), r5_hi, 50), r7_hi, 18);
            o1 = vmlsl_n_s16(vmlsl_n_s16(vmlsl_n_s16(vmull_n_s16(r1_hi, 75), r3_hi, 18), r5_hi, 89), r7_hi, 50);
            o2 = vmlal_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(r1_hi, 50), r3_hi, 89), r5_hi, 18), r7_hi, 75);
            o3 = vmlsl_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(r1_hi, 18), r3_hi, 50), r5_hi, 75), r7_hi, 89);

            t0_hi = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_1);
            t1_hi = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t2_hi = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t3_hi = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t4_hi = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_1);
            t5_hi = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_1);
            t6_hi = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_1);
            t7_hi = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_1);
        }

        /* Transpose 8x8 Stage 1 output into row0..row7 (lo and hi) */
        tr0 = vtrn_s16(t0_lo, t1_lo);
        tr1 = vtrn_s16(t2_lo, t3_lo);
        tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
        tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
        row0_lo = vreinterpret_s16_s32(tr2.val[0]);
        row1_lo = vreinterpret_s16_s32(tr3.val[0]);
        row2_lo = vreinterpret_s16_s32(tr2.val[1]);
        row3_lo = vreinterpret_s16_s32(tr3.val[1]);

        tr0 = vtrn_s16(t4_lo, t5_lo);
        tr1 = vtrn_s16(t6_lo, t7_lo);
        tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
        tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
        row0_hi = vreinterpret_s16_s32(tr2.val[0]);
        row1_hi = vreinterpret_s16_s32(tr3.val[0]);
        row2_hi = vreinterpret_s16_s32(tr2.val[1]);
        row3_hi = vreinterpret_s16_s32(tr3.val[1]);

        tr0 = vtrn_s16(t0_hi, t1_hi);
        tr1 = vtrn_s16(t2_hi, t3_hi);
        tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
        tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
        row4_lo = vreinterpret_s16_s32(tr2.val[0]);
        row5_lo = vreinterpret_s16_s32(tr3.val[0]);
        row6_lo = vreinterpret_s16_s32(tr2.val[1]);
        row7_lo = vreinterpret_s16_s32(tr3.val[1]);

        tr0 = vtrn_s16(t4_hi, t5_hi);
        tr1 = vtrn_s16(t6_hi, t7_hi);
        tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
        tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
        row4_hi = vreinterpret_s16_s32(tr2.val[0]);
        row5_hi = vreinterpret_s16_s32(tr3.val[0]);
        row6_hi = vreinterpret_s16_s32(tr2.val[1]);
        row7_hi = vreinterpret_s16_s32(tr3.val[1]);

        /* ---------------- Stage 2 (8 active rows, 8 columns) ---------------- */
        {
            int16x4_t s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, s5_lo, s6_lo, s7_lo;
            int16x4_t s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, s5_hi, s6_hi, s7_hi;

            /* Low 4 columns */
            ee0 = vshlq_n_s32(vaddl_s16(row0_lo, row4_lo), 6);
            ee1 = vshlq_n_s32(vsubl_s16(row0_lo, row4_lo), 6);
            eo0 = vmlal_n_s16(vmull_n_s16(row2_lo, 83), row6_lo, 36);
            eo1 = vmlsl_n_s16(vmull_n_s16(row2_lo, 36), row6_lo, 83);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee1, eo1);
            e2 = vsubq_s32(ee1, eo1);

            o0 = vmlal_n_s16(vmlal_n_s16(vmlal_n_s16(vmull_n_s16(row1_lo, 89), row3_lo, 75), row5_lo, 50), row7_lo, 18);
            o1 = vmlsl_n_s16(vmlsl_n_s16(vmlsl_n_s16(vmull_n_s16(row1_lo, 75), row3_lo, 18), row5_lo, 89), row7_lo, 50);
            o2 = vmlal_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(row1_lo, 50), row3_lo, 89), row5_lo, 18), row7_lo, 75);
            o3 = vmlsl_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(row1_lo, 18), row3_lo, 50), row5_lo, 75), row7_lo, 89);

            s0_lo = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_2);
            s1_lo = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s2_lo = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s3_lo = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s4_lo = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s5_lo = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s6_lo = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s7_lo = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_2);

            /* High 4 columns */
            ee0 = vshlq_n_s32(vaddl_s16(row0_hi, row4_hi), 6);
            ee1 = vshlq_n_s32(vsubl_s16(row0_hi, row4_hi), 6);
            eo0 = vmlal_n_s16(vmull_n_s16(row2_hi, 83), row6_hi, 36);
            eo1 = vmlsl_n_s16(vmull_n_s16(row2_hi, 36), row6_hi, 83);

            e0 = vaddq_s32(ee0, eo0);
            e3 = vsubq_s32(ee0, eo0);
            e1 = vaddq_s32(ee1, eo1);
            e2 = vsubq_s32(ee1, eo1);

            o0 = vmlal_n_s16(vmlal_n_s16(vmlal_n_s16(vmull_n_s16(row1_hi, 89), row3_hi, 75), row5_hi, 50), row7_hi, 18);
            o1 = vmlsl_n_s16(vmlsl_n_s16(vmlsl_n_s16(vmull_n_s16(row1_hi, 75), row3_hi, 18), row5_hi, 89), row7_hi, 50);
            o2 = vmlal_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(row1_hi, 50), row3_hi, 89), row5_hi, 18), row7_hi, 75);
            o3 = vmlsl_n_s16(vmlal_n_s16(vmlsl_n_s16(vmull_n_s16(row1_hi, 18), row3_hi, 50), row5_hi, 75), row7_hi, 89);

            s0_hi = vqrshrn_n_s32(vaddq_s32(e0, o0), IT_SHIFT_STAGE_2);
            s1_hi = vqrshrn_n_s32(vaddq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s2_hi = vqrshrn_n_s32(vaddq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s3_hi = vqrshrn_n_s32(vaddq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s4_hi = vqrshrn_n_s32(vsubq_s32(e3, o3), IT_SHIFT_STAGE_2);
            s5_hi = vqrshrn_n_s32(vsubq_s32(e2, o2), IT_SHIFT_STAGE_2);
            s6_hi = vqrshrn_n_s32(vsubq_s32(e1, o1), IT_SHIFT_STAGE_2);
            s7_hi = vqrshrn_n_s32(vsubq_s32(e0, o0), IT_SHIFT_STAGE_2);

            s0 = vcombine_s16(s0_lo, s0_hi);
            s1 = vcombine_s16(s1_lo, s1_hi);
            s2 = vcombine_s16(s2_lo, s2_hi);
            s3 = vcombine_s16(s3_lo, s3_hi);
            s4 = vcombine_s16(s4_lo, s4_hi);
            s5 = vcombine_s16(s5_lo, s5_hi);
            s6 = vcombine_s16(s6_lo, s6_hi);
            s7 = vcombine_s16(s7_lo, s7_hi);
        }
    }

    /* ---------------- 8x8 Matrix Transposition ---------------- */
    {
        int16x8x2_t t01 = vtrnq_s16(s0, s1);
        int16x8x2_t t23 = vtrnq_s16(s2, s3);
        int16x8x2_t t45 = vtrnq_s16(s4, s5);
        int16x8x2_t t67 = vtrnq_s16(s6, s7);

        int32x4x2_t u02 = vtrnq_s32(vreinterpretq_s32_s16(t01.val[0]), vreinterpretq_s32_s16(t23.val[0]));
        int32x4x2_t u13 = vtrnq_s32(vreinterpretq_s32_s16(t01.val[1]), vreinterpretq_s32_s16(t23.val[1]));
        int32x4x2_t u46 = vtrnq_s32(vreinterpretq_s32_s16(t45.val[0]), vreinterpretq_s32_s16(t67.val[0]));
        int32x4x2_t u57 = vtrnq_s32(vreinterpretq_s32_s16(t45.val[1]), vreinterpretq_s32_s16(t67.val[1]));

        dst_row0 = vcombine_s16(vget_low_s16(vreinterpretq_s16_s32(u02.val[0])), vget_low_s16(vreinterpretq_s16_s32(u46.val[0])));
        dst_row1 = vcombine_s16(vget_low_s16(vreinterpretq_s16_s32(u13.val[0])), vget_low_s16(vreinterpretq_s16_s32(u57.val[0])));
        dst_row2 = vcombine_s16(vget_low_s16(vreinterpretq_s16_s32(u02.val[1])), vget_low_s16(vreinterpretq_s16_s32(u46.val[1])));
        dst_row3 = vcombine_s16(vget_low_s16(vreinterpretq_s16_s32(u13.val[1])), vget_low_s16(vreinterpretq_s16_s32(u57.val[1])));

        dst_row4 = vcombine_s16(vget_high_s16(vreinterpretq_s16_s32(u02.val[0])), vget_high_s16(vreinterpretq_s16_s32(u46.val[0])));
        dst_row5 = vcombine_s16(vget_high_s16(vreinterpretq_s16_s32(u13.val[0])), vget_high_s16(vreinterpretq_s16_s32(u57.val[0])));
        dst_row6 = vcombine_s16(vget_high_s16(vreinterpretq_s16_s32(u02.val[1])), vget_high_s16(vreinterpretq_s16_s32(u46.val[1])));
        dst_row7 = vcombine_s16(vget_high_s16(vreinterpretq_s16_s32(u13.val[1])), vget_high_s16(vreinterpretq_s16_s32(u57.val[1])));
    }

    /* ---------------- Recon & Store ---------------- */
    {
        int16x8_t clip_min = vdupq_n_s16(0);
        int16x8_t clip_max = vreinterpretq_s16_u16(mask_00ff);

        uint16x8_t pred_0 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);
        pu1_pred += pred_strd;
        uint16x8_t pred_1 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);
        pu1_pred += pred_strd;
        uint16x8_t pred_2 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);
        pu1_pred += pred_strd;
        uint16x8_t pred_3 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);
        pu1_pred += pred_strd;
        uint16x8_t pred_4 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);
        pu1_pred += pred_strd;
        uint16x8_t pred_5 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);
        pu1_pred += pred_strd;
        uint16x8_t pred_6 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);
        pu1_pred += pred_strd;
        uint16x8_t pred_7 = vandq_u16(vreinterpretq_u16_u8(vld1q_u8(pu1_pred)), mask_00ff);

        int16x8_t recon_0 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row0, vreinterpretq_s16_u16(pred_0)), clip_min), clip_max);
        int16x8_t recon_1 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row1, vreinterpretq_s16_u16(pred_1)), clip_min), clip_max);
        int16x8_t recon_2 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row2, vreinterpretq_s16_u16(pred_2)), clip_min), clip_max);
        int16x8_t recon_3 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row3, vreinterpretq_s16_u16(pred_3)), clip_min), clip_max);
        int16x8_t recon_4 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row4, vreinterpretq_s16_u16(pred_4)), clip_min), clip_max);
        int16x8_t recon_5 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row5, vreinterpretq_s16_u16(pred_5)), clip_min), clip_max);
        int16x8_t recon_6 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row6, vreinterpretq_s16_u16(pred_6)), clip_min), clip_max);
        int16x8_t recon_7 = vminq_s16(vmaxq_s16(vqaddq_s16(dst_row7, vreinterpretq_s16_u16(pred_7)), clip_min), clip_max);

        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_0), vld1q_u8(pu1_dst)));
        pu1_dst += dst_strd;
        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_1), vld1q_u8(pu1_dst)));
        pu1_dst += dst_strd;
        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_2), vld1q_u8(pu1_dst)));
        pu1_dst += dst_strd;
        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_3), vld1q_u8(pu1_dst)));
        pu1_dst += dst_strd;
        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_4), vld1q_u8(pu1_dst)));
        pu1_dst += dst_strd;
        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_5), vld1q_u8(pu1_dst)));
        pu1_dst += dst_strd;
        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_6), vld1q_u8(pu1_dst)));
        pu1_dst += dst_strd;
        vst1q_u8(pu1_dst, vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_7), vld1q_u8(pu1_dst)));
    }
}
