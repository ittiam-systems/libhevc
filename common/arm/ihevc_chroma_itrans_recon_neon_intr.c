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
 *  ihevc_chroma_itrans_recon_neon_intr.c
 *
 * @brief
 *  Contains optimized function definitions for 8-bit inverse transform
 *  and reconstruction of chroma interleaved data using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_chroma_itrans_recon_4x4_neonintr()
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
 *  This function performs 8-bit Inverse transform and reconstruction for 4x4
 *  chroma interleaved input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs 4x4 inverse DCT transform, adds the 8-bit interleaved chroma
 *  prediction data (U or V component at stride 2), and clips output to [0, 255]
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu1_pred
 *  Prediction 4x4 interleaved chroma block
 *
 * @param[out] pu1_dst
 *  Output 4x4 interleaved chroma block
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
void ihevc_chroma_itrans_recon_4x4_neonintr(WORD16 *pi2_src,
                                            WORD16 *pi2_tmp,
                                            UWORD8 *pu1_pred,
                                            UWORD8 *pu1_dst,
                                            WORD32 src_strd,
                                            WORD32 pred_strd,
                                            WORD32 dst_strd,
                                            WORD32 zero_cols,
                                            WORD32 zero_rows)
{
    UNUSED(pi2_tmp);
    uint16x8_t mask_00ff = vdupq_n_u16(0x00FF);
    uint8x16_t blend_mask = vreinterpretq_u8_u16(mask_00ff);

    /* Fast zero-block bypass: avoids transform computation when block has zero columns or rows */
    if(((UWORD32)(zero_cols & 0xF) == 0xFU) || ((UWORD32)(zero_rows & 0xF) == 0xFU))
    {
        uint8x8_t p0 = vld1_u8(pu1_pred);
        uint8x8_t p1 = vld1_u8(pu1_pred + pred_strd);
        uint8x8_t p2 = vld1_u8(pu1_pred + 2 * pred_strd);
        uint8x8_t p3 = vld1_u8(pu1_pred + 3 * pred_strd);

        uint8x8_t d0 = vld1_u8(pu1_dst);
        uint8x8_t d1 = vld1_u8(pu1_dst + dst_strd);
        uint8x8_t d2 = vld1_u8(pu1_dst + 2 * dst_strd);
        uint8x8_t d3 = vld1_u8(pu1_dst + 3 * dst_strd);

        uint8x16_t out_01 = vbslq_u8(blend_mask, vcombine_u8(p0, p1), vcombine_u8(d0, d1));
        uint8x16_t out_23 = vbslq_u8(blend_mask, vcombine_u8(p2, p3), vcombine_u8(d2, d3));

        vst1_u8(pu1_dst, vget_low_u8(out_01));
        vst1_u8(pu1_dst + dst_strd, vget_high_u8(out_01));
        vst1_u8(pu1_dst + 2 * dst_strd, vget_low_u8(out_23));
        vst1_u8(pu1_dst + 3 * dst_strd, vget_high_u8(out_23));
        return;
    }

    /* ---------------- Stage 1 ---------------- */
    int16x4_t r0 = vld1_s16(pi2_src);
    pi2_src += src_strd;
    int16x4_t r1 = vld1_s16(pi2_src);
    pi2_src += src_strd;
    int16x4_t r2 = vld1_s16(pi2_src);
    pi2_src += src_strd;
    int16x4_t r3 = vld1_s16(pi2_src);

    /* Interleaved butterfly to maximize dual-issue ALU and multiply execution */
    int32x4_t sum_r02  = vaddl_s16(r0, r2);
    int32x4_t diff_r02 = vsubl_s16(r0, r2);
    int32x4_t o0_part  = vmull_n_s16(r1, 83);
    int32x4_t o1_part  = vmull_n_s16(r1, 36);

    int32x4_t e0 = vshlq_n_s32(sum_r02, 6);
    int32x4_t e1 = vshlq_n_s32(diff_r02, 6);
    int32x4_t o0 = vmlal_n_s16(o0_part, r3, 36);
    int32x4_t o1 = vmlsl_n_s16(o1_part, r3, 83);

    int32x4_t out0 = vaddq_s32(e0, o0);
    int32x4_t out1 = vaddq_s32(e1, o1);
    int32x4_t out2 = vsubq_s32(e1, o1);
    int32x4_t out3 = vsubq_s32(e0, o0);

    /* Stage 1 shift: 7 with rounding & saturation to int16 */
    int16x4_t t0 = vqrshrn_n_s32(out0, IT_SHIFT_STAGE_1);
    int16x4_t t1 = vqrshrn_n_s32(out1, IT_SHIFT_STAGE_1);
    int16x4_t t2 = vqrshrn_n_s32(out2, IT_SHIFT_STAGE_1);
    int16x4_t t3 = vqrshrn_n_s32(out3, IT_SHIFT_STAGE_1);

    /* In-Register 4x4 Transposition (Stage 1) */
    int16x4x2_t tr0 = vtrn_s16(t0, t1);
    int16x4x2_t tr1 = vtrn_s16(t2, t3);
    int32x2x2_t tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
    int32x2x2_t tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
    int16x4_t row0 = vreinterpret_s16_s32(tr2.val[0]);
    int16x4_t row1 = vreinterpret_s16_s32(tr3.val[0]);
    int16x4_t row2 = vreinterpret_s16_s32(tr2.val[1]);
    int16x4_t row3 = vreinterpret_s16_s32(tr3.val[1]);

    /* ---------------- Stage 2 ---------------- */
    sum_r02  = vaddl_s16(row0, row2);
    diff_r02 = vsubl_s16(row0, row2);
    o0_part  = vmull_n_s16(row1, 83);
    o1_part  = vmull_n_s16(row1, 36);

    e0 = vshlq_n_s32(sum_r02, 6);
    e1 = vshlq_n_s32(diff_r02, 6);
    o0 = vmlal_n_s16(o0_part, row3, 36);
    o1 = vmlsl_n_s16(o1_part, row3, 83);

    out0 = vaddq_s32(e0, o0);
    out1 = vaddq_s32(e1, o1);
    out2 = vsubq_s32(e1, o1);
    out3 = vsubq_s32(e0, o0);

    /* Stage 2 shift: 12 with rounding & saturation to int16 */
    int16x4_t s0 = vqrshrn_n_s32(out0, IT_SHIFT_STAGE_2);
    int16x4_t s1 = vqrshrn_n_s32(out1, IT_SHIFT_STAGE_2);
    int16x4_t s2 = vqrshrn_n_s32(out2, IT_SHIFT_STAGE_2);
    int16x4_t s3 = vqrshrn_n_s32(out3, IT_SHIFT_STAGE_2);

    /* In-Register 4x4 Transposition (Stage 2) */
    tr0 = vtrn_s16(s0, s1);
    tr1 = vtrn_s16(s2, s3);
    tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
    tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
    int16x4_t dst_row0 = vreinterpret_s16_s32(tr2.val[0]);
    int16x4_t dst_row1 = vreinterpret_s16_s32(tr3.val[0]);
    int16x4_t dst_row2 = vreinterpret_s16_s32(tr2.val[1]);
    int16x4_t dst_row3 = vreinterpret_s16_s32(tr3.val[1]);

    /* ---------------- Recon & Store (128-bit paired SIMD) ---------------- */
    int16x8_t dst_01 = vcombine_s16(dst_row0, dst_row1);
    int16x8_t dst_23 = vcombine_s16(dst_row2, dst_row3);

    uint8x8_t pred_r0 = vld1_u8(pu1_pred);
    uint8x8_t pred_r1 = vld1_u8(pu1_pred + pred_strd);
    uint8x8_t pred_r2 = vld1_u8(pu1_pred + 2 * pred_strd);
    uint8x8_t pred_r3 = vld1_u8(pu1_pred + 3 * pred_strd);

    uint16x8_t pred_01 = vandq_u16(vreinterpretq_u16_u8(vcombine_u8(pred_r0, pred_r1)), mask_00ff);
    uint16x8_t pred_23 = vandq_u16(vreinterpretq_u16_u8(vcombine_u8(pred_r2, pred_r3)), mask_00ff);

    int16x8_t recon_01 = vqaddq_s16(dst_01, vreinterpretq_s16_u16(pred_01));
    int16x8_t recon_23 = vqaddq_s16(dst_23, vreinterpretq_s16_u16(pred_23));

    int16x8_t clip_min = vdupq_n_s16(0);
    int16x8_t clip_max = vreinterpretq_s16_u16(mask_00ff);

    recon_01 = vminq_s16(vmaxq_s16(recon_01, clip_min), clip_max);
    recon_23 = vminq_s16(vmaxq_s16(recon_23, clip_min), clip_max);

    uint8x8_t dst_r0 = vld1_u8(pu1_dst);
    uint8x8_t dst_r1 = vld1_u8(pu1_dst + dst_strd);
    uint8x8_t dst_r2 = vld1_u8(pu1_dst + 2 * dst_strd);
    uint8x8_t dst_r3 = vld1_u8(pu1_dst + 3 * dst_strd);

    uint8x16_t dst_orig_01 = vcombine_u8(dst_r0, dst_r1);
    uint8x16_t dst_orig_23 = vcombine_u8(dst_r2, dst_r3);

    uint8x16_t out_01 = vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_01), dst_orig_01);
    uint8x16_t out_23 = vbslq_u8(blend_mask, vreinterpretq_u8_s16(recon_23), dst_orig_23);

    vst1_u8(pu1_dst, vget_low_u8(out_01));
    vst1_u8(pu1_dst + dst_strd, vget_high_u8(out_01));
    vst1_u8(pu1_dst + 2 * dst_strd, vget_low_u8(out_23));
    vst1_u8(pu1_dst + 3 * dst_strd, vget_high_u8(out_23));
}
