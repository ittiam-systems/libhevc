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
 *  ihevc_hbd_itrans_recon_neon_intr.c
 *
 * @brief
 *  Contains optimized function definitions for HBD inverse transform and
 *  reconstruction using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_hbd_itrans_recon_4x4_ttype1_neonintr()
 *  - ihevc_hbd_itrans_recon_4x4_neonintr()
 *
 * @remarks
 *  None
 *
 ******************************************************************************
 */
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_itrans_recon.h"
#include "ihevc_function_selector.h"
#include "ihevc_trans_macros.h"
#include "arm_neon.h"

/**
 * In-register 4x4 matrix transposition of 16-bit elements using NEON primitives
 */
static inline void transpose4x4(int16x4_t t0, int16x4_t t1, int16x4_t t2, int16x4_t t3,
                                int16x4_t *r0, int16x4_t *r1, int16x4_t *r2, int16x4_t *r3)
{
    int16x4x2_t tr0 = vtrn_s16(t0, t1);
    int16x4x2_t tr1 = vtrn_s16(t2, t3);
    int32x2x2_t tr2 = vtrn_s32(vreinterpret_s32_s16(tr0.val[0]), vreinterpret_s32_s16(tr1.val[0]));
    int32x2x2_t tr3 = vtrn_s32(vreinterpret_s32_s16(tr0.val[1]), vreinterpret_s32_s16(tr1.val[1]));
    *r0 = vreinterpret_s16_s32(tr2.val[0]);
    *r1 = vreinterpret_s16_s32(tr3.val[0]);
    *r2 = vreinterpret_s16_s32(tr2.val[1]);
    *r3 = vreinterpret_s16_s32(tr3.val[1]);
}

/**
 * Luma direct reconstruction and storage for 4 pixels
 */
static inline void luma_recon_store_4(int16x4_t recon_row,
                                      const UWORD16 *pu2_pred,
                                      UWORD16 *pu2_dst,
                                      int16x4_t clip_min,
                                      int16x4_t clip_max)
{
    uint16x4_t pred = vld1_u16(pu2_pred);
    int16x4_t a = vqadd_s16(recon_row, vreinterpret_s16_u16(pred));
    int16x4_t c = vmin_s16(vmax_s16(a, clip_min), clip_max);
    vst1_u16(pu2_dst, vreinterpret_u16_s16(c));
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs HBD Inverse transform type 1 (DST) and reconstruction
 *  for 4x4 input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs inverse transform and adds the prediction data and clips output
 *  to the configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 4x4 block
 *
 * @param[out] pu2_dst
 *  Output 4x4 block
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
void ihevc_hbd_itrans_recon_4x4_ttype1_neonintr(WORD16 *pi2_src,
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

    /* Fast zero-block bypass: both zero_cols and zero_rows checked */
    if(((UWORD32)(zero_cols & 0xF) == 0xFU) || ((UWORD32)(zero_rows & 0xF) == 0xFU))
    {
        for(WORD32 i = 0; i < 4; i++)
        {
            vst1_u16(pu2_dst, vld1_u16(pu2_pred));
            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
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

    /*
     * DST-VII 4x4 matrix equations:
     * out0 = 29 * r0 + 74 * r1 + 84 * r2 + 55 * r3
     * out1 = 55 * r0 + 74 * r1 - 29 * r2 - 84 * r3
     * out2 = 74 * r0           - 74 * r2 + 74 * r3
     * out3 = 84 * r0 - 74 * r1 + 55 * r2 - 29 * r3
     *
     * Interleaved 16-bit widening MACs maximize dual-issue pipeline utilization
     * and eliminate slow 32-bit vector multiplies and dependency bubbles.
     */
    int32x4_t out0 = vmull_n_s16(r1, 74);
    int32x4_t out1 = vmull_n_s16(r1, 74);
    int32x4_t out2 = vmull_n_s16(r0, 74);
    int32x4_t out3 = vmull_n_s16(r0, 84);

    out0 = vmlal_n_s16(out0, r0, 29);
    out1 = vmlal_n_s16(out1, r0, 55);
    out2 = vmlsl_n_s16(out2, r2, 74);
    out3 = vmlsl_n_s16(out3, r1, 74);

    out0 = vmlal_n_s16(out0, r2, 84);
    out1 = vmlsl_n_s16(out1, r2, 29);
    out2 = vmlal_n_s16(out2, r3, 74);
    out3 = vmlal_n_s16(out3, r2, 55);

    out0 = vmlal_n_s16(out0, r3, 55);
    out1 = vmlsl_n_s16(out1, r3, 84);
    out3 = vmlsl_n_s16(out3, r3, 29);

    /* Stage 1 shift: 7 with rounding & saturation to int16 */
    int16x4_t t0 = vqrshrn_n_s32(out0, 7);
    int16x4_t t1 = vqrshrn_n_s32(out1, 7);
    int16x4_t t2 = vqrshrn_n_s32(out2, 7);
    int16x4_t t3 = vqrshrn_n_s32(out3, 7);

    /* 4x4 In-Register Transposition */
    int16x4_t row0, row1, row2, row3;
    transpose4x4(t0, t1, t2, t3, &row0, &row1, &row2, &row3);

    /* ---------------- Stage 2 ---------------- */
    out0 = vmull_n_s16(row1, 74);
    out1 = vmull_n_s16(row1, 74);
    out2 = vmull_n_s16(row0, 74);
    out3 = vmull_n_s16(row0, 84);

    out0 = vmlal_n_s16(out0, row0, 29);
    out1 = vmlal_n_s16(out1, row0, 55);
    out2 = vmlsl_n_s16(out2, row2, 74);
    out3 = vmlsl_n_s16(out3, row1, 74);

    out0 = vmlal_n_s16(out0, row2, 84);
    out1 = vmlsl_n_s16(out1, row2, 29);
    out2 = vmlal_n_s16(out2, row3, 74);
    out3 = vmlal_n_s16(out3, row2, 55);

    out0 = vmlal_n_s16(out0, row3, 55);
    out1 = vmlsl_n_s16(out1, row3, 84);
    out3 = vmlsl_n_s16(out3, row3, 29);

    /* Stage 2 shift: -(20 - u1_bit_depth) with rounding & saturation */
    int32x4_t neg_shift = vdupq_n_s32(-(20 - u1_bit_depth));
    int16x4_t s0 = vqmovn_s32(vrshlq_s32(out0, neg_shift));
    int16x4_t s1 = vqmovn_s32(vrshlq_s32(out1, neg_shift));
    int16x4_t s2 = vqmovn_s32(vrshlq_s32(out2, neg_shift));
    int16x4_t s3 = vqmovn_s32(vrshlq_s32(out3, neg_shift));

    /* 4x4 In-Register Transposition */
    int16x4_t dst_row0, dst_row1, dst_row2, dst_row3;
    transpose4x4(s0, s1, s2, s3, &dst_row0, &dst_row1, &dst_row2, &dst_row3);

    /* ---------------- Recon & Store ---------------- */
    int16x4_t clip_min = vdup_n_s16(0);
    int16x4_t clip_max = vdup_n_s16((1 << u1_bit_depth) - 1);

    luma_recon_store_4(dst_row0, pu2_pred + 0 * pred_strd, pu2_dst + 0 * dst_strd, clip_min, clip_max);
    luma_recon_store_4(dst_row1, pu2_pred + 1 * pred_strd, pu2_dst + 1 * dst_strd, clip_min, clip_max);
    luma_recon_store_4(dst_row2, pu2_pred + 2 * pred_strd, pu2_dst + 2 * dst_strd, clip_min, clip_max);
    luma_recon_store_4(dst_row3, pu2_pred + 3 * pred_strd, pu2_dst + 3 * dst_strd, clip_min, clip_max);
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs HBD Inverse transform and reconstruction for 4x4
 *  input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs 4x4 inverse DCT transform, adds the 16-bit prediction data, and
 *  clips output to the configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 4x4 block
 *
 * @param[out] pu2_dst
 *  Output 4x4 block
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
void ihevc_hbd_itrans_recon_4x4_neonintr(WORD16 *pi2_src,
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

    /* Fast zero-block bypass: both zero_cols and zero_rows checked */
    if(((UWORD32)(zero_cols & 0xF) == 0xFU) || ((UWORD32)(zero_rows & 0xF) == 0xFU))
    {
        for(WORD32 i = 0; i < 4; i++)
        {
            vst1_u16(pu2_dst, vld1_u16(pu2_pred));
            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
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
    int16x4_t t0 = vqrshrn_n_s32(out0, 7);
    int16x4_t t1 = vqrshrn_n_s32(out1, 7);
    int16x4_t t2 = vqrshrn_n_s32(out2, 7);
    int16x4_t t3 = vqrshrn_n_s32(out3, 7);

    /* 4x4 In-Register Transposition */
    int16x4_t row0, row1, row2, row3;
    transpose4x4(t0, t1, t2, t3, &row0, &row1, &row2, &row3);

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

    /* Stage 2 shift: -(20 - u1_bit_depth) with rounding & saturation */
    int32x4_t neg_shift = vdupq_n_s32(-(20 - u1_bit_depth));
    int16x4_t s0 = vqmovn_s32(vrshlq_s32(out0, neg_shift));
    int16x4_t s1 = vqmovn_s32(vrshlq_s32(out1, neg_shift));
    int16x4_t s2 = vqmovn_s32(vrshlq_s32(out2, neg_shift));
    int16x4_t s3 = vqmovn_s32(vrshlq_s32(out3, neg_shift));

    /* 4x4 In-Register Transposition */
    int16x4_t dst_row0, dst_row1, dst_row2, dst_row3;
    transpose4x4(s0, s1, s2, s3, &dst_row0, &dst_row1, &dst_row2, &dst_row3);

    /* ---------------- Recon & Store ---------------- */
    int16x4_t clip_min = vdup_n_s16(0);
    int16x4_t clip_max = vdup_n_s16((1 << u1_bit_depth) - 1);

    luma_recon_store_4(dst_row0, pu2_pred + 0 * pred_strd, pu2_dst + 0 * dst_strd, clip_min, clip_max);
    luma_recon_store_4(dst_row1, pu2_pred + 1 * pred_strd, pu2_dst + 1 * dst_strd, clip_min, clip_max);
    luma_recon_store_4(dst_row2, pu2_pred + 2 * pred_strd, pu2_dst + 2 * dst_strd, clip_min, clip_max);
    luma_recon_store_4(dst_row3, pu2_pred + 3 * pred_strd, pu2_dst + 3 * dst_strd, clip_min, clip_max);
}
