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
*  ihevc_hbd_weighted_pred_neon_intr.c
*
* @brief
*  Contains function definitions for weighted prediction used in inter
* prediction
*
*
* @par List of Functions:
*   - ihevc_hbd_weighted_pred_uni_neonintr()
*   - ihevc_hbd_weighted_pred_chroma_uni_neonintr()
*   - ihevc_hbd_weighted_pred_bi_neonintr()
*   - ihevc_hbd_weighted_pred_chroma_bi_neonintr()
*   - ihevc_hbd_weighted_pred_bi_default_neonintr()
*   - ihevc_hbd_weighted_pred_chroma_bi_default_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/
/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_inter_pred.h"
#include <arm_neon.h>

/**
*******************************************************************************
*
* @brief
*  Does uni-weighted prediction on the array pointed by  pi2_src and stores
* it at the location pointed by pu2_dst
*
* @par Description:
*  dst = ( (src + lvl_shift) * wgt0 + (1 << (shift - 1)) )  >> shift +
* offset
*
* @param[in] pi2_src
*  Pointer to the source
*
* @param[out] pu2_dst
*  Pointer to the destination
*
* @param[in] src_strd
*  Source stride
*
* @param[in] dst_strd
*  Destination stride
*
* @param[in] wgt0
*  weight to be multiplied to the source
*
* @param[in] off0
*  offset to be added after rounding and
*
* @param[in] shift
*  (14 Bit depth) + log2_weight_denominator
*
* @param[in] lvl_shift
*  added before shift and offset
*
* @param[in] ht
*  height of the source
*
* @param[in] wd
*  width of the source
*
* @param[in] bit_depth
*  bit depth of the pixel
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_weighted_pred_uni_neonintr(WORD16 *pi2_src,
                                          UWORD16 *pu2_dst,
                                          WORD32 src_strd,
                                          WORD32 dst_strd,
                                          WORD32 wgt0,
                                          WORD32 off0,
                                          WORD32 shift,
                                          WORD32 lvl_shift,
                                          WORD32 ht,
                                          WORD32 wd,
                                          UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 tmp_shift = 0 - shift;
    WORD32 tmp_lvl_shift = lvl_shift * wgt0 + (off0 * (1 << shift));
    tmp_lvl_shift += (1 << (shift - 1));
    int32x4_t tmp_lvl_shift_t = vmovq_n_s32(tmp_lvl_shift);
    int32x4_t tmp_shift_t = vmovq_n_s32(tmp_shift);
    uint16x4_t max_pixel_val = vdup_n_u16((1 << bit_depth) - 1);
    uint16x8_t max_pixel_val_q = vdupq_n_u16((1 << bit_depth) - 1);

    for(row = ht; row > 0; row -= 2)
    {
        WORD16 *pi2_src_tmp = pi2_src + src_strd;
        UWORD16 *pu2_dst_tmp = pu2_dst + dst_strd;

        for(col = wd; col >= 8; col -= 8)
        {
            int16x8_t s0 = vld1q_s16((int16_t *)pi2_src);
            int16x8_t s1 = vld1q_s16((int16_t *)pi2_src_tmp);
            pi2_src += 8;
            pi2_src_tmp += 8;

            int32x4_t r0_lo = vmull_n_s16(vget_low_s16(s0), (int16_t)wgt0);
            int32x4_t r0_hi = vmull_n_s16(vget_high_s16(s0), (int16_t)wgt0);
            int32x4_t r1_lo = vmull_n_s16(vget_low_s16(s1), (int16_t)wgt0);
            int32x4_t r1_hi = vmull_n_s16(vget_high_s16(s1), (int16_t)wgt0);

            r0_lo = vaddq_s32(r0_lo, tmp_lvl_shift_t);
            r0_hi = vaddq_s32(r0_hi, tmp_lvl_shift_t);
            r1_lo = vaddq_s32(r1_lo, tmp_lvl_shift_t);
            r1_hi = vaddq_s32(r1_hi, tmp_lvl_shift_t);

            r0_lo = vshlq_s32(r0_lo, tmp_shift_t);
            r0_hi = vshlq_s32(r0_hi, tmp_shift_t);
            r1_lo = vshlq_s32(r1_lo, tmp_shift_t);
            r1_hi = vshlq_s32(r1_hi, tmp_shift_t);

            uint16x8_t d0 = vcombine_u16(vqmovun_s32(r0_lo), vqmovun_s32(r0_hi));
            uint16x8_t d1 = vcombine_u16(vqmovun_s32(r1_lo), vqmovun_s32(r1_hi));

            d0 = vminq_u16(d0, max_pixel_val_q);
            d1 = vminq_u16(d1, max_pixel_val_q);

            vst1q_u16(pu2_dst, d0);
            vst1q_u16(pu2_dst_tmp, d1);
            pu2_dst += 8;
            pu2_dst_tmp += 8;
        }
        if(col > 0)
        {
            int16x4_t pi2_src_val1 = vld1_s16((int16_t *)pi2_src);
            pi2_src += 4;
            int16x4_t pi2_src_val2 = vld1_s16((int16_t *)pi2_src_tmp);

            int32x4_t i4_tmp1_t = vmull_n_s16(pi2_src_val1, (int16_t)wgt0);
            i4_tmp1_t = vaddq_s32(i4_tmp1_t, tmp_lvl_shift_t);
            int32x4_t sto_res_tmp1 = vshlq_s32(i4_tmp1_t, tmp_shift_t);
            uint16x4_t sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            uint16x4_t sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst, sto_res_tmp3);
            pu2_dst += 4;

            int32x4_t i4_tmp2_t = vmull_n_s16(pi2_src_val2, (int16_t)wgt0);
            i4_tmp2_t = vaddq_s32(i4_tmp2_t, tmp_lvl_shift_t);
            sto_res_tmp1 = vshlq_s32(i4_tmp2_t, tmp_shift_t);
            sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst_tmp, sto_res_tmp3);
        }
        pi2_src += 2 * src_strd - wd;
        pu2_dst += 2 * dst_strd - wd;
    }
}

/**
*******************************************************************************
*
* @brief
* Does chroma uni-weighted prediction on array pointed by pi2_src and stores
* it at the location pointed by pu2_dst
*
* @par Description:
*  dst = ( (src + lvl_shift) * wgt0 + (1 << (shift - 1)) )  >> shift +
* offset
*
* @param[in] pi2_src
*  Pointer to the source
*
* @param[out] pu2_dst
*  Pointer to the destination
*
* @param[in] src_strd
*  Source stride
*
* @param[in] dst_strd
*  Destination stride
*
* @param[in] wgt0_cb
*  weight to be multiplied to the source Cb
*
* @param[in] wgt0_cr
*  weight to be multiplied to the source Cr
*
* @param[in] off0_cb
*  offset to be added after rounding and shifting for Cb
*
* @param[in] off0_cr
*  offset to be added after rounding and shifting for Cr
*
* @param[in] shift
*  (14 Bit depth) + log2_weight_denominator
*
* @param[in] lvl_shift
*  added before shift and offset
*
* @param[in] ht
*  height of the source
*
* @param[in] wd
*  width of the source (each colour component)
*
* @param[in] bit_depth
*  bit depth of the pixel
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_weighted_pred_chroma_uni_neonintr(WORD16 *pi2_src,
                                                 UWORD16 *pu2_dst,
                                                 WORD32 src_strd,
                                                 WORD32 dst_strd,
                                                 WORD32 wgt0_cb,
                                                 WORD32 wgt0_cr,
                                                 WORD32 off0_cb,
                                                 WORD32 off0_cr,
                                                 WORD32 shift,
                                                 WORD32 lvl_shift,
                                                 WORD32 ht,
                                                 WORD32 wd,
                                                 UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 tmp_shift = 0 - shift;
    int32x4_t tmp_shift_t = vmovq_n_s32(tmp_shift);
    uint16x4_t max_pixel_val = vdup_n_u16((1 << bit_depth) - 1);
    uint16x8_t max_pixel_val_q = vdupq_n_u16((1 << bit_depth) - 1);

    WORD32 tmp_lvl_shift_cb = lvl_shift * wgt0_cb + (off0_cb * (1 << shift)) + (1 << (shift - 1));
    WORD32 tmp_lvl_shift_cr = lvl_shift * wgt0_cr + (off0_cr * (1 << shift)) + (1 << (shift - 1));
    uint64_t u8_lvl_shift_uv = ((uint32_t)tmp_lvl_shift_cb) | ((uint64_t)(uint32_t)tmp_lvl_shift_cr << 32);
    int32x4_t tmp_lvl_shift_t = vreinterpretq_s32_u64(vdupq_n_u64(u8_lvl_shift_uv));

    uint32_t u4_wgt0_uv = ((uint16_t)wgt0_cb) | ((uint32_t)(uint16_t)wgt0_cr << 16);
    int16x4_t wgt0 = vreinterpret_s16_u32(vdup_n_u32(u4_wgt0_uv));

    for(row = ht; row > 0; row -= 2)
    {
        WORD16 *pi2_src_tmp = pi2_src + src_strd;
        UWORD16 *pu2_dst_tmp = pu2_dst + dst_strd;

        for(col = 2 * wd; col >= 8; col -= 8)
        {
            int16x8_t s0 = vld1q_s16((int16_t *)pi2_src);
            int16x8_t s1 = vld1q_s16((int16_t *)pi2_src_tmp);
            pi2_src += 8;
            pi2_src_tmp += 8;

            int32x4_t r0_lo = vmull_s16(vget_low_s16(s0), wgt0);
            int32x4_t r0_hi = vmull_s16(vget_high_s16(s0), wgt0);
            int32x4_t r1_lo = vmull_s16(vget_low_s16(s1), wgt0);
            int32x4_t r1_hi = vmull_s16(vget_high_s16(s1), wgt0);

            r0_lo = vaddq_s32(r0_lo, tmp_lvl_shift_t);
            r0_hi = vaddq_s32(r0_hi, tmp_lvl_shift_t);
            r1_lo = vaddq_s32(r1_lo, tmp_lvl_shift_t);
            r1_hi = vaddq_s32(r1_hi, tmp_lvl_shift_t);

            r0_lo = vshlq_s32(r0_lo, tmp_shift_t);
            r0_hi = vshlq_s32(r0_hi, tmp_shift_t);
            r1_lo = vshlq_s32(r1_lo, tmp_shift_t);
            r1_hi = vshlq_s32(r1_hi, tmp_shift_t);

            uint16x8_t d0 = vcombine_u16(vqmovun_s32(r0_lo), vqmovun_s32(r0_hi));
            uint16x8_t d1 = vcombine_u16(vqmovun_s32(r1_lo), vqmovun_s32(r1_hi));

            d0 = vminq_u16(d0, max_pixel_val_q);
            d1 = vminq_u16(d1, max_pixel_val_q);

            vst1q_u16(pu2_dst, d0);
            vst1q_u16(pu2_dst_tmp, d1);
            pu2_dst += 8;
            pu2_dst_tmp += 8;
        }
        if(col > 0)
        {
            int16x4_t pi2_src_val1 = vld1_s16((int16_t *)pi2_src);
            pi2_src += 4;
            int16x4_t pi2_src_val2 = vld1_s16((int16_t *)pi2_src_tmp);

            int32x4_t i4_tmp1_t = vmull_s16(pi2_src_val1, wgt0);
            i4_tmp1_t = vaddq_s32(i4_tmp1_t, tmp_lvl_shift_t);
            int32x4_t sto_res_tmp1 = vshlq_s32(i4_tmp1_t, tmp_shift_t);
            uint16x4_t sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            uint16x4_t sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst, sto_res_tmp3);
            pu2_dst += 4;

            int32x4_t i4_tmp2_t = vmull_s16(pi2_src_val2, wgt0);
            i4_tmp2_t = vaddq_s32(i4_tmp2_t, tmp_lvl_shift_t);
            sto_res_tmp1 = vshlq_s32(i4_tmp2_t, tmp_shift_t);
            sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst_tmp, sto_res_tmp3);
        }
        pi2_src += 2 * src_strd - 2 * wd;
        pu2_dst += 2 * dst_strd - 2 * wd;
    }
}

/**
*******************************************************************************
*
* @brief
*  Does bi-weighted prediction on the arrays pointed by  pi2_src1 and
* pi2_src2 and stores it at location pointed  by pu2_dst
*
* @par Description:
*  dst = ( (src1 + lvl_shift1)*wgt0 +  (src2 + lvl_shift2)*wgt1 +  (off0 +
* off1 + 1) << (shift - 1) ) >> shift
*
* @param[in] pi2_src1
*  Pointer to source 1
*
* @param[in] pi2_src2
*  Pointer to source 2
*
* @param[out] pu2_dst
*  Pointer to destination
*
* @param[in] src_strd1
*  Source stride 1
*
* @param[in] src_strd2
*  Source stride 2
*
* @param[in] dst_strd
*  Destination stride
*
* @param[in] wgt0
*  weight to be multiplied to source 1
*
* @param[in] off0
*  offset 0
*
* @param[in] wgt1
*  weight to be multiplied to source 2
*
* @param[in] off1
*  offset 1
*
* @param[in] shift
*  (14 Bit depth) + log2_weight_denominator
*
* @param[in] lvl_shift1
*  added before shift and offset
*
* @param[in] lvl_shift2
*  added before shift and offset
*
* @param[in] ht
*  height of the source
*
* @param[in] wd
*  width of the source
*
* @param[in] bit_depth
*  bit depth of the pixel
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_weighted_pred_bi_neonintr(WORD16 *pi2_src1,
                                         WORD16 *pi2_src2,
                                         UWORD16 *pu2_dst,
                                         WORD32 src_strd1,
                                         WORD32 src_strd2,
                                         WORD32 dst_strd,
                                         WORD32 wgt0,
                                         WORD32 off0,
                                         WORD32 wgt1,
                                         WORD32 off1,
                                         WORD32 shift,
                                         WORD32 lvl_shift1,
                                         WORD32 lvl_shift2,
                                         WORD32 ht,
                                         WORD32 wd,
                                         UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 tmp_shift = 0 - shift;
    WORD32 tmp_lvl_shift = (lvl_shift1 * wgt0) + (lvl_shift2 * wgt1);
    tmp_lvl_shift += ((off0 + off1 + 1) * (1 << (shift - 1)));
    int32x4_t tmp_lvl_shift_t = vmovq_n_s32(tmp_lvl_shift);
    int32x4_t tmp_shift_t = vmovq_n_s32(tmp_shift);
    uint16x4_t max_pixel_val = vdup_n_u16((1 << bit_depth) - 1);
    uint16x8_t max_pixel_val_q = vdupq_n_u16((1 << bit_depth) - 1);

    for(row = ht; row > 0; row -= 2)
    {
        WORD16 *pi2_src_tmp1 = pi2_src1 + src_strd1;
        WORD16 *pi2_src_tmp2 = pi2_src2 + src_strd2;
        UWORD16 *pu2_dst_tmp = pu2_dst + dst_strd;

        for(col = wd; col >= 8; col -= 8)
        {
            int16x8_t v1_r0 = vld1q_s16((int16_t *)pi2_src1);
            int16x8_t v2_r0 = vld1q_s16((int16_t *)pi2_src2);
            int16x8_t v1_r1 = vld1q_s16((int16_t *)pi2_src_tmp1);
            int16x8_t v2_r1 = vld1q_s16((int16_t *)pi2_src_tmp2);
            pi2_src1 += 8;
            pi2_src2 += 8;
            pi2_src_tmp1 += 8;
            pi2_src_tmp2 += 8;

            int32x4_t r0_lo = vmull_n_s16(vget_low_s16(v1_r0), (int16_t)wgt0);
            int32x4_t r0_hi = vmull_n_s16(vget_high_s16(v1_r0), (int16_t)wgt0);
            int32x4_t r1_lo = vmull_n_s16(vget_low_s16(v1_r1), (int16_t)wgt0);
            int32x4_t r1_hi = vmull_n_s16(vget_high_s16(v1_r1), (int16_t)wgt0);

            r0_lo = vmlal_n_s16(r0_lo, vget_low_s16(v2_r0), (int16_t)wgt1);
            r0_hi = vmlal_n_s16(r0_hi, vget_high_s16(v2_r0), (int16_t)wgt1);
            r1_lo = vmlal_n_s16(r1_lo, vget_low_s16(v2_r1), (int16_t)wgt1);
            r1_hi = vmlal_n_s16(r1_hi, vget_high_s16(v2_r1), (int16_t)wgt1);

            r0_lo = vaddq_s32(r0_lo, tmp_lvl_shift_t);
            r0_hi = vaddq_s32(r0_hi, tmp_lvl_shift_t);
            r1_lo = vaddq_s32(r1_lo, tmp_lvl_shift_t);
            r1_hi = vaddq_s32(r1_hi, tmp_lvl_shift_t);

            r0_lo = vshlq_s32(r0_lo, tmp_shift_t);
            r0_hi = vshlq_s32(r0_hi, tmp_shift_t);
            r1_lo = vshlq_s32(r1_lo, tmp_shift_t);
            r1_hi = vshlq_s32(r1_hi, tmp_shift_t);

            uint16x8_t out0 = vcombine_u16(vqmovun_s32(r0_lo), vqmovun_s32(r0_hi));
            uint16x8_t out1 = vcombine_u16(vqmovun_s32(r1_lo), vqmovun_s32(r1_hi));

            out0 = vminq_u16(out0, max_pixel_val_q);
            out1 = vminq_u16(out1, max_pixel_val_q);

            vst1q_u16(pu2_dst, out0);
            vst1q_u16(pu2_dst_tmp, out1);
            pu2_dst += 8;
            pu2_dst_tmp += 8;
        }
        if(col > 0)
        {
            int16x4_t pi2_src1_val1 = vld1_s16((int16_t *)pi2_src1);
            pi2_src1 += 4;
            int16x4_t pi2_src2_val1 = vld1_s16((int16_t *)pi2_src2);
            pi2_src2 += 4;
            int16x4_t pi2_src1_val2 = vld1_s16((int16_t *)pi2_src_tmp1);
            int16x4_t pi2_src2_val2 = vld1_s16((int16_t *)pi2_src_tmp2);

            int32x4_t i4_tmp1_t1 = vmull_n_s16(pi2_src1_val1, (int16_t)wgt0);
            i4_tmp1_t1 = vmlal_n_s16(i4_tmp1_t1, pi2_src2_val1, (int16_t)wgt1);
            i4_tmp1_t1 = vaddq_s32(i4_tmp1_t1, tmp_lvl_shift_t);
            int32x4_t sto_res_tmp1 = vshlq_s32(i4_tmp1_t1, tmp_shift_t);
            uint16x4_t sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            uint16x4_t sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst, sto_res_tmp3);
            pu2_dst += 4;

            int32x4_t i4_tmp2_t1 = vmull_n_s16(pi2_src1_val2, (int16_t)wgt0);
            i4_tmp2_t1 = vmlal_n_s16(i4_tmp2_t1, pi2_src2_val2, (int16_t)wgt1);
            i4_tmp2_t1 = vaddq_s32(i4_tmp2_t1, tmp_lvl_shift_t);
            sto_res_tmp1 = vshlq_s32(i4_tmp2_t1, tmp_shift_t);
            sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst_tmp, sto_res_tmp3);
        }
        pi2_src1 += 2 * src_strd1 - wd;
        pi2_src2 += 2 * src_strd2 - wd;
        pu2_dst  += 2 * dst_strd  - wd;
    }
}

/**
*******************************************************************************
*
* @brief
*  Does chroma bi-weighted prediction on the arrays pointed by pi2_src1 and
* pi2_src2 and stores it at location pointed  by pu2_dst
*
* @par Description:
*  dst = ( (src1 + lvl_shift1)*wgt0 +  (src2 + lvl_shift2)*wgt1 +  (off0 +
* off1 + 1) << (shift - 1) ) >> shift
*
* @param[in] pi2_src1
*  Pointer to source 1
*
* @param[in] pi2_src2
*  Pointer to source 2
*
* @param[out] pu2_dst
*  Pointer to destination
*
* @param[in] src_strd1
*  Source stride 1
*
* @param[in] src_strd2
*  Source stride 2
*
* @param[in] dst_strd
*  Destination stride
*
* @param[in] wgt0_cb
*  weight to be multiplied to source 1 Cb
*
* @param[in] wgt0_cr
*  weight to be multiplied to source 1 Cr
*
* @param[in] off0_cb
*  offset 0 Cb
*
* @param[in] off0_cr
*  offset 0 Cr
*
* @param[in] wgt1_cb
*  weight to be multiplied to source 2 Cb
*
* @param[in] wgt1_cr
*  weight to be multiplied to source 2 Cr
*
* @param[in] off1_cb
*  offset 1 Cb
*
* @param[in] off1_cr
*  offset 1 Cr
*
* @param[in] shift
*  (14 Bit depth) + log2_weight_denominator
*
* @param[in] lvl_shift1
*  added before shift and offset
*
* @param[in] lvl_shift2
*  added before shift and offset
*
* @param[in] ht
*  height of the source
*
* @param[in] wd
*  width of the source (each colour component)
*
* @param[in] bit_depth
*  bit depth of the pixel
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_weighted_pred_chroma_bi_neonintr(WORD16 *pi2_src1,
                                                WORD16 *pi2_src2,
                                                UWORD16 *pu2_dst,
                                                WORD32 src_strd1,
                                                WORD32 src_strd2,
                                                WORD32 dst_strd,
                                                WORD32 wgt0_cb,
                                                WORD32 wgt0_cr,
                                                WORD32 off0_cb,
                                                WORD32 off0_cr,
                                                WORD32 wgt1_cb,
                                                WORD32 wgt1_cr,
                                                WORD32 off1_cb,
                                                WORD32 off1_cr,
                                                WORD32 shift,
                                                WORD32 lvl_shift1,
                                                WORD32 lvl_shift2,
                                                WORD32 ht,
                                                WORD32 wd,
                                                UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 tmp_shift = 0 - shift;
    int32x4_t tmp_shift_t = vmovq_n_s32(tmp_shift);
    uint16x4_t max_pixel_val = vdup_n_u16((1 << bit_depth) - 1);
    uint16x8_t max_pixel_val_q = vdupq_n_u16((1 << bit_depth) - 1);

    WORD32 tmp_lvl_shift_cb = (lvl_shift1 * wgt0_cb) + (lvl_shift2 * wgt1_cb) + ((off0_cb + off1_cb + 1) * (1 << (shift - 1)));
    WORD32 tmp_lvl_shift_cr = (lvl_shift1 * wgt0_cr) + (lvl_shift2 * wgt1_cr) + ((off0_cr + off1_cr + 1) * (1 << (shift - 1)));
    uint64_t u8_lvl_shift_uv = ((uint32_t)tmp_lvl_shift_cb) | ((uint64_t)(uint32_t)tmp_lvl_shift_cr << 32);
    int32x4_t tmp_lvl_shift_t = vreinterpretq_s32_u64(vdupq_n_u64(u8_lvl_shift_uv));

    uint32_t u4_wgt0_uv = ((uint16_t)wgt0_cb) | ((uint32_t)(uint16_t)wgt0_cr << 16);
    int16x4_t wgt0 = vreinterpret_s16_u32(vdup_n_u32(u4_wgt0_uv));
    uint32_t u4_wgt1_uv = ((uint16_t)wgt1_cb) | ((uint32_t)(uint16_t)wgt1_cr << 16);
    int16x4_t wgt1 = vreinterpret_s16_u32(vdup_n_u32(u4_wgt1_uv));

    for(row = ht; row > 0; row -= 2)
    {
        WORD16 *pi2_src_tmp1 = pi2_src1 + src_strd1;
        WORD16 *pi2_src_tmp2 = pi2_src2 + src_strd2;
        UWORD16 *pu2_dst_tmp = pu2_dst + dst_strd;

        for(col = 2 * wd; col >= 8; col -= 8)
        {
            int16x8_t v1_r0 = vld1q_s16((int16_t *)pi2_src1);
            int16x8_t v2_r0 = vld1q_s16((int16_t *)pi2_src2);
            int16x8_t v1_r1 = vld1q_s16((int16_t *)pi2_src_tmp1);
            int16x8_t v2_r1 = vld1q_s16((int16_t *)pi2_src_tmp2);
            pi2_src1 += 8;
            pi2_src2 += 8;
            pi2_src_tmp1 += 8;
            pi2_src_tmp2 += 8;

            int32x4_t r0_lo = vmull_s16(vget_low_s16(v1_r0), wgt0);
            int32x4_t r0_hi = vmull_s16(vget_high_s16(v1_r0), wgt0);
            int32x4_t r1_lo = vmull_s16(vget_low_s16(v1_r1), wgt0);
            int32x4_t r1_hi = vmull_s16(vget_high_s16(v1_r1), wgt0);

            r0_lo = vmlal_s16(r0_lo, vget_low_s16(v2_r0), wgt1);
            r0_hi = vmlal_s16(r0_hi, vget_high_s16(v2_r0), wgt1);
            r1_lo = vmlal_s16(r1_lo, vget_low_s16(v2_r1), wgt1);
            r1_hi = vmlal_s16(r1_hi, vget_high_s16(v2_r1), wgt1);

            r0_lo = vaddq_s32(r0_lo, tmp_lvl_shift_t);
            r0_hi = vaddq_s32(r0_hi, tmp_lvl_shift_t);
            r1_lo = vaddq_s32(r1_lo, tmp_lvl_shift_t);
            r1_hi = vaddq_s32(r1_hi, tmp_lvl_shift_t);

            r0_lo = vshlq_s32(r0_lo, tmp_shift_t);
            r0_hi = vshlq_s32(r0_hi, tmp_shift_t);
            r1_lo = vshlq_s32(r1_lo, tmp_shift_t);
            r1_hi = vshlq_s32(r1_hi, tmp_shift_t);

            uint16x8_t out0 = vcombine_u16(vqmovun_s32(r0_lo), vqmovun_s32(r0_hi));
            uint16x8_t out1 = vcombine_u16(vqmovun_s32(r1_lo), vqmovun_s32(r1_hi));

            out0 = vminq_u16(out0, max_pixel_val_q);
            out1 = vminq_u16(out1, max_pixel_val_q);

            vst1q_u16(pu2_dst, out0);
            vst1q_u16(pu2_dst_tmp, out1);
            pu2_dst += 8;
            pu2_dst_tmp += 8;
        }
        if(col > 0)
        {
            int16x4_t pi2_src1_val1 = vld1_s16((int16_t *)pi2_src1);
            pi2_src1 += 4;
            int16x4_t pi2_src2_val1 = vld1_s16((int16_t *)pi2_src2);
            pi2_src2 += 4;
            int16x4_t pi2_src1_val2 = vld1_s16((int16_t *)pi2_src_tmp1);
            int16x4_t pi2_src2_val2 = vld1_s16((int16_t *)pi2_src_tmp2);

            int32x4_t i4_tmp1_t1 = vmull_s16(pi2_src1_val1, wgt0);
            i4_tmp1_t1 = vmlal_s16(i4_tmp1_t1, pi2_src2_val1, wgt1);
            i4_tmp1_t1 = vaddq_s32(i4_tmp1_t1, tmp_lvl_shift_t);
            int32x4_t sto_res_tmp1 = vshlq_s32(i4_tmp1_t1, tmp_shift_t);
            uint16x4_t sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            uint16x4_t sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst, sto_res_tmp3);
            pu2_dst += 4;

            int32x4_t i4_tmp2_t1 = vmull_s16(pi2_src1_val2, wgt0);
            i4_tmp2_t1 = vmlal_s16(i4_tmp2_t1, pi2_src2_val2, wgt1);
            i4_tmp2_t1 = vaddq_s32(i4_tmp2_t1, tmp_lvl_shift_t);
            sto_res_tmp1 = vshlq_s32(i4_tmp2_t1, tmp_shift_t);
            sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst_tmp, sto_res_tmp3);
        }
        pi2_src1 += 2 * src_strd1 - 2 * wd;
        pi2_src2 += 2 * src_strd2 - 2 * wd;
        pu2_dst  += 2 * dst_strd  - 2 * wd;
    }
}

/**
*******************************************************************************
*
* @brief
*  Does default bi-weighted prediction on the arrays pointed by pi2_src1 and
* pi2_src2 and stores it at location  pointed by pu2_dst
*
* @par Description:
*  dst = ( (src1 + lvl_shift1) +  (src2 + lvl_shift2) +  1 << (shift - 1) )
* >> shift  where shift = 15 - BitDepth
*
* @param[in] pi2_src1
*  Pointer to source 1
*
* @param[in] pi2_src2
*  Pointer to source 2
*
* @param[out] pu2_dst
*  Pointer to destination
*
* @param[in] src_strd1
*  Source stride 1
*
* @param[in] src_strd2
*  Source stride 2
*
* @param[in] dst_strd
*  Destination stride
*
* @param[in] lvl_shift1
*  added before shift and offset
*
* @param[in] lvl_shift2
*  added before shift and offset
*
* @param[in] ht
*  height of the source
*
* @param[in] wd
*  width of the source
*
* @param[in] bit_depth
*  bit depth of the pixel
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_weighted_pred_bi_default_neonintr(WORD16 *pi2_src1,
                                                 WORD16 *pi2_src2,
                                                 UWORD16 *pu2_dst,
                                                 WORD32 src_strd1,
                                                 WORD32 src_strd2,
                                                 WORD32 dst_strd,
                                                 WORD32 lvl_shift1,
                                                 WORD32 lvl_shift2,
                                                 WORD32 ht,
                                                 WORD32 wd,
                                                 UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 shift;
    WORD16 *tmp_src1, *tmp_src2;
    UWORD16 *tmp_dst;
    int16x8_t src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
    int16x8_t src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;
    int16x8_t tmp_lvl_shift_q, tmp_shift_q, zero_q;

    shift = 14 - bit_depth + 1;
    tmp_lvl_shift_q = vdupq_n_s16((int16_t)(lvl_shift1 + lvl_shift2 + (1 << (shift - 1))));
    tmp_shift_q = vdupq_n_s16((int16_t)(0 - shift));
    zero_q = vdupq_n_s16(0);

    for(row = ht; row >= 4; row -= 4)
    {
        for(col = wd; col >= 32; col -= 32)
        {
            int16x8_t src_temp9_8x16b, src_temp10_8x16b, src_temp11_8x16b, src_temp12_8x16b;
            int16x8_t src_temp13_8x16b, src_temp14_8x16b, src_temp15_8x16b, src_temp16_8x16b;

            tmp_src1 = pi2_src1 + (wd - col);
            tmp_src2 = pi2_src2 + (wd - col);
            tmp_dst = pu2_dst + (wd - col);

            /* Rows 0 and 1 (32 columns = 4 vectors per row) */
            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp9_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp10_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp13_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp14_8x16b = vld1q_s16(tmp_src2 + 24);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp11_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp12_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp15_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp16_8x16b = vld1q_s16(tmp_src2 + 24);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, src_temp10_8x16b);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, src_temp14_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, src_temp12_8x16b);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, src_temp16_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, tmp_lvl_shift_q);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, tmp_lvl_shift_q);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp9_8x16b = vshlq_s16(src_temp9_8x16b, tmp_shift_q);
            src_temp13_8x16b = vshlq_s16(src_temp13_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);
            src_temp11_8x16b = vshlq_s16(src_temp11_8x16b, tmp_shift_q);
            src_temp15_8x16b = vshlq_s16(src_temp15_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp9_8x16b = vmaxq_s16(src_temp9_8x16b, zero_q);
            src_temp13_8x16b = vmaxq_s16(src_temp13_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);
            src_temp11_8x16b = vmaxq_s16(src_temp11_8x16b, zero_q);
            src_temp15_8x16b = vmaxq_s16(src_temp15_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp9_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp13_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp11_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp15_8x16b));

            /* Rows 2 and 3 (32 columns = 4 vectors per row) */
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;
            tmp_dst += dst_strd;

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp9_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp10_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp13_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp14_8x16b = vld1q_s16(tmp_src2 + 24);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp11_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp12_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp15_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp16_8x16b = vld1q_s16(tmp_src2 + 24);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, src_temp10_8x16b);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, src_temp14_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, src_temp12_8x16b);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, src_temp16_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, tmp_lvl_shift_q);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, tmp_lvl_shift_q);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp9_8x16b = vshlq_s16(src_temp9_8x16b, tmp_shift_q);
            src_temp13_8x16b = vshlq_s16(src_temp13_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);
            src_temp11_8x16b = vshlq_s16(src_temp11_8x16b, tmp_shift_q);
            src_temp15_8x16b = vshlq_s16(src_temp15_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp9_8x16b = vmaxq_s16(src_temp9_8x16b, zero_q);
            src_temp13_8x16b = vmaxq_s16(src_temp13_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);
            src_temp11_8x16b = vmaxq_s16(src_temp11_8x16b, zero_q);
            src_temp15_8x16b = vmaxq_s16(src_temp15_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp9_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp13_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp11_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp15_8x16b));
        }

        if(col >= 16)
        {
            tmp_src1 = pi2_src1 + (wd - col);
            tmp_src2 = pi2_src2 + (wd - col);
            tmp_dst = pu2_dst + (wd - col);

            /* Rows 0 and 1 */
            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));

            /* Rows 2 and 3 */
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;
            tmp_dst += dst_strd;

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));

            col -= 16;
        }

        if(col >= 8)
        {
            tmp_src1 = pi2_src1 + (wd - col);
            tmp_src2 = pi2_src2 + (wd - col);
            tmp_dst = pu2_dst + (wd - col);

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp5_8x16b = vld1q_s16(tmp_src1);
            src_temp6_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp7_8x16b = vld1q_s16(tmp_src1);
            src_temp8_8x16b = vld1q_s16(tmp_src2);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp7_8x16b));

            col -= 8;
        }

        if(col > 0)
        {
            int16x4_t s1_r0, s2_r0, s1_r1, s2_r1, s1_r2, s2_r2, s1_r3, s2_r3;

            tmp_src1 = pi2_src1 + (wd - col);
            tmp_src2 = pi2_src2 + (wd - col);
            tmp_dst = pu2_dst + (wd - col);

            s1_r0 = vld1_s16(tmp_src1);
            s2_r0 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r1 = vld1_s16(tmp_src1);
            s2_r1 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r2 = vld1_s16(tmp_src1);
            s2_r2 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r3 = vld1_s16(tmp_src1);
            s2_r3 = vld1_s16(tmp_src2);

            src_temp1_8x16b = vcombine_s16(s1_r0, s1_r1);
            src_temp2_8x16b = vcombine_s16(s2_r0, s2_r1);
            src_temp3_8x16b = vcombine_s16(s1_r2, s1_r3);
            src_temp4_8x16b = vcombine_s16(s2_r2, s2_r3);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);

            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_low_s16(src_temp1_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_high_s16(src_temp1_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_low_s16(src_temp3_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_high_s16(src_temp3_8x16b)));
        }

        pi2_src1 += 4 * src_strd1;
        pi2_src2 += 4 * src_strd2;
        pu2_dst += 4 * dst_strd;
    }

    if(row > 0)
    {
        for(col = wd; col >= 16; col -= 16)
        {
            tmp_src1 = pi2_src1 + (wd - col);
            tmp_src2 = pi2_src2 + (wd - col);
            tmp_dst = pu2_dst + (wd - col);

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));
        }

        if(col >= 8)
        {
            tmp_src1 = pi2_src1 + (wd - col);
            tmp_src2 = pi2_src2 + (wd - col);
            tmp_dst = pu2_dst + (wd - col);

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));

            col -= 8;
        }

        if(col > 0)
        {
            int16x4_t s1_r0, s2_r0, s1_r1, s2_r1;

            tmp_src1 = pi2_src1 + (wd - col);
            tmp_src2 = pi2_src2 + (wd - col);
            tmp_dst = pu2_dst + (wd - col);

            s1_r0 = vld1_s16(tmp_src1);
            s2_r0 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r1 = vld1_s16(tmp_src1);
            s2_r1 = vld1_s16(tmp_src2);

            src_temp1_8x16b = vcombine_s16(s1_r0, s1_r1);
            src_temp2_8x16b = vcombine_s16(s2_r0, s2_r1);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);

            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_low_s16(src_temp1_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_high_s16(src_temp1_8x16b)));
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Does chroma default bi-weighted prediction on arrays pointed by pi2_src1 and
* pi2_src2 and stores it at location  pointed by pu2_dst
*
* @par Description:
*  dst = ( (src1 + lvl_shift1) +  (src2 + lvl_shift2) +  1 << (shift - 1) )
* >> shift  where shift = 15 - BitDepth
*
* @param[in] pi2_src1
*  Pointer to source 1
*
* @param[in] pi2_src2
*  Pointer to source 2
*
* @param[out] pu2_dst
*  Pointer to destination
*
* @param[in] src_strd1
*  Source stride 1
*
* @param[in] src_strd2
*  Source stride 2
*
* @param[in] dst_strd
*  Destination stride
*
* @param[in] lvl_shift1
*  added before shift and offset
*
* @param[in] lvl_shift2
*  added before shift and offset
*
* @param[in] ht
*  height of the source
*
* @param[in] wd
*  width of the source (each colour component)
*
* @param[in] bit_depth
*  bit depth of the pixel
*
* @returns
*  None
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_weighted_pred_chroma_bi_default_neonintr(WORD16 *pi2_src1,
                                                        WORD16 *pi2_src2,
                                                        UWORD16 *pu2_dst,
                                                        WORD32 src_strd1,
                                                        WORD32 src_strd2,
                                                        WORD32 dst_strd,
                                                        WORD32 lvl_shift1,
                                                        WORD32 lvl_shift2,
                                                        WORD32 ht,
                                                        WORD32 wd,
                                                        UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD32 shift;
    WORD16 *tmp_src1, *tmp_src2;
    UWORD16 *tmp_dst;
    int16x8_t src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
    int16x8_t src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;
    int16x8_t tmp_lvl_shift_q, tmp_shift_q, zero_q;

    shift = 14 - bit_depth + 1;
    tmp_lvl_shift_q = vdupq_n_s16((int16_t)(lvl_shift1 + lvl_shift2 + (1 << (shift - 1))));
    tmp_shift_q = vdupq_n_s16((int16_t)(0 - shift));
    zero_q = vdupq_n_s16(0);

    for(row = ht; row >= 4; row -= 4)
    {
        for(col = 2 * wd; col >= 32; col -= 32)
        {
            int16x8_t src_temp9_8x16b, src_temp10_8x16b, src_temp11_8x16b, src_temp12_8x16b;
            int16x8_t src_temp13_8x16b, src_temp14_8x16b, src_temp15_8x16b, src_temp16_8x16b;

            tmp_src1 = pi2_src1 + (2 * wd - col);
            tmp_src2 = pi2_src2 + (2 * wd - col);
            tmp_dst = pu2_dst + (2 * wd - col);

            /* Rows 0 and 1 (32 columns = 4 vectors per row) */
            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp9_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp10_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp13_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp14_8x16b = vld1q_s16(tmp_src2 + 24);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp11_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp12_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp15_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp16_8x16b = vld1q_s16(tmp_src2 + 24);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, src_temp10_8x16b);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, src_temp14_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, src_temp12_8x16b);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, src_temp16_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, tmp_lvl_shift_q);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, tmp_lvl_shift_q);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp9_8x16b = vshlq_s16(src_temp9_8x16b, tmp_shift_q);
            src_temp13_8x16b = vshlq_s16(src_temp13_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);
            src_temp11_8x16b = vshlq_s16(src_temp11_8x16b, tmp_shift_q);
            src_temp15_8x16b = vshlq_s16(src_temp15_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp9_8x16b = vmaxq_s16(src_temp9_8x16b, zero_q);
            src_temp13_8x16b = vmaxq_s16(src_temp13_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);
            src_temp11_8x16b = vmaxq_s16(src_temp11_8x16b, zero_q);
            src_temp15_8x16b = vmaxq_s16(src_temp15_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp9_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp13_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp11_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp15_8x16b));

            /* Rows 2 and 3 (32 columns = 4 vectors per row) */
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;
            tmp_dst += dst_strd;

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp9_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp10_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp13_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp14_8x16b = vld1q_s16(tmp_src2 + 24);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);
            src_temp11_8x16b = vld1q_s16(tmp_src1 + 16);
            src_temp12_8x16b = vld1q_s16(tmp_src2 + 16);
            src_temp15_8x16b = vld1q_s16(tmp_src1 + 24);
            src_temp16_8x16b = vld1q_s16(tmp_src2 + 24);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, src_temp10_8x16b);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, src_temp14_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, src_temp12_8x16b);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, src_temp16_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp9_8x16b = vqaddq_s16(src_temp9_8x16b, tmp_lvl_shift_q);
            src_temp13_8x16b = vqaddq_s16(src_temp13_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);
            src_temp11_8x16b = vqaddq_s16(src_temp11_8x16b, tmp_lvl_shift_q);
            src_temp15_8x16b = vqaddq_s16(src_temp15_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp9_8x16b = vshlq_s16(src_temp9_8x16b, tmp_shift_q);
            src_temp13_8x16b = vshlq_s16(src_temp13_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);
            src_temp11_8x16b = vshlq_s16(src_temp11_8x16b, tmp_shift_q);
            src_temp15_8x16b = vshlq_s16(src_temp15_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp9_8x16b = vmaxq_s16(src_temp9_8x16b, zero_q);
            src_temp13_8x16b = vmaxq_s16(src_temp13_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);
            src_temp11_8x16b = vmaxq_s16(src_temp11_8x16b, zero_q);
            src_temp15_8x16b = vmaxq_s16(src_temp15_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp9_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp13_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));
            vst1q_u16(tmp_dst + 16, vreinterpretq_u16_s16(src_temp11_8x16b));
            vst1q_u16(tmp_dst + 24, vreinterpretq_u16_s16(src_temp15_8x16b));
        }

        if(col >= 16)
        {
            tmp_src1 = pi2_src1 + (2 * wd - col);
            tmp_src2 = pi2_src2 + (2 * wd - col);
            tmp_dst = pu2_dst + (2 * wd - col);

            /* Rows 0 and 1 */
            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));

            /* Rows 2 and 3 */
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;
            tmp_dst += dst_strd;

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));

            col -= 16;
        }

        if(col >= 8)
        {
            tmp_src1 = pi2_src1 + (2 * wd - col);
            tmp_src2 = pi2_src2 + (2 * wd - col);
            tmp_dst = pu2_dst + (2 * wd - col);

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp5_8x16b = vld1q_s16(tmp_src1);
            src_temp6_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp7_8x16b = vld1q_s16(tmp_src1);
            src_temp8_8x16b = vld1q_s16(tmp_src2);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp7_8x16b));

            col -= 8;
        }

        if(col > 0)
        {
            int16x4_t s1_r0, s2_r0, s1_r1, s2_r1, s1_r2, s2_r2, s1_r3, s2_r3;

            tmp_src1 = pi2_src1 + (2 * wd - col);
            tmp_src2 = pi2_src2 + (2 * wd - col);
            tmp_dst = pu2_dst + (2 * wd - col);

            s1_r0 = vld1_s16(tmp_src1);
            s2_r0 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r1 = vld1_s16(tmp_src1);
            s2_r1 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r2 = vld1_s16(tmp_src1);
            s2_r2 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r3 = vld1_s16(tmp_src1);
            s2_r3 = vld1_s16(tmp_src2);

            src_temp1_8x16b = vcombine_s16(s1_r0, s1_r1);
            src_temp2_8x16b = vcombine_s16(s2_r0, s2_r1);
            src_temp3_8x16b = vcombine_s16(s1_r2, s1_r3);
            src_temp4_8x16b = vcombine_s16(s2_r2, s2_r3);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);

            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_low_s16(src_temp1_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_high_s16(src_temp1_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_low_s16(src_temp3_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_high_s16(src_temp3_8x16b)));
        }

        pi2_src1 += 4 * src_strd1;
        pi2_src2 += 4 * src_strd2;
        pu2_dst += 4 * dst_strd;
    }

    if(row > 0)
    {
        for(col = 2 * wd; col >= 16; col -= 16)
        {
            tmp_src1 = pi2_src1 + (2 * wd - col);
            tmp_src2 = pi2_src2 + (2 * wd - col);
            tmp_dst = pu2_dst + (2 * wd - col);

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            src_temp5_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp6_8x16b = vld1q_s16(tmp_src2 + 8);

            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);
            src_temp7_8x16b = vld1q_s16(tmp_src1 + 8);
            src_temp8_8x16b = vld1q_s16(tmp_src2 + 8);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, src_temp6_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, src_temp8_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp5_8x16b = vqaddq_s16(src_temp5_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);
            src_temp7_8x16b = vqaddq_s16(src_temp7_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp5_8x16b = vshlq_s16(src_temp5_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);
            src_temp7_8x16b = vshlq_s16(src_temp7_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp5_8x16b = vmaxq_s16(src_temp5_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);
            src_temp7_8x16b = vmaxq_s16(src_temp7_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp5_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));
            vst1q_u16(tmp_dst + 8, vreinterpretq_u16_s16(src_temp7_8x16b));
        }

        if(col >= 8)
        {
            tmp_src1 = pi2_src1 + (2 * wd - col);
            tmp_src2 = pi2_src2 + (2 * wd - col);
            tmp_dst = pu2_dst + (2 * wd - col);

            src_temp1_8x16b = vld1q_s16(tmp_src1);
            src_temp2_8x16b = vld1q_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            src_temp3_8x16b = vld1q_s16(tmp_src1);
            src_temp4_8x16b = vld1q_s16(tmp_src2);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, src_temp4_8x16b);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp3_8x16b = vqaddq_s16(src_temp3_8x16b, tmp_lvl_shift_q);

            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp3_8x16b = vshlq_s16(src_temp3_8x16b, tmp_shift_q);

            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);
            src_temp3_8x16b = vmaxq_s16(src_temp3_8x16b, zero_q);

            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp1_8x16b));
            tmp_dst += dst_strd;
            vst1q_u16(tmp_dst, vreinterpretq_u16_s16(src_temp3_8x16b));

            col -= 8;
        }

        if(col > 0)
        {
            int16x4_t s1_r0, s2_r0, s1_r1, s2_r1;

            tmp_src1 = pi2_src1 + (2 * wd - col);
            tmp_src2 = pi2_src2 + (2 * wd - col);
            tmp_dst = pu2_dst + (2 * wd - col);

            s1_r0 = vld1_s16(tmp_src1);
            s2_r0 = vld1_s16(tmp_src2);
            tmp_src1 += src_strd1;
            tmp_src2 += src_strd2;

            s1_r1 = vld1_s16(tmp_src1);
            s2_r1 = vld1_s16(tmp_src2);

            src_temp1_8x16b = vcombine_s16(s1_r0, s1_r1);
            src_temp2_8x16b = vcombine_s16(s2_r0, s2_r1);

            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, src_temp2_8x16b);
            src_temp1_8x16b = vqaddq_s16(src_temp1_8x16b, tmp_lvl_shift_q);
            src_temp1_8x16b = vshlq_s16(src_temp1_8x16b, tmp_shift_q);
            src_temp1_8x16b = vmaxq_s16(src_temp1_8x16b, zero_q);

            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_low_s16(src_temp1_8x16b)));
            tmp_dst += dst_strd;
            vst1_u16(tmp_dst, vreinterpret_u16_s16(vget_high_s16(src_temp1_8x16b)));
        }
    }
}
