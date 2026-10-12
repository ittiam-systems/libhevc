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
    WORD32 tmp_lvl_shift = lvl_shift * wgt0 + (off0 << shift);
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
    int32x4_t tmp_lvl_shift_t_u, tmp_lvl_shift_t_v;
    int32x4x2_t tmp_lvl_shift_t;
    WORD32 tmp_shift = 0 - shift;
    int32x4_t tmp_shift_t;
    int16x4_t tmp_wgt0_u, tmp_wgt0_v;
    int16x4x2_t wgt0;
    uint16x4_t max_pixel_val = vdup_n_u16((1 << bit_depth) - 1);
    uint16x8_t max_pixel_val_q = vdupq_n_u16((1 << bit_depth) - 1);

    WORD32 tmp_lvl_shift = lvl_shift * wgt0_cb + (off0_cb << shift);
    tmp_lvl_shift += (1 << (shift - 1));
    tmp_lvl_shift_t_u = vmovq_n_s32(tmp_lvl_shift);

    tmp_lvl_shift = lvl_shift * wgt0_cr + (off0_cr << shift);
    tmp_lvl_shift += (1 << (shift - 1));
    tmp_lvl_shift_t_v = vmovq_n_s32(tmp_lvl_shift);

    tmp_lvl_shift_t = vzipq_s32(tmp_lvl_shift_t_u, tmp_lvl_shift_t_v);
    tmp_shift_t = vmovq_n_s32(tmp_shift);

    tmp_wgt0_u = vdup_n_s16(wgt0_cb);
    tmp_wgt0_v = vdup_n_s16(wgt0_cr);
    wgt0 = vzip_s16(tmp_wgt0_u, tmp_wgt0_v);

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

            int32x4_t r0_lo = vmull_s16(vget_low_s16(s0), wgt0.val[0]);
            int32x4_t r0_hi = vmull_s16(vget_high_s16(s0), wgt0.val[0]);
            int32x4_t r1_lo = vmull_s16(vget_low_s16(s1), wgt0.val[0]);
            int32x4_t r1_hi = vmull_s16(vget_high_s16(s1), wgt0.val[0]);

            r0_lo = vaddq_s32(r0_lo, tmp_lvl_shift_t.val[0]);
            r0_hi = vaddq_s32(r0_hi, tmp_lvl_shift_t.val[0]);
            r1_lo = vaddq_s32(r1_lo, tmp_lvl_shift_t.val[0]);
            r1_hi = vaddq_s32(r1_hi, tmp_lvl_shift_t.val[0]);

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

            int32x4_t i4_tmp1_t = vmull_s16(pi2_src_val1, wgt0.val[0]);
            i4_tmp1_t = vaddq_s32(i4_tmp1_t, tmp_lvl_shift_t.val[0]);
            int32x4_t sto_res_tmp1 = vshlq_s32(i4_tmp1_t, tmp_shift_t);
            uint16x4_t sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            uint16x4_t sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst, sto_res_tmp3);
            pu2_dst += 4;

            int32x4_t i4_tmp2_t = vmull_s16(pi2_src_val2, wgt0.val[0]);
            i4_tmp2_t = vaddq_s32(i4_tmp2_t, tmp_lvl_shift_t.val[0]);
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
    tmp_lvl_shift += ((off0 + off1 + 1) << (shift - 1));
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
    int32x4_t tmp_lvl_shift_t_u, tmp_lvl_shift_t_v;
    int32x4x2_t tmp_lvl_shift_t;
    WORD32 tmp_shift = 0 - shift;
    int32x4_t tmp_shift_t;
    int16x4_t tmp_wgt0_u, tmp_wgt0_v, tmp_wgt1_u, tmp_wgt1_v;
    int16x4x2_t wgt0, wgt1;
    uint16x4_t max_pixel_val = vdup_n_u16((1 << bit_depth) - 1);
    uint16x8_t max_pixel_val_q = vdupq_n_u16((1 << bit_depth) - 1);

    WORD32 tmp_lvl_shift = (lvl_shift1 * wgt0_cb) + (lvl_shift2 * wgt1_cb);
    tmp_lvl_shift += ((off0_cb + off1_cb + 1) << (shift - 1));
    tmp_lvl_shift_t_u = vmovq_n_s32(tmp_lvl_shift);

    tmp_lvl_shift = (lvl_shift1 * wgt0_cr) + (lvl_shift2 * wgt1_cr);
    tmp_lvl_shift += ((off0_cr + off1_cr + 1) << (shift - 1));
    tmp_lvl_shift_t_v = vmovq_n_s32(tmp_lvl_shift);

    tmp_lvl_shift_t = vzipq_s32(tmp_lvl_shift_t_u, tmp_lvl_shift_t_v);
    tmp_shift_t = vmovq_n_s32(tmp_shift);

    tmp_wgt0_u = vdup_n_s16(wgt0_cb);
    tmp_wgt0_v = vdup_n_s16(wgt0_cr);
    wgt0 = vzip_s16(tmp_wgt0_u, tmp_wgt0_v);
    tmp_wgt1_u = vdup_n_s16(wgt1_cb);
    tmp_wgt1_v = vdup_n_s16(wgt1_cr);
    wgt1 = vzip_s16(tmp_wgt1_u, tmp_wgt1_v);

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

            int32x4_t r0_lo = vmull_s16(vget_low_s16(v1_r0), wgt0.val[0]);
            int32x4_t r0_hi = vmull_s16(vget_high_s16(v1_r0), wgt0.val[0]);
            int32x4_t r1_lo = vmull_s16(vget_low_s16(v1_r1), wgt0.val[0]);
            int32x4_t r1_hi = vmull_s16(vget_high_s16(v1_r1), wgt0.val[0]);

            r0_lo = vmlal_s16(r0_lo, vget_low_s16(v2_r0), wgt1.val[0]);
            r0_hi = vmlal_s16(r0_hi, vget_high_s16(v2_r0), wgt1.val[0]);
            r1_lo = vmlal_s16(r1_lo, vget_low_s16(v2_r1), wgt1.val[0]);
            r1_hi = vmlal_s16(r1_hi, vget_high_s16(v2_r1), wgt1.val[0]);

            r0_lo = vaddq_s32(r0_lo, tmp_lvl_shift_t.val[0]);
            r0_hi = vaddq_s32(r0_hi, tmp_lvl_shift_t.val[0]);
            r1_lo = vaddq_s32(r1_lo, tmp_lvl_shift_t.val[0]);
            r1_hi = vaddq_s32(r1_hi, tmp_lvl_shift_t.val[0]);

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

            int32x4_t i4_tmp1_t1 = vmull_s16(pi2_src1_val1, wgt0.val[0]);
            i4_tmp1_t1 = vmlal_s16(i4_tmp1_t1, pi2_src2_val1, wgt1.val[0]);
            i4_tmp1_t1 = vaddq_s32(i4_tmp1_t1, tmp_lvl_shift_t.val[0]);
            int32x4_t sto_res_tmp1 = vshlq_s32(i4_tmp1_t1, tmp_shift_t);
            uint16x4_t sto_res_tmp2 = vqmovun_s32(sto_res_tmp1);
            uint16x4_t sto_res_tmp3 = vmin_u16(sto_res_tmp2, max_pixel_val);
            vst1_u16(pu2_dst, sto_res_tmp3);
            pu2_dst += 4;

            int32x4_t i4_tmp2_t1 = vmull_s16(pi2_src1_val2, wgt0.val[0]);
            i4_tmp2_t1 = vmlal_s16(i4_tmp2_t1, pi2_src2_val2, wgt1.val[0]);
            i4_tmp2_t1 = vaddq_s32(i4_tmp2_t1, tmp_lvl_shift_t.val[0]);
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
    WORD32 shift = 14 - bit_depth + 1;
    WORD32 tmp_shift = 0 - shift;
    WORD32 tmp_lvl_shift = lvl_shift1 + lvl_shift2 + (1 << (shift - 1));
    int16x8_t tmp_lvl_shift_q = vdupq_n_s16((int16_t)tmp_lvl_shift);
    int16x8_t tmp_shift_q = vdupq_n_s16((int16_t)tmp_shift);
    int16x8_t zero_q = vdupq_n_s16(0);
    int16x4_t tmp_lvl_shift_t = vdup_n_s16((int16_t)tmp_lvl_shift);
    int16x4_t tmp_shift_t = vdup_n_s16((int16_t)tmp_shift);
    int16x4_t zero = vdup_n_s16(0);

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

            int16x8_t r0 = vqaddq_s16(v1_r0, v2_r0);
            int16x8_t r1 = vqaddq_s16(v1_r1, v2_r1);

            r0 = vqaddq_s16(r0, tmp_lvl_shift_q);
            r1 = vqaddq_s16(r1, tmp_lvl_shift_q);

            r0 = vshlq_s16(r0, tmp_shift_q);
            r1 = vshlq_s16(r1, tmp_shift_q);

            r0 = vmaxq_s16(r0, zero_q);
            r1 = vmaxq_s16(r1, zero_q);

            vst1q_u16(pu2_dst, vreinterpretq_u16_s16(r0));
            vst1q_u16(pu2_dst_tmp, vreinterpretq_u16_s16(r1));
            pu2_dst += 8;
            pu2_dst_tmp += 8;
        }
        if(col > 0)
        {
            int16x4_t v1_r0 = vld1_s16((int16_t *)pi2_src1);
            pi2_src1 += 4;
            int16x4_t v2_r0 = vld1_s16((int16_t *)pi2_src2);
            pi2_src2 += 4;
            int16x4_t v1_r1 = vld1_s16((int16_t *)pi2_src_tmp1);
            int16x4_t v2_r1 = vld1_s16((int16_t *)pi2_src_tmp2);

            int16x4_t r0 = vqadd_s16(v1_r0, v2_r0);
            int16x4_t r1 = vqadd_s16(v1_r1, v2_r1);

            r0 = vqadd_s16(r0, tmp_lvl_shift_t);
            r1 = vqadd_s16(r1, tmp_lvl_shift_t);

            r0 = vshl_s16(r0, tmp_shift_t);
            r1 = vshl_s16(r1, tmp_shift_t);

            r0 = vmax_s16(r0, zero);
            r1 = vmax_s16(r1, zero);

            vst1_u16(pu2_dst, vreinterpret_u16_s16(r0));
            vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(r1));
            pu2_dst += 4;
        }
        pi2_src1 += 2 * src_strd1 - wd;
        pi2_src2 += 2 * src_strd2 - wd;
        pu2_dst += 2 * dst_strd - wd;
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
    WORD32 shift = 14 - bit_depth + 1;
    WORD32 tmp_shift = 0 - shift;
    WORD32 tmp_lvl_shift = lvl_shift1 + lvl_shift2 + (1 << (shift - 1));
    int16x8_t tmp_lvl_shift_q = vdupq_n_s16((int16_t)tmp_lvl_shift);
    int16x8_t tmp_shift_q = vdupq_n_s16((int16_t)tmp_shift);
    int16x8_t zero_q = vdupq_n_s16(0);
    int16x4_t tmp_lvl_shift_t = vdup_n_s16((int16_t)tmp_lvl_shift);
    int16x4_t tmp_shift_t = vdup_n_s16((int16_t)tmp_shift);
    int16x4_t zero = vdup_n_s16(0);

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

            int16x8_t r0 = vqaddq_s16(v1_r0, v2_r0);
            int16x8_t r1 = vqaddq_s16(v1_r1, v2_r1);

            r0 = vqaddq_s16(r0, tmp_lvl_shift_q);
            r1 = vqaddq_s16(r1, tmp_lvl_shift_q);

            r0 = vshlq_s16(r0, tmp_shift_q);
            r1 = vshlq_s16(r1, tmp_shift_q);

            r0 = vmaxq_s16(r0, zero_q);
            r1 = vmaxq_s16(r1, zero_q);

            vst1q_u16(pu2_dst, vreinterpretq_u16_s16(r0));
            vst1q_u16(pu2_dst_tmp, vreinterpretq_u16_s16(r1));
            pu2_dst += 8;
            pu2_dst_tmp += 8;
        }
        if(col > 0)
        {
            int16x4_t v1_r0 = vld1_s16((int16_t *)pi2_src1);
            pi2_src1 += 4;
            int16x4_t v2_r0 = vld1_s16((int16_t *)pi2_src2);
            pi2_src2 += 4;
            int16x4_t v1_r1 = vld1_s16((int16_t *)pi2_src_tmp1);
            int16x4_t v2_r1 = vld1_s16((int16_t *)pi2_src_tmp2);

            int16x4_t r0 = vqadd_s16(v1_r0, v2_r0);
            int16x4_t r1 = vqadd_s16(v1_r1, v2_r1);

            r0 = vqadd_s16(r0, tmp_lvl_shift_t);
            r1 = vqadd_s16(r1, tmp_lvl_shift_t);

            r0 = vshl_s16(r0, tmp_shift_t);
            r1 = vshl_s16(r1, tmp_shift_t);

            r0 = vmax_s16(r0, zero);
            r1 = vmax_s16(r1, zero);

            vst1_u16(pu2_dst, vreinterpret_u16_s16(r0));
            vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(r1));
            pu2_dst += 4;
        }
        pi2_src1 += 2 * src_strd1 - 2 * wd;
        pi2_src2 += 2 * src_strd2 - 2 * wd;
        pu2_dst += 2 * dst_strd - 2 * wd;
    }
}
