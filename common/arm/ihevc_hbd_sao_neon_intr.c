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
*  ihevc_hbd_sao_neon_intr.c
*
* @brief
*  Contains 10-bit High Bit Depth NEON intrinsic implementations of HEVC
*  Sample Adaptive Offset (SAO) band offset and edge offset functions.
*
*******************************************************************************
*/

#include <string.h>
#include <arm_neon.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_sao.h"
#include "ihevc_common_tables.h"

/**
*******************************************************************************
*
* @brief
*  SAO band offset filter for 10-bit Luma blocks using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_band_offset_luma_neonintr(UWORD16 *pu2_src,
                                             WORD32 src_strd,
                                             UWORD16 *pu2_src_left,
                                             UWORD16 *pu2_src_top,
                                             UWORD16 *pu2_src_top_left,
                                             WORD32 sao_band_pos,
                                             WORD8 *pi1_sao_offset,
                                             WORD32 wd,
                                             WORD32 ht,
                                             UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    const WORD32 band_shift = u4_bit_depth - 5;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Update left, top-left, and top boundary buffers from unmodified pu2_src */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[row] = pu2_src[row * src_strd + (wd - 1)];
    }
    pu2_src_top_left[0] = pu2_src_top[wd - 1];

    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src[(ht - 1) * src_strd + col]));
    }

    int8x8_t offset_tbl    = vld1_s8(pi1_sao_offset);
    uint8x16_t v_band_pos  = vdupq_n_u8((uint8_t)(sao_band_pos - 1));
    uint8x16_t v_mask_31   = vdupq_n_u8(31);
    int16x8_t v_band_shift = vdupq_n_s16((int16_t)(-band_shift));
    int16x8_t v_zero_s16   = vdupq_n_s16(0);
    uint16x8_t v_max_val   = vdupq_n_u16((uint16_t)max_pixel_val);

    for(row = 0; row < ht; row +=2)
    {
        UWORD16 *pu2_row0 = pu2_src;
        UWORD16 *pu2_row1 = pu2_src + src_strd;

        for(col = 0; col <= wd - 8; col += 8)
        {
            uint16x8_t src0_u16 = vld1q_u16(&pu2_row0[col]);
            uint16x8_t src1_u16 = vld1q_u16(&pu2_row1[col]);

            uint8x8_t idx0_u8   = vmovn_u16(vshlq_u16(src0_u16, v_band_shift));
            uint8x8_t idx1_u8   = vmovn_u16(vshlq_u16(src1_u16, v_band_shift));
            uint8x16_t idx_u8   = vcombine_u8(idx0_u8, idx1_u8);

            uint8x16_t rel_u8   = vandq_u8(vsubq_u8(idx_u8, v_band_pos), v_mask_31);
            int8x8_t off0_s8    = vtbl1_s8(offset_tbl, vreinterpret_s8_u8(vget_low_u8(rel_u8)));
            int8x8_t off1_s8    = vtbl1_s8(offset_tbl, vreinterpret_s8_u8(vget_high_u8(rel_u8)));

            int16x8_t sum0_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src0_u16), off0_s8), v_zero_s16);
            int16x8_t sum1_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src1_u16), off1_s8), v_zero_s16);

            uint16x8_t res0_u16 = vminq_u16(vreinterpretq_u16_s16(sum0_s16), v_max_val);
            uint16x8_t res1_u16 = vminq_u16(vreinterpretq_u16_s16(sum1_s16), v_max_val);

            vst1q_u16(&pu2_row0[col], res0_u16);
            vst1q_u16(&pu2_row1[col], res1_u16);
        }
        pu2_src += 2 *src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*  SAO band offset filter for 10-bit interleaved Chroma (U, V) blocks
*  using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_band_offset_chroma_neonintr(UWORD16 *pu2_src,
                                               WORD32 src_strd,
                                               UWORD16 *pu2_src_left,
                                               UWORD16 *pu2_src_top,
                                               UWORD16 *pu2_src_top_left,
                                               WORD32 sao_band_pos_u,
                                               WORD32 sao_band_pos_v,
                                               WORD8 *pi1_sao_offset_u,
                                               WORD8 *pi1_sao_offset_v,
                                               WORD32 wd,
                                               WORD32 ht,
                                               UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    const WORD32 band_shift = u4_bit_depth - 5;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Update left, top-left, and top boundary buffers from unmodified pu2_src */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[2 * row] = pu2_src[row * src_strd + (wd - 2)];
        pu2_src_left[2 * row + 1] = pu2_src[row * src_strd + (wd - 1)];
    }
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];

    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src[(ht - 1) * src_strd + col]));
    }

    int8x8_t offset_tbl_u = vld1_s8(pi1_sao_offset_u);
    int8x8_t offset_tbl_v = vld1_s8(pi1_sao_offset_v);

    uint16_t u2_band_pos_uv  = (uint16_t)((((sao_band_pos_v - 1) & 0xFF) << 8) | ((sao_band_pos_u - 1) & 0xFF));
    uint8x16_t v_band_pos_uv = vreinterpretq_u8_u16(vdupq_n_u16(u2_band_pos_uv));
    uint8x16_t v_mask_31     = vdupq_n_u8(31);
    uint8x16_t v_odd_lanes   = vreinterpretq_u8_u16(vdupq_n_u16(0xFF00));
    int16x8_t v_band_shift   = vdupq_n_s16((int16_t)(-band_shift));
    int16x8_t v_zero_s16     = vdupq_n_s16(0);
    uint16x8_t v_max_val     = vdupq_n_u16((uint16_t)max_pixel_val);

    for(row = 0; row < ht; row +=2)
    {
        UWORD16 *pu2_row0 = pu2_src;
        UWORD16 *pu2_row1 = pu2_src + src_strd;

        for(col = 0; col <= wd - 8; col += 8)
        {
            uint16x8_t src0_u16 = vld1q_u16(&pu2_row0[col]);
            uint16x8_t src1_u16 = vld1q_u16(&pu2_row1[col]);

            uint8x8_t idx0_u8   = vmovn_u16(vshlq_u16(src0_u16, v_band_shift));
            uint8x8_t idx1_u8   = vmovn_u16(vshlq_u16(src1_u16, v_band_shift));
            uint8x16_t idx_u8   = vcombine_u8(idx0_u8, idx1_u8);

            uint8x16_t rel_uv   = vandq_u8(vsubq_u8(idx_u8, v_band_pos_uv), v_mask_31);

            int8x16_t off_u_s8  = vcombine_s8(vtbl1_s8(offset_tbl_u, vreinterpret_s8_u8(vget_low_u8(rel_uv))),
                                              vtbl1_s8(offset_tbl_u, vreinterpret_s8_u8(vget_high_u8(rel_uv))));
            int8x16_t off_v_s8  = vcombine_s8(vtbl1_s8(offset_tbl_v, vreinterpret_s8_u8(vget_low_u8(rel_uv))),
                                              vtbl1_s8(offset_tbl_v, vreinterpret_s8_u8(vget_high_u8(rel_uv))));
            int8x16_t off_uv_s8 = vbslq_s8(v_odd_lanes, off_v_s8, off_u_s8);

            int16x8_t sum0_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src0_u16), vget_low_s8(off_uv_s8)), v_zero_s16);
            int16x8_t sum1_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src1_u16), vget_high_s8(off_uv_s8)), v_zero_s16);

            uint16x8_t res0_u16 = vminq_u16(vreinterpretq_u16_s16(sum0_s16), v_max_val);
            uint16x8_t res1_u16 = vminq_u16(vreinterpretq_u16_s16(sum1_s16), v_max_val);

            vst1q_u16(&pu2_row0[col], res0_u16);
            vst1q_u16(&pu2_row1[col], res1_u16);
        }
        pu2_src += 2 *src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 0 (Horizontal 0-degree) filter for 10-bit Luma blocks
*  using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class0_neonintr(UWORD16 *pu2_src,
                                               WORD32 src_strd,
                                               UWORD16 *pu2_src_left,
                                               UWORD16 *pu2_src_top,
                                               UWORD16 *pu2_src_top_left,
                                               UWORD16 *pu2_src_top_right,
                                               UWORD16 *pu2_src_bot_left,
                                               UWORD8 *pu1_avail,
                                               WORD8 *pi1_sao_offset,
                                               WORD32 wd,
                                               WORD32 ht,
                                               UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD16 *pu2_src_cpy, *pu2_src_left_cpy;

    uint16x8_t src_temp0_lo_16x8b, src_temp0_hi_16x8b;
    uint16x8_t src_temp1_lo_16x8b, src_temp1_hi_16x8b;
    uint16x8_t left0_lo_16x8b, left0_hi_16x8b;
    uint16x8_t left1_lo_16x8b, left1_hi_16x8b;
    uint16x8_t right0_lo_16x8b, right0_hi_16x8b;
    uint16x8_t right1_lo_16x8b, right1_hi_16x8b;
    int8x16_t sign_left0_16x8b, sign_left1_16x8b;
    int8x16_t sign_right0_16x8b, sign_right1_16x8b;
    int8x16_t edge_idx0_16x8b, edge_idx1_16x8b;
    int8x8_t edge_idx_8x8b, sao_offset_8x8b, offset_edge_tbl_8x8b;
    int8x16_t const2_16x8b;
    int16x8_t m_max, m_min;

    UNUSED(pu2_src_top_right);
    UNUSED(pu2_src_bot_left);

    m_max = vdupq_n_s16((1 << u4_bit_depth) - 1);
    m_min = vdupq_n_s16(0);

    /* Update top-left array */
    *pu2_src_top_left = pu2_src_top[wd - 1];

    edge_idx_8x8b        = vld1_s8(gi1_table_edge_idx);
    sao_offset_8x8b      = vld1_s8(pi1_sao_offset);
    offset_edge_tbl_8x8b = vtbl1_s8(sao_offset_8x8b, edge_idx_8x8b);

    int8_t i1_avail_left  = pu1_avail[0] ? 2 : 10;
    int8_t i1_avail_right = pu1_avail[1] ? 2 : 10;
    const2_16x8b          = vdupq_n_s8(2);

    for(col = 0; col <= wd - 16; col += 16)
    {
        int8x16_t const2_cur = const2_16x8b;

        src_temp0_lo_16x8b = vld1q_u16(pu2_src + (ht - 1) * src_strd);
        src_temp0_hi_16x8b = vld1q_u16(pu2_src + 8 + (ht - 1) * src_strd);
        vst1q_u16(pu2_src_top, src_temp0_lo_16x8b);
        vst1q_u16(pu2_src_top + 8, src_temp0_hi_16x8b);

        pu2_src_cpy = pu2_src;
        pu2_src_left_cpy = pu2_src_left;
        if(0 == col)
        {
            const2_cur = vsetq_lane_s8(i1_avail_left, const2_cur, 0);
        }
        if(col == wd - 16)
        {
            const2_cur = vsetq_lane_s8(i1_avail_right, const2_cur, 15);
        }
        for(row = 0; row < ht; row += 2)
        {
            int8x8_t off0_lo, off0_hi, off1_lo, off1_hi;

            //row = 0 load 16 pixel values
            src_temp0_lo_16x8b = vld1q_u16(pu2_src_cpy);
            src_temp0_hi_16x8b = vld1q_u16(pu2_src_cpy + 8);
            // row = 1
            src_temp1_lo_16x8b = vld1q_u16(pu2_src_cpy + src_strd);
            src_temp1_hi_16x8b = vld1q_u16(pu2_src_cpy + src_strd + 8);

            left0_lo_16x8b = vld1q_dup_u16(pu2_src_left_cpy);
            left1_lo_16x8b = vld1q_dup_u16(pu2_src_left_cpy + 1);
            left0_hi_16x8b = vdupq_lane_u16(vget_low_u16(src_temp0_lo_16x8b), 0);
            left1_hi_16x8b = vdupq_lane_u16(vget_low_u16(src_temp1_lo_16x8b), 0);

            //row = 0 right
            right0_lo_16x8b = vld1q_u16(pu2_src_cpy + 1);
            right0_hi_16x8b = vld1q_u16(pu2_src_cpy + 9);
            // row = 1 right
            right1_lo_16x8b = vld1q_u16(pu2_src_cpy + src_strd + 1);
            right1_hi_16x8b = vld1q_u16(pu2_src_cpy + src_strd + 9);

            vst1q_lane_u16(pu2_src_left_cpy, src_temp0_hi_16x8b, 7);
            vst1q_lane_u16(pu2_src_left_cpy + 1, src_temp1_hi_16x8b, 7);

            //sign_right using native unsigned 16-bit comparisons
            sign_right0_16x8b = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(src_temp0_lo_16x8b, right0_lo_16x8b),
                                    vcgtq_u16(src_temp0_lo_16x8b, right0_lo_16x8b))),
                vmovn_u16(vsubq_u16(vcltq_u16(src_temp0_hi_16x8b, right0_hi_16x8b),
                                    vcgtq_u16(src_temp0_hi_16x8b, right0_hi_16x8b)))));
            sign_right1_16x8b = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(src_temp1_lo_16x8b, right1_lo_16x8b),
                                    vcgtq_u16(src_temp1_lo_16x8b, right1_lo_16x8b))),
                vmovn_u16(vsubq_u16(vcltq_u16(src_temp1_hi_16x8b, right1_hi_16x8b),
                                    vcgtq_u16(src_temp1_hi_16x8b, right1_hi_16x8b)))));

            // Derive sign_left from left_edge and negated sign_right[0..14]
            sign_left0_16x8b = vreinterpretq_s8_u8(vsubq_u8(
                vreinterpretq_u8_u16(vcltq_u16(left0_hi_16x8b, left0_lo_16x8b)),
                vreinterpretq_u8_u16(vcgtq_u16(left0_hi_16x8b, left0_lo_16x8b))));
            sign_left1_16x8b = vreinterpretq_s8_u8(vsubq_u8(
                vreinterpretq_u8_u16(vcltq_u16(left1_hi_16x8b, left1_lo_16x8b)),
                vreinterpretq_u8_u16(vcgtq_u16(left1_hi_16x8b, left1_lo_16x8b))));
            sign_left0_16x8b = vextq_s8(sign_left0_16x8b, vnegq_s8(sign_right0_16x8b), 15);
            sign_left1_16x8b = vextq_s8(sign_left1_16x8b, vnegq_s8(sign_right1_16x8b), 15);

            //combining sign-left and sign_right + constant 2
            edge_idx0_16x8b = vaddq_s8(vaddq_s8(sign_right0_16x8b, const2_cur), sign_left0_16x8b);
            edge_idx1_16x8b = vaddq_s8(vaddq_s8(sign_right1_16x8b, const2_cur), sign_left1_16x8b);

            //table lookup (indices >= 8 automatically yield 0 in vtbl1_s8)
            off0_lo = vtbl1_s8(offset_edge_tbl_8x8b, vget_low_s8(edge_idx0_16x8b));
            off0_hi = vtbl1_s8(offset_edge_tbl_8x8b, vget_high_s8(edge_idx0_16x8b));
            off1_lo = vtbl1_s8(offset_edge_tbl_8x8b, vget_low_s8(edge_idx1_16x8b));
            off1_hi = vtbl1_s8(offset_edge_tbl_8x8b, vget_high_s8(edge_idx1_16x8b));

            //convert to 16 bit then add and clip
            src_temp0_lo_16x8b = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src_temp0_lo_16x8b), off0_lo), m_min), m_max));
            src_temp0_hi_16x8b = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src_temp0_hi_16x8b), off0_hi), m_min), m_max));

            src_temp1_lo_16x8b = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src_temp1_lo_16x8b), off1_lo), m_min), m_max));
            src_temp1_hi_16x8b = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(src_temp1_hi_16x8b), off1_hi), m_min), m_max));

            //row = 0 store 16 pixel values
            vst1q_u16(pu2_src_cpy, src_temp0_lo_16x8b);
            vst1q_u16(pu2_src_cpy + 8, src_temp0_hi_16x8b);
            // row = 1
            vst1q_u16(pu2_src_cpy + src_strd, src_temp1_lo_16x8b);
            vst1q_u16(pu2_src_cpy + src_strd + 8, src_temp1_hi_16x8b);

            pu2_src_cpy += (src_strd << 1);
            pu2_src_left_cpy += 2;
        }
        pu2_src += 16;
        pu2_src_top += 16;
    }

    if(col < wd)
    {
        int8x8_t const2_rem = vdup_n_s8(2);

        src_temp0_lo_16x8b = vld1q_u16(pu2_src + (ht - 1) * src_strd);
        vst1q_u16(pu2_src_top, src_temp0_lo_16x8b);

        if(0 == col)
        {
            const2_rem = vset_lane_s8(i1_avail_left, const2_rem, 0);
        }
        const2_rem = vset_lane_s8(i1_avail_right, const2_rem, 7);
        pu2_src_cpy = pu2_src;
        pu2_src_left_cpy = pu2_src_left;
        for(row = 0; row < ht; row += 2)
        {
            int8x8_t sign_l0, sign_l1, sign_r0, sign_r1, idx0, idx1, off0, off1;

            src_temp0_lo_16x8b = vld1q_u16(pu2_src_cpy);
            right0_lo_16x8b    = vld1q_u16(pu2_src_cpy + 1);
            src_temp1_lo_16x8b = vld1q_u16(pu2_src_cpy + src_strd);
            right1_lo_16x8b    = vld1q_u16(pu2_src_cpy + src_strd + 1);

            left0_lo_16x8b = vld1q_dup_u16(pu2_src_left_cpy);
            left1_lo_16x8b = vld1q_dup_u16(pu2_src_left_cpy + 1);
            left0_lo_16x8b = vextq_u16(left0_lo_16x8b, src_temp0_lo_16x8b, 7);
            left1_lo_16x8b = vextq_u16(left1_lo_16x8b, src_temp1_lo_16x8b, 7);

            vst1q_lane_u16(pu2_src_left_cpy,     src_temp0_lo_16x8b, 7);
            vst1q_lane_u16(pu2_src_left_cpy + 1, src_temp1_lo_16x8b, 7);

            sign_l0 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(
                vcltq_u16(src_temp0_lo_16x8b, left0_lo_16x8b),
                vcgtq_u16(src_temp0_lo_16x8b, left0_lo_16x8b))));
            sign_l1 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(
                vcltq_u16(src_temp1_lo_16x8b, left1_lo_16x8b),
                vcgtq_u16(src_temp1_lo_16x8b, left1_lo_16x8b))));

            sign_r0 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(
                vcltq_u16(src_temp0_lo_16x8b, right0_lo_16x8b),
                vcgtq_u16(src_temp0_lo_16x8b, right0_lo_16x8b))));
            sign_r1 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(
                vcltq_u16(src_temp1_lo_16x8b, right1_lo_16x8b),
                vcgtq_u16(src_temp1_lo_16x8b, right1_lo_16x8b))));

            idx0 = vadd_s8(vadd_s8(sign_r0, sign_l0), const2_rem);
            idx1 = vadd_s8(vadd_s8(sign_r1, sign_l1), const2_rem);

            off0 = vtbl1_s8(offset_edge_tbl_8x8b, idx0);
            off1 = vtbl1_s8(offset_edge_tbl_8x8b, idx1);

            src_temp0_lo_16x8b = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(
                vaddw_s8(vreinterpretq_s16_u16(src_temp0_lo_16x8b), off0), m_min), m_max));
            src_temp1_lo_16x8b = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(
                vaddw_s8(vreinterpretq_s16_u16(src_temp1_lo_16x8b), off1), m_min), m_max));

            vst1q_u16(pu2_src_cpy,            src_temp0_lo_16x8b);
            vst1q_u16(pu2_src_cpy + src_strd, src_temp1_lo_16x8b);

            pu2_src_cpy += (src_strd << 1);
            pu2_src_left_cpy += 2;
        }
        pu2_src += wd;
    }
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 1 (Vertical 90-degree) filter for 10-bit Luma blocks
*  using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class1_neonintr(UWORD16 *pu2_src,
                                               WORD32 src_strd,
                                               UWORD16 *pu2_src_left,
                                               UWORD16 *pu2_src_top,
                                               UWORD16 *pu2_src_top_left,
                                               UWORD16 *pu2_src_top_right,
                                               UWORD16 *pu2_src_bot_left,
                                               UWORD8 *pu1_avail,
                                               WORD8 *pi1_sao_offset,
                                               WORD32 wd,
                                               WORD32 ht,
                                               UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    WORD32 ht_tmp = ht;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    int8x8_t offset_edge_tbl = vtbl1_s8(vld1_s8(pi1_sao_offset), vld1_s8(gi1_table_edge_idx));

    int8x16_t const_2_q  = vdupq_n_s8(2);
    int8x8_t const_2     = vdup_n_s8(2);
    int16x8_t v_zero_s16 = vdupq_n_s16(0);
    uint16x8_t v_max_val = vdupq_n_u16((uint16_t)max_pixel_val);

    *pu2_src_top_left = pu2_src_top[wd - 1];

    /* 1. Update left boundary buffer from unmodified pu2_src */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[row] = pu2_src[row * src_strd + wd - 1];
    }

    /* 2. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht_tmp--;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    for(col = 0; col <= wd - 16; col += 16)
    {
        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_lo    = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_hi    = vld1q_u16(&pu2_src_top_cpy[col + 8]);

        vst1q_u16(&pu2_src_top[col],     vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
        vst1q_u16(&pu2_src_top[col + 8], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col + 8]));

        uint16x8_t cur0_lo = vld1q_u16(pu2_src_cpy);
        uint16x8_t cur0_hi = vld1q_u16(pu2_src_cpy + 8);
        int8x16_t sign_up0 = vreinterpretq_s8_u8(vcombine_u8(
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, top_lo), vcgtq_u16(cur0_lo, top_lo))),
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, top_hi), vcgtq_u16(cur0_hi, top_hi)))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0  = pu2_src_cpy;
            UWORD16 *pu2_row1  = pu2_src_cpy + src_strd;
            uint16x8_t cur1_lo = vld1q_u16(pu2_row1);
            uint16x8_t cur1_hi = vld1q_u16(pu2_row1 + 8);
            uint16x8_t next_lo = vld1q_u16(pu2_row1 + src_strd);
            uint16x8_t next_hi = vld1q_u16(pu2_row1 + src_strd + 8);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, cur1_lo), vcgtq_u16(cur0_lo, cur1_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, cur1_hi), vcgtq_u16(cur0_hi, cur1_hi)))));
            int8x16_t sign_down1 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_lo, next_lo), vcgtq_u16(cur1_lo, next_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_hi, next_hi), vcgtq_u16(cur1_hi, next_hi)))));

            int8x16_t sign_up1 = vnegq_s8(sign_down0);
            int8x16_t idx0     = vaddq_s8(vaddq_s8(sign_down0, const_2_q), sign_up0);
            int8x16_t idx1     = vaddq_s8(vaddq_s8(sign_down1, const_2_q), sign_up1);
            sign_up0           = vnegq_s8(sign_down1);

            int8x8_t off0_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx0));
            int8x8_t off1_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx1));
            int8x8_t off1_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx1));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);
            int16x8_t sum1_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_lo), off1_lo), v_zero_s16);
            int16x8_t sum1_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_hi), off1_hi), v_zero_s16);

            vst1q_u16(pu2_row0,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_row0 + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));
            vst1q_u16(pu2_row1,     vminq_u16(vreinterpretq_u16_s16(sum1_lo), v_max_val));
            vst1q_u16(pu2_row1 + 8, vminq_u16(vreinterpretq_u16_s16(sum1_hi), v_max_val));

            cur0_lo = next_lo;
            cur0_hi = next_hi;
            pu2_src_cpy += 2 * src_strd;
        }
        if(row < ht_tmp)
        {
            uint16x8_t next_lo = vld1q_u16(pu2_src_cpy + src_strd);
            uint16x8_t next_hi = vld1q_u16(pu2_src_cpy + src_strd + 8);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, next_lo), vcgtq_u16(cur0_lo, next_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, next_hi), vcgtq_u16(cur0_hi, next_hi)))));

            int8x16_t idx0 = vaddq_s8(vaddq_s8(sign_down0, const_2_q), sign_up0);
            int8x8_t off0_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_src_cpy,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_src_cpy + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));
        }
    }

    if(col < wd)
    {
        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col]);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row0 = vld1q_u16(pu2_src_cpy);
        int8x8_t sign_up0   = vreinterpret_s8_u8(vmovn_u16(
            vsubq_u16(vcltq_u16(cur_row0, top_row), vcgtq_u16(cur_row0, top_row))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0   = pu2_src_cpy;
            UWORD16 *pu2_row1   = pu2_src_cpy + src_strd;
            uint16x8_t cur_row1 = vld1q_u16(pu2_row1);
            uint16x8_t next_row = vld1q_u16(pu2_row1 + src_strd);

            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row0, cur_row1), vcgtq_u16(cur_row0, cur_row1))));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row1, next_row), vcgtq_u16(cur_row1, next_row))));

            int8x8_t sign_up1 = vneg_s8(sign_down0);
            int8x8_t idx0     = vadd_s8(vadd_s8(sign_down0, const_2), sign_up0);
            int8x8_t idx1     = vadd_s8(vadd_s8(sign_down1, const_2), sign_up1);
            sign_up0          = vneg_s8(sign_down1);

            int8x8_t off0 = vtbl1_s8(offset_edge_tbl, idx0);
            int8x8_t off1 = vtbl1_s8(offset_edge_tbl, idx1);

            int16x8_t sum0 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), off0), v_zero_s16);
            int16x8_t sum1 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row1), off1), v_zero_s16);

            vst1q_u16(pu2_row0, vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val));
            vst1q_u16(pu2_row1, vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val));

            cur_row0 = next_row;
            pu2_src_cpy += 2 * src_strd;
        }
        if(row < ht_tmp)
        {
            uint16x8_t next_row = vld1q_u16(pu2_src_cpy + src_strd);
            int8x8_t sign_down  = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row0, next_row), vcgtq_u16(cur_row0, next_row))));

            int8x8_t raw_idx = vadd_s8(vadd_s8(sign_down, const_2), sign_up0);
            int8x8_t offset  = vtbl1_s8(offset_edge_tbl, raw_idx);

            int16x8_t sum_s16 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), offset), v_zero_s16);
            vst1q_u16(pu2_src_cpy, vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val));
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 2 (135-degree diagonal) filter for 10-bit Luma blocks
*  using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class2_neonintr(UWORD16 *pu2_src,
                                               WORD32 src_strd,
                                               UWORD16 *pu2_src_left,
                                               UWORD16 *pu2_src_top,
                                               UWORD16 *pu2_src_top_left,
                                               UWORD16 *pu2_src_top_right,
                                               UWORD16 *pu2_src_bot_left,
                                               UWORD8 *pu1_avail,
                                               WORD8 *pi1_sao_offset,
                                               WORD32 wd,
                                               WORD32 ht,
                                               UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    WORD32 ht_tmp = ht;
    UWORD16 u2_src_top_left_tmp;
    UWORD16 u2_pos_0_0_tmp;
    UWORD16 u2_pos_wd_ht_tmp;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and corner pixels before any modifications */
    u2_src_top_left_tmp = pu2_src_top[wd - 1];
    UWORD16 u2_top_left = pu2_src_top_left[0];

    if(0 != pu1_avail[4])
    {
        WORD32 edge_idx = 2 + SIGN(pu2_src[0] - u2_top_left) +
                              SIGN(pu2_src[0] - pu2_src[1 + src_strd]);
        edge_idx = gi1_table_edge_idx[edge_idx];
        u2_pos_0_0_tmp = (0 != edge_idx) ? CLIP3(pu2_src[0] + pi1_sao_offset[edge_idx], 0, max_pixel_val)
                                         : pu2_src[0];
    }
    else
    {
        u2_pos_0_0_tmp = pu2_src[0];
    }

    if(0 != pu1_avail[7])
    {
        WORD32 br_idx = wd - 1 + (ht - 1) * src_strd;
        WORD32 edge_idx = 2 + SIGN(pu2_src[br_idx] - pu2_src[br_idx - 1 - src_strd]) +
                              SIGN(pu2_src[br_idx] - pu2_src[br_idx + 1 + src_strd]);
        edge_idx = gi1_table_edge_idx[edge_idx];
        u2_pos_wd_ht_tmp = (0 != edge_idx) ? CLIP3(pu2_src[br_idx] + pi1_sao_offset[edge_idx], 0, max_pixel_val)
                                           : pu2_src[br_idx];
    }
    else
    {
        u2_pos_wd_ht_tmp = pu2_src[wd - 1 + (ht - 1) * src_strd];
    }

    *pu2_src_top_left = u2_src_top_left_tmp;

    /* 2. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        u2_top_left = pu2_src_left[0];
        pu2_src_left[0] = pu2_src[wd - 1];
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 1;
    }
    if(0 == pu1_avail[3])
    {
        pu2_src_left[ht - 1] = pu2_src_org[(ht - 1) * src_strd + wd - 1];
        ht_tmp--;
    }

    int8x8_t offset_edge_tbl = vtbl1_s8(vld1_s8(pi1_sao_offset), vld1_s8(gi1_table_edge_idx));

    int8_t i1_avail_left    = (0 == pu1_avail[0]) ? 10: 2;
    int8_t i1_avail_right   = (0 == pu1_avail[1]) ? 10: 2;
    int8x8_t const_2     = vdup_n_s8(2);
    int8x16_t const_2_q  = vdupq_n_s8(2);
    int16x8_t v_zero_s16 = vdupq_n_s16(0);
    uint16x8_t v_max_val = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 16; col += 16)
    {
        int8x16_t const2_cur = const_2_q;
        if(0 == col)
        {
            const2_cur = vsetq_lane_s8(i1_avail_left, const2_cur, 0);
        }
        if(col == wd - 16)
        {
            const2_cur = vsetq_lane_s8(i1_avail_right, const2_cur, 15);
        }

        UWORD16 *pu2_src_cpy  = pu2_src + col;
        uint16x8_t top_lo_raw = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_hi_raw = vld1q_u16(&pu2_src_top_cpy[col + 8]);
        uint16x8_t top_lo     = vextq_u16(vsetq_lane_u16(u2_top_left, top_lo_raw, 7), top_lo_raw, 7);
        uint16x8_t top_hi     = vextq_u16(top_lo_raw, top_hi_raw, 7);
        u2_top_left           = pu2_src_top_cpy[col + 15];

        vst1q_u16(&pu2_src_top[col],     vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
        vst1q_u16(&pu2_src_top[col + 8], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col + 8]));

        uint16x8_t cur0_lo = vld1q_u16(pu2_src_cpy);
        uint16x8_t cur0_hi = vld1q_u16(pu2_src_cpy + 8);
        int8x16_t sign_up0 = vreinterpretq_s8_u8(vcombine_u8(
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, top_lo), vcgtq_u16(cur0_lo, top_lo))),
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, top_hi), vcgtq_u16(cur0_hi, top_hi)))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t cur1_lo   = vld1q_u16(pu2_row1);
            uint16x8_t cur1_hi   = vld1q_u16(pu2_row1 + 8);
            uint16x8_t cur1_lo_s = vld1q_u16(pu2_row1 + 1);
            uint16x8_t cur1_hi_s = vld1q_u16(pu2_row1 + 9);

            UWORD16 u2_left0          = pu2_src_left_cpy[row];
            pu2_src_left_cpy[row]     = pu2_row0[15];

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, cur1_lo_s), vcgtq_u16(cur0_lo, cur1_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, cur1_hi_s), vcgtq_u16(cur0_hi, cur1_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_row0,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_row0 + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));

            int8x16_t neg_down0 = vnegq_s8(sign_down0);
            int8x16_t sign_up1  = vsetq_lane_s8(SIGN(pu2_row1[0] - u2_left0), neg_down0, 15);
            sign_up1            = vextq_s8(sign_up1, neg_down0, 15);

            uint16x8_t next_lo   = vld1q_u16(pu2_row2);
            uint16x8_t next_hi   = vld1q_u16(pu2_row2 + 8);
            uint16x8_t next_lo_s = vld1q_u16(pu2_row2 + 1);
            uint16x8_t next_hi_s = vld1q_u16(pu2_row2 + 9);

            UWORD16 u2_left1          = pu2_src_left_cpy[row + 1];
            pu2_src_left_cpy[row + 1] = pu2_row1[15];

            int8x16_t sign_down1 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_lo, next_lo_s), vcgtq_u16(cur1_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_hi, next_hi_s), vcgtq_u16(cur1_hi, next_hi_s)))));

            int8x16_t idx1   = vaddq_s8(vaddq_s8(sign_down1, const2_cur), sign_up1);
            int8x8_t off1_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx1));
            int8x8_t off1_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx1));

            int16x8_t sum1_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_lo), off1_lo), v_zero_s16);
            int16x8_t sum1_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_hi), off1_hi), v_zero_s16);

            vst1q_u16(pu2_row1,     vminq_u16(vreinterpretq_u16_s16(sum1_lo), v_max_val));
            vst1q_u16(pu2_row1 + 8, vminq_u16(vreinterpretq_u16_s16(sum1_hi), v_max_val));

            int8x16_t neg_down1 = vnegq_s8(sign_down1);
            sign_up0            = vsetq_lane_s8(SIGN(pu2_row2[0] - u2_left1), neg_down1, 15);
            sign_up0            = vextq_s8(sign_up0, neg_down1, 15);

            cur0_lo     = next_lo;
            cur0_hi     = next_hi;
            pu2_src_cpy = pu2_row2;
        }

        if(row < ht_tmp)
        {
            UWORD16 *pu2_next    = pu2_src_cpy + src_strd;
            uint16x8_t next_lo_s = vld1q_u16(pu2_next + 1);
            uint16x8_t next_hi_s = vld1q_u16(pu2_next + 9);

            pu2_src_left_cpy[row] = pu2_src_cpy[15];

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, next_lo_s), vcgtq_u16(cur0_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, next_hi_s), vcgtq_u16(cur0_hi, next_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_src_cpy,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_src_cpy + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));
        }
    }

    if(col < wd)
    {
        int8x8_t const2_cur = const_2;
        if(0 == col)
        {
            const2_cur = vset_lane_s8(i1_avail_left, const2_cur, 0);
        }
        const2_cur = vset_lane_s8(i1_avail_right, const2_cur, 7);

        UWORD16 *pu2_src_cpy   = pu2_src + col;
        uint16x8_t top_row_raw = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_row     = vextq_u16(vsetq_lane_u16(u2_top_left, top_row_raw, 7), top_row_raw, 7);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(
            vsubq_u16(vcltq_u16(cur_row, top_row), vcgtq_u16(cur_row, top_row))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1     = vld1q_u16(pu2_row1);
            uint16x8_t next_row1_tmp = vld1q_u16(pu2_row1 + 1);
            uint16x8_t next_row2     = vld1q_u16(pu2_row2);
            uint16x8_t next_row2_tmp = vld1q_u16(pu2_row2 + 1);

            UWORD16 u2_left0 = pu2_src_left_cpy[row];
            UWORD16 u2_left1 = pu2_src_left_cpy[row + 1];
            pu2_src_left_cpy[row]     = pu2_row0[7];
            pu2_src_left_cpy[row + 1] = pu2_row1[7];

            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row1_tmp), vcgtq_u16(cur_row, next_row1_tmp))));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(next_row1, next_row2_tmp), vcgtq_u16(next_row1, next_row2_tmp))));

            int8x8_t idx0 = vadd_s8(vadd_s8(sign_down0, const2_cur), sign_up0);

            int8x8_t neg_down0 = vneg_s8(sign_down0);
            int8x8_t sign_up1  = vset_lane_s8(SIGN(pu2_row1[0] - u2_left0), neg_down0, 7);
            sign_up1           = vext_s8(sign_up1, neg_down0, 7);
            int8x8_t idx1      = vadd_s8(vadd_s8(sign_down1, const2_cur), sign_up1);

            int8x8_t neg_down1 = vneg_s8(sign_down1);
            sign_up0           = vset_lane_s8(SIGN(pu2_row2[0] - u2_left1), neg_down1, 7);
            sign_up0           = vext_s8(sign_up0, neg_down1, 7);

            int8x8_t off0 = vtbl1_s8(offset_edge_tbl, idx0);
            int8x8_t off1 = vtbl1_s8(offset_edge_tbl, idx1);

            int16x8_t sum0 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), off0), v_zero_s16);
            int16x8_t sum1 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), off1), v_zero_s16);

            vst1q_u16(pu2_row0, vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val));
            vst1q_u16(pu2_row1, vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val));

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        if(row < ht_tmp)
        {
            uint16x8_t next_row_tmp = vld1q_u16(pu2_src_cpy + src_strd + 1);
            pu2_src_left_cpy[row]   = pu2_src_cpy[7];

            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row_tmp), vcgtq_u16(cur_row, next_row_tmp))));

            int8x8_t idx0   = vadd_s8(vadd_s8(sign_down, const2_cur), sign_up0);
            int8x8_t offset = vtbl1_s8(offset_edge_tbl, idx0);

            int16x8_t sum_s16 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            vst1q_u16(pu2_src_cpy, vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val));
        }
    }

    pu2_src_org[0] = u2_pos_0_0_tmp;
    pu2_src_org[wd - 1 + (ht - 1) * src_strd] = u2_pos_wd_ht_tmp;
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 3 (45-degree diagonal) filter for 10-bit Luma blocks
*  using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class3_neonintr(UWORD16 *pu2_src,
                                               WORD32 src_strd,
                                               UWORD16 *pu2_src_left,
                                               UWORD16 *pu2_src_top,
                                               UWORD16 *pu2_src_top_left,
                                               UWORD16 *pu2_src_top_right,
                                               UWORD16 *pu2_src_bot_left,
                                               UWORD8 *pu1_avail,
                                               WORD8 *pi1_sao_offset,
                                               WORD32 wd,
                                               WORD32 ht,
                                               UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    WORD32 ht_tmp = ht;
    UWORD16 u2_src_top_left_tmp;
    UWORD16 u2_pos_wd_0_tmp;
    UWORD16 u2_pos_0_ht_tmp;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and corner pixels before any modifications */
    u2_src_top_left_tmp = pu2_src_top[wd - 1];

    if(0 != pu1_avail[5])
    {
        WORD32 edge_idx = 2 + SIGN(pu2_src[wd - 1] - pu2_src_top_right[0]) +
                              SIGN(pu2_src[wd - 1] - pu2_src[wd - 2 + src_strd]);
        edge_idx = gi1_table_edge_idx[edge_idx];
        u2_pos_wd_0_tmp = (0 != edge_idx) ? CLIP3(pu2_src[wd - 1] + pi1_sao_offset[edge_idx], 0, max_pixel_val)
                                          : pu2_src[wd - 1];
    }
    else
    {
        u2_pos_wd_0_tmp = pu2_src[wd - 1];
    }

    if(0 != pu1_avail[6])
    {
        WORD32 bl_idx = (ht - 1) * src_strd;
        WORD32 edge_idx = 2 + SIGN(pu2_src[bl_idx] - pu2_src[bl_idx + 1 - src_strd]) +
                              SIGN(pu2_src[bl_idx] - pu2_src_bot_left[0]);
        edge_idx = gi1_table_edge_idx[edge_idx];
        u2_pos_0_ht_tmp = (0 != edge_idx) ? CLIP3(pu2_src[bl_idx] + pi1_sao_offset[edge_idx], 0, max_pixel_val)
                                          : pu2_src[bl_idx];
    }
    else
    {
        u2_pos_0_ht_tmp = pu2_src[(ht - 1) * src_strd];
    }

    *pu2_src_top_left = u2_src_top_left_tmp;

    /* 2. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src_left[0] = pu2_src[wd - 1];
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 1;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    int8x8_t offset_edge_tbl = vtbl1_s8(vld1_s8(pi1_sao_offset), vld1_s8(gi1_table_edge_idx));

    int8_t i1_avail_left    = (0 == pu1_avail[0]) ? 10: 2;
    int8_t i1_avail_right   = (0 == pu1_avail[1]) ? 10: 2;
    int8x8_t const_2     = vdup_n_s8(2);
    int8x16_t const_2_q  = vdupq_n_s8(2);
    int16x8_t v_zero_s16 = vdupq_n_s16(0);
    uint16x8_t v_max_val = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 16; col += 16)
    {
        int8x16_t const2_cur = const_2_q;
        if(0 == col)
        {
            const2_cur = vsetq_lane_s8(i1_avail_left, const2_cur, 0);
        }
        if(col == wd - 16)
        {
            const2_cur = vsetq_lane_s8(i1_avail_right, const2_cur, 15);
        }

        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_lo    = vld1q_u16(&pu2_src_top_cpy[col + 1]);
        uint16x8_t top_hi    = vld1q_u16(&pu2_src_top_cpy[col + 9]);

        vst1q_u16(&pu2_src_top[col],     vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
        vst1q_u16(&pu2_src_top[col + 8], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col + 8]));

        uint16x8_t cur0_lo = vld1q_u16(pu2_src_cpy);
        uint16x8_t cur0_hi = vld1q_u16(pu2_src_cpy + 8);
        int8x16_t sign_up0 = vreinterpretq_s8_u8(vcombine_u8(
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, top_lo), vcgtq_u16(cur0_lo, top_lo))),
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, top_hi), vcgtq_u16(cur0_hi, top_hi)))));

        for(row = 0; row < ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t cur1_lo    = vld1q_u16(pu2_row1);
            uint16x8_t cur1_hi    = vld1q_u16(pu2_row1 + 8);
            UWORD16 u2_dl0        = pu2_src_left_cpy[row + 1];
            pu2_src_left_cpy[row] = pu2_row0[15];

            uint16x8_t cur1_lo_s = vextq_u16(vdupq_n_u16(u2_dl0), cur1_lo, 7);
            uint16x8_t cur1_hi_s = vextq_u16(cur1_lo, cur1_hi, 7);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, cur1_lo_s), vcgtq_u16(cur0_lo, cur1_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, cur1_hi_s), vcgtq_u16(cur0_hi, cur1_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_row0,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_row0 + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));

            UWORD16 u2_row1_15  = pu2_row1[15];
            int8x16_t neg_down0 = vnegq_s8(sign_down0);
            int8x16_t sign_up1  = vextq_s8(neg_down0, vdupq_n_s8(SIGN(u2_row1_15 - pu2_row0[16])), 1);

            uint16x8_t next_lo        = vld1q_u16(pu2_row2);
            uint16x8_t next_hi        = vld1q_u16(pu2_row2 + 8);
            UWORD16 u2_dl1            = pu2_src_left_cpy[row + 2];
            pu2_src_left_cpy[row + 1] = u2_row1_15;

            uint16x8_t next_lo_s = vextq_u16(vdupq_n_u16(u2_dl1), next_lo, 7);
            uint16x8_t next_hi_s = vextq_u16(next_lo, next_hi, 7);

            int8x16_t sign_down1 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_lo, next_lo_s), vcgtq_u16(cur1_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_hi, next_hi_s), vcgtq_u16(cur1_hi, next_hi_s)))));

            int8x16_t idx1   = vaddq_s8(vaddq_s8(sign_down1, const2_cur), sign_up1);
            int8x8_t off1_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx1));
            int8x8_t off1_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx1));

            int16x8_t sum1_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_lo), off1_lo), v_zero_s16);
            int16x8_t sum1_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_hi), off1_hi), v_zero_s16);

            vst1q_u16(pu2_row1,     vminq_u16(vreinterpretq_u16_s16(sum1_lo), v_max_val));
            vst1q_u16(pu2_row1 + 8, vminq_u16(vreinterpretq_u16_s16(sum1_hi), v_max_val));

            int8x16_t neg_down1 = vnegq_s8(sign_down1);
            sign_up0            = vextq_s8(neg_down1, vdupq_n_s8(SIGN(pu2_row2[15] - pu2_row1[16])), 1);

            cur0_lo     = next_lo;
            cur0_hi     = next_hi;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            UWORD16 *pu2_next     = pu2_src_cpy + src_strd;
            uint16x8_t next_lo    = vld1q_u16(pu2_next);
            uint16x8_t next_hi    = vld1q_u16(pu2_next + 8);
            UWORD16 u2_dl         = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                    ? pu2_next[-1]
                                    : pu2_src_left_cpy[row + 1];
            pu2_src_left_cpy[row] = pu2_src_cpy[15];

            uint16x8_t next_lo_s = vextq_u16(vdupq_n_u16(u2_dl), next_lo, 7);
            uint16x8_t next_hi_s = vextq_u16(next_lo, next_hi, 7);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, next_lo_s), vcgtq_u16(cur0_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, next_hi_s), vcgtq_u16(cur0_hi, next_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl1_s8(offset_edge_tbl, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl1_s8(offset_edge_tbl, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_src_cpy,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_src_cpy + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));

            int8x16_t neg_down0 = vnegq_s8(sign_down0);
            sign_up0    = vextq_s8(neg_down0, vdupq_n_s8(SIGN(pu2_next[15] - pu2_src_cpy[16])), 1);
            cur0_lo     = next_lo;
            cur0_hi     = next_hi;
            pu2_src_cpy = pu2_next;
        }

        if(0 == pu1_avail[3])
        {
            pu2_src_left[ht - 1] = pu2_src_cpy[15];
        }
    }

    if(col < wd)
    {
        int8x8_t const2_cur = const_2;
        if(0 == col)
        {
            const2_cur = vset_lane_s8(i1_avail_left, const2_cur, 0);
        }
        const2_cur = vset_lane_s8(i1_avail_right, const2_cur, 7);

        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col + 1]);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(
            vsubq_u16(vcltq_u16(cur_row, top_row), vcgtq_u16(cur_row, top_row))));

        for(row = 0; row < ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1  = vld1q_u16(pu2_row1);
            UWORD16 u2_dl0        = pu2_src_left_cpy[row + 1];
            pu2_src_left_cpy[row] = pu2_row0[7];
            uint16x8_t next_row1_tmp = vextq_u16(vdupq_n_u16(u2_dl0), next_row1, 7);

            uint16x8_t next_row2      = vld1q_u16(pu2_row2);
            UWORD16 u2_dl1            = pu2_src_left_cpy[row + 2];
            UWORD16 u2_row1_7         = pu2_row1[7];
            pu2_src_left_cpy[row + 1] = u2_row1_7;
            uint16x8_t next_row2_tmp  = vextq_u16(vdupq_n_u16(u2_dl1), next_row2, 7);

            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row1_tmp), vcgtq_u16(cur_row, next_row1_tmp))));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(next_row1, next_row2_tmp), vcgtq_u16(next_row1, next_row2_tmp))));

            int8x8_t idx0 = vadd_s8(vadd_s8(sign_down0, const2_cur), sign_up0);

            int8x8_t neg_down0 = vneg_s8(sign_down0);
            int8x8_t sign_up1  = vext_s8(neg_down0, vdup_n_s8(SIGN(u2_row1_7 - pu2_row0[8])), 1);
            int8x8_t idx1      = vadd_s8(vadd_s8(sign_down1, const2_cur), sign_up1);

            int8x8_t neg_down1 = vneg_s8(sign_down1);
            sign_up0           = vext_s8(neg_down1, vdup_n_s8(SIGN(pu2_row2[7] - pu2_row1[8])), 1);

            int8x8_t off0 = vtbl1_s8(offset_edge_tbl, idx0);
            int8x8_t off1 = vtbl1_s8(offset_edge_tbl, idx1);

            int16x8_t sum0 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), off0), v_zero_s16);
            int16x8_t sum1 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), off1), v_zero_s16);

            vst1q_u16(pu2_row0, vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val));
            vst1q_u16(pu2_row1, vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val));

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            UWORD16 *pu2_next     = pu2_src_cpy + src_strd;
            uint16x8_t next_row   = vld1q_u16(pu2_next);
            UWORD16 u2_dl         = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                    ? pu2_next[-1]
                                    : pu2_src_left_cpy[row + 1];
            pu2_src_left_cpy[row] = pu2_src_cpy[7];
            uint16x8_t next_row_tmp = vextq_u16(vdupq_n_u16(u2_dl), next_row, 7);

            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row_tmp), vcgtq_u16(cur_row, next_row_tmp))));

            int8x8_t idx0   = vadd_s8(vadd_s8(sign_down, const2_cur), sign_up0);
            int8x8_t offset = vtbl1_s8(offset_edge_tbl, idx0);

            int16x8_t sum_s16 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            vst1q_u16(pu2_src_cpy, vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val));

            int8x8_t neg_down = vneg_s8(sign_down);
            sign_up0    = vext_s8(neg_down, vdup_n_s8(SIGN(pu2_next[7] - pu2_src_cpy[8])), 1);
            cur_row     = next_row;
            pu2_src_cpy = pu2_next;
        }

        if(0 == pu1_avail[3])
        {
            pu2_src_left[ht - 1] = pu2_src_cpy[7];
        }
    }

    pu2_src_org[wd - 1] = u2_pos_wd_0_tmp;
    pu2_src_org[(ht - 1) * src_strd] = u2_pos_0_ht_tmp;
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 0 (Horizontal 0-degree) filter for 10-bit interleaved
*  Chroma (U, V) blocks using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class0_chroma_neonintr(UWORD16 *pu2_src,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_src_left,
                                                      UWORD16 *pu2_src_top,
                                                      UWORD16 *pu2_src_top_left,
                                                      UWORD16 *pu2_src_top_right,
                                                      UWORD16 *pu2_src_bot_left,
                                                      UWORD8 *pu1_avail,
                                                      WORD8 *pi1_sao_offset_u,
                                                      WORD8 *pi1_sao_offset_v,
                                                      WORD32 wd,
                                                      WORD32 ht,
                                                      UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD16 *pu2_src_cpy, *pu2_src_left_cpy;

    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Update top-left boundary buffer from unmodified pu2_src_top */
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];

    int8x8_t edge_idx_tbl         = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u    = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v    = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);
    int8x8x2_t offset_edge_tbl_uv = { { offset_edge_tbl_u, offset_edge_tbl_v } };

    uint16_t u2_avail_left  = (0 == pu1_avail[0]) ? 0x2020 : 0x0A02;
    uint16_t u2_avail_right = (0 == pu1_avail[1]) ? 0x2020 : 0x0A02;
    int8x16_t const_2_10_q  = vreinterpretq_s8_u16(vdupq_n_u16(0x0A02));
    int16x8_t v_zero_s16    = vdupq_n_s16(0);
    uint16x8_t v_max_val    = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 16; col += 16)
    {
        int8x16_t const2_cur = const_2_10_q;

        vst1q_u16(pu2_src_top,     vld1q_u16(pu2_src + (ht - 1) * src_strd));
        vst1q_u16(pu2_src_top + 8, vld1q_u16(pu2_src + 8 + (ht - 1) * src_strd));

        pu2_src_cpy          = pu2_src;
        pu2_src_left_cpy     = pu2_src_left;
        if(0 == col)
        {
            const2_cur = vreinterpretq_s8_u16(
                vsetq_lane_u16(u2_avail_left, vreinterpretq_u16_s8(const2_cur), 0));
        }
        if(col == wd - 16)
        {
            const2_cur = vreinterpretq_s8_u16(
                vsetq_lane_u16(u2_avail_right, vreinterpretq_u16_s8(const2_cur), 7));
        }

        for(row = 0; row < ht; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;

            uint16x8_t s0_lo     = vld1q_u16(pu2_row0);
            uint16x8_t s0_hi     = vld1q_u16(pu2_row0 + 8);
            uint16x8_t right0_lo = vld1q_u16(pu2_row0 + 2);
            uint16x8_t right0_hi = vld1q_u16(pu2_row0 + 10);

            uint16x8_t s1_lo     = vld1q_u16(pu2_row1);
            uint16x8_t s1_hi     = vld1q_u16(pu2_row1 + 8);
            uint16x8_t right1_lo = vld1q_u16(pu2_row1 + 2);
            uint16x8_t right1_hi = vld1q_u16(pu2_row1 + 10);

            uint16x8_t left0_lo = vreinterpretq_u16_u32(vld1q_dup_u32((const uint32_t *)pu2_src_left_cpy));
            uint16x8_t left1_lo = vreinterpretq_u16_u32(vld1q_dup_u32((const uint32_t *)(pu2_src_left_cpy + 2)));
            uint16x8_t left0_hi = vreinterpretq_u16_u32(vdupq_lane_u32(vreinterpret_u32_u16(vget_low_u16(s0_lo)), 0));
            uint16x8_t left1_hi = vreinterpretq_u16_u32(vdupq_lane_u32(vreinterpret_u32_u16(vget_low_u16(s1_lo)), 0));

            vst1q_lane_u32((uint32_t *)pu2_src_left_cpy,       vreinterpretq_u32_u16(s0_hi), 3);
            vst1q_lane_u32((uint32_t *)(pu2_src_left_cpy + 2), vreinterpretq_u32_u16(s1_hi), 3);

            int8x16_t sign_right0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(s0_lo, right0_lo), vcgtq_u16(s0_lo, right0_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(s0_hi, right0_hi), vcgtq_u16(s0_hi, right0_hi)))));
            int8x16_t sign_right1 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(s1_lo, right1_lo), vcgtq_u16(s1_lo, right1_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(s1_hi, right1_hi), vcgtq_u16(s1_hi, right1_hi)))));

            int8x8_t sign_l0_8x8 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(left0_hi, left0_lo), vcgtq_u16(left0_hi, left0_lo))));
            int8x8_t sign_l1_8x8 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(left1_hi, left1_lo), vcgtq_u16(left1_hi, left1_lo))));

            int8x16_t sign_left0 = vextq_s8(vcombine_s8(sign_l0_8x8, sign_l0_8x8), vnegq_s8(sign_right0), 14);
            int8x16_t sign_left1 = vextq_s8(vcombine_s8(sign_l1_8x8, sign_l1_8x8), vnegq_s8(sign_right1), 14);

            int8x16_t idx0 = vaddq_s8(vaddq_s8(sign_right0, const2_cur), sign_left0);
            int8x16_t idx1 = vaddq_s8(vaddq_s8(sign_right1, const2_cur), sign_left1);

            int8x8_t off0_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx0));
            int8x8_t off1_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx1));
            int8x8_t off1_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx1));

            s0_lo = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(s0_lo), off0_lo), v_zero_s16)), v_max_val);
            s0_hi = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(s0_hi), off0_hi), v_zero_s16)), v_max_val);
            s1_lo = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(s1_lo), off1_lo), v_zero_s16)), v_max_val);
            s1_hi = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(s1_hi), off1_hi), v_zero_s16)), v_max_val);

            vst1q_u16(pu2_row0,     s0_lo);
            vst1q_u16(pu2_row0 + 8, s0_hi);
            vst1q_u16(pu2_row1,     s1_lo);
            vst1q_u16(pu2_row1 + 8, s1_hi);

            pu2_src_cpy += 2 * src_strd;
            pu2_src_left_cpy += 4;
        }
        pu2_src += 16;
        pu2_src_top += 16;
    }

    if(col < wd)
    {
        int8x8_t const2_rem = vget_low_s8(const_2_10_q);

        vst1q_u16(pu2_src_top, vld1q_u16(pu2_src + (ht - 1) * src_strd));

        if(0 == col)
        {
            const2_rem = vreinterpret_s8_u16(
                vset_lane_u16(u2_avail_left, vreinterpret_u16_s8(const2_rem), 0));
        }
        const2_rem = vreinterpret_s8_u16(
            vset_lane_u16(u2_avail_right, vreinterpret_u16_s8(const2_rem), 3));
        int8x16_t const2_rem_q = vcombine_s8(const2_rem, const2_rem);

        pu2_src_cpy      = pu2_src;
        pu2_src_left_cpy = pu2_src_left;
        for(row = 0; row < ht; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;

            uint16x8_t cur_row0   = vld1q_u16(pu2_row0);
            uint16x8_t right_row0 = vld1q_u16(pu2_row0 + 2);
            uint16x8_t cur_row1   = vld1q_u16(pu2_row1);
            uint16x8_t right_row1 = vld1q_u16(pu2_row1 + 2);

            uint16x8_t left_row0 = vreinterpretq_u16_u32(vld1q_dup_u32((const uint32_t *)pu2_src_left_cpy));
            uint16x8_t left_row1 = vreinterpretq_u16_u32(vld1q_dup_u32((const uint32_t *)(pu2_src_left_cpy + 2)));
            left_row0            = vextq_u16(left_row0, cur_row0, 6);
            left_row1            = vextq_u16(left_row1, cur_row1, 6);

            vst1q_lane_u32((uint32_t *)pu2_src_left_cpy,       vreinterpretq_u32_u16(cur_row0), 3);
            vst1q_lane_u32((uint32_t *)(pu2_src_left_cpy + 2), vreinterpretq_u32_u16(cur_row1), 3);

            uint16x8_t diff_l0  = vsubq_u16(vcltq_u16(cur_row0, left_row0), vcgtq_u16(cur_row0, left_row0));
            uint16x8_t diff_l1  = vsubq_u16(vcltq_u16(cur_row1, left_row1), vcgtq_u16(cur_row1, left_row1));
            int8x16_t sign_left = vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(diff_l0), vmovn_u16(diff_l1)));

            uint16x8_t diff_r0   = vsubq_u16(vcltq_u16(cur_row0, right_row0), vcgtq_u16(cur_row0, right_row0));
            uint16x8_t diff_r1   = vsubq_u16(vcltq_u16(cur_row1, right_row1), vcgtq_u16(cur_row1, right_row1));
            int8x16_t sign_right = vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(diff_r0), vmovn_u16(diff_r1)));

            int8x16_t raw_idx = vaddq_s8(vaddq_s8(const2_rem_q, sign_left), sign_right);
            int8x8_t off0     = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(raw_idx));
            int8x8_t off1     = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(raw_idx));

            int16x8_t sum0  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), off0), v_zero_s16);
            int16x8_t sum1  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row1), off1), v_zero_s16);

            vst1q_u16(pu2_row0, vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val));
            vst1q_u16(pu2_row1, vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val));

            pu2_src_cpy += 2 * src_strd;
            pu2_src_left_cpy += 4;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 1 (Vertical 90-degree) filter for 10-bit interleaved
*  Chroma (U, V) blocks using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class1_chroma_neonintr(UWORD16 *pu2_src,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_src_left,
                                                      UWORD16 *pu2_src_top,
                                                      UWORD16 *pu2_src_top_left,
                                                      UWORD16 *pu2_src_top_right,
                                                      UWORD16 *pu2_src_bot_left,
                                                      UWORD8 *pu1_avail,
                                                      WORD8 *pi1_sao_offset_u,
                                                      WORD8 *pi1_sao_offset_v,
                                                      WORD32 wd,
                                                      WORD32 ht,
                                                      UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    WORD32 ht_tmp = ht;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    int8x8_t edge_idx_tbl         = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u    = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v    = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);
    int8x8x2_t offset_edge_tbl_uv = { { offset_edge_tbl_u, offset_edge_tbl_v } };

    int8x16_t const_2_10_q = vreinterpretq_s8_u16(vdupq_n_u16(0x0A02));
    int8x8_t const_2_10    = vget_low_s8(const_2_10_q);
    int16x8_t v_zero_s16   = vdupq_n_s16(0);
    uint16x8_t v_max_val   = vdupq_n_u16((uint16_t)max_pixel_val);

    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];

    /* 1. Update left boundary buffer from unmodified pu2_src */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[2 * row]     = pu2_src[row * src_strd + wd - 2];
        pu2_src_left[2 * row + 1] = pu2_src[row * src_strd + wd - 1];
    }

    /* 2. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht_tmp--;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    for(col = 0; col <= wd - 16; col += 16)
    {
        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_lo    = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_hi    = vld1q_u16(&pu2_src_top_cpy[col + 8]);

        vst1q_u16(&pu2_src_top[col],     vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
        vst1q_u16(&pu2_src_top[col + 8], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col + 8]));

        uint16x8_t cur0_lo = vld1q_u16(pu2_src_cpy);
        uint16x8_t cur0_hi = vld1q_u16(pu2_src_cpy + 8);
        int8x16_t sign_up0 = vreinterpretq_s8_u8(vcombine_u8(
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, top_lo), vcgtq_u16(cur0_lo, top_lo))),
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, top_hi), vcgtq_u16(cur0_hi, top_hi)))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0  = pu2_src_cpy;
            UWORD16 *pu2_row1  = pu2_src_cpy + src_strd;
            uint16x8_t cur1_lo = vld1q_u16(pu2_row1);
            uint16x8_t cur1_hi = vld1q_u16(pu2_row1 + 8);
            uint16x8_t next_lo = vld1q_u16(pu2_row1 + src_strd);
            uint16x8_t next_hi = vld1q_u16(pu2_row1 + src_strd + 8);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, cur1_lo), vcgtq_u16(cur0_lo, cur1_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, cur1_hi), vcgtq_u16(cur0_hi, cur1_hi)))));
            int8x16_t sign_down1 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_lo, next_lo), vcgtq_u16(cur1_lo, next_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_hi, next_hi), vcgtq_u16(cur1_hi, next_hi)))));

            int8x16_t sign_up1 = vnegq_s8(sign_down0);
            int8x16_t idx0     = vaddq_s8(vaddq_s8(sign_down0, const_2_10_q), sign_up0);
            int8x16_t idx1     = vaddq_s8(vaddq_s8(sign_down1, const_2_10_q), sign_up1);
            sign_up0           = vnegq_s8(sign_down1);

            int8x8_t off0_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx0));
            int8x8_t off1_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx1));
            int8x8_t off1_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx1));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);
            int16x8_t sum1_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_lo), off1_lo), v_zero_s16);
            int16x8_t sum1_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_hi), off1_hi), v_zero_s16);

            vst1q_u16(pu2_row0,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_row0 + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));
            vst1q_u16(pu2_row1,     vminq_u16(vreinterpretq_u16_s16(sum1_lo), v_max_val));
            vst1q_u16(pu2_row1 + 8, vminq_u16(vreinterpretq_u16_s16(sum1_hi), v_max_val));

            cur0_lo = next_lo;
            cur0_hi = next_hi;
            pu2_src_cpy += 2 * src_strd;
        }
        if(row < ht_tmp)
        {
            uint16x8_t next_lo = vld1q_u16(pu2_src_cpy + src_strd);
            uint16x8_t next_hi = vld1q_u16(pu2_src_cpy + src_strd + 8);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, next_lo), vcgtq_u16(cur0_lo, next_lo))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, next_hi), vcgtq_u16(cur0_hi, next_hi)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const_2_10_q), sign_up0);
            int8x8_t off0_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_src_cpy,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_src_cpy + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));
        }
    }

    if(col < wd)
    {
        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col]);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row0 = vld1q_u16(pu2_src_cpy);
        int8x8_t sign_up0   = vreinterpret_s8_u8(vmovn_u16(
            vsubq_u16(vcltq_u16(cur_row0, top_row), vcgtq_u16(cur_row0, top_row))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0   = pu2_src_cpy;
            UWORD16 *pu2_row1   = pu2_src_cpy + src_strd;
            uint16x8_t cur_row1 = vld1q_u16(pu2_row1);
            uint16x8_t next_row = vld1q_u16(pu2_row1 + src_strd);

            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row0, cur_row1), vcgtq_u16(cur_row0, cur_row1))));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row1, next_row), vcgtq_u16(cur_row1, next_row))));

            int8x8_t sign_up1 = vneg_s8(sign_down0);
            int8x8_t idx0     = vadd_s8(vadd_s8(sign_down0, const_2_10), sign_up0);
            int8x8_t idx1     = vadd_s8(vadd_s8(sign_down1, const_2_10), sign_up1);
            sign_up0          = vneg_s8(sign_down1);

            int8x8_t off0 = vtbl2_s8(offset_edge_tbl_uv, idx0);
            int8x8_t off1 = vtbl2_s8(offset_edge_tbl_uv, idx1);

            int16x8_t sum0 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), off0), v_zero_s16);
            int16x8_t sum1 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row1), off1), v_zero_s16);

            vst1q_u16(pu2_row0, vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val));
            vst1q_u16(pu2_row1, vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val));

            cur_row0 = next_row;
            pu2_src_cpy += 2 * src_strd;
        }
        if(row < ht_tmp)
        {
            uint16x8_t next_row = vld1q_u16(pu2_src_cpy + src_strd);
            int8x8_t sign_down  = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row0, next_row), vcgtq_u16(cur_row0, next_row))));

            int8x8_t raw_idx = vadd_s8(vadd_s8(sign_down, const_2_10), sign_up0);
            int8x8_t offset  = vtbl2_s8(offset_edge_tbl_uv, raw_idx);

            int16x8_t sum_s16 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), offset), v_zero_s16);
            vst1q_u16(pu2_src_cpy, vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val));
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 2 (135-degree diagonal) filter for 10-bit interleaved
*  Chroma (U, V) blocks using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class2_chroma_neonintr(UWORD16 *pu2_src,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_src_left,
                                                      UWORD16 *pu2_src_top,
                                                      UWORD16 *pu2_src_top_left,
                                                      UWORD16 *pu2_src_top_right,
                                                      UWORD16 *pu2_src_bot_left,
                                                      UWORD8 *pu1_avail,
                                                      WORD8 *pi1_sao_offset_u,
                                                      WORD8 *pi1_sao_offset_v,
                                                      WORD32 wd,
                                                      WORD32 ht,
                                                      UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    WORD32 ht_tmp = ht;
    UWORD16 au2_src_top_left_tmp[2];
    UWORD16 u2_pos_0_0_tmp_u, u2_pos_0_0_tmp_v;
    UWORD16 u2_pos_wd_ht_tmp_u, u2_pos_wd_ht_tmp_v;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and corner pixels before any modifications */
    au2_src_top_left_tmp[0] = pu2_src_top[wd - 2];
    au2_src_top_left_tmp[1] = pu2_src_top[wd - 1];

    if(0 != pu1_avail[4])
    {
        WORD32 edge_idx_u = 2 + SIGN(pu2_src[0] - pu2_src_top_left[0]) +
                                SIGN(pu2_src[0] - pu2_src[2 + src_strd]);
        edge_idx_u = gi1_table_edge_idx[edge_idx_u];
        u2_pos_0_0_tmp_u = (0 != edge_idx_u) ? CLIP3(pu2_src[0] + pi1_sao_offset_u[edge_idx_u], 0, max_pixel_val)
                                             : pu2_src[0];

        WORD32 edge_idx_v = 2 + SIGN(pu2_src[1] - pu2_src_top_left[1]) +
                                SIGN(pu2_src[1] - pu2_src[1 + 2 + src_strd]);
        edge_idx_v = gi1_table_edge_idx[edge_idx_v];
        u2_pos_0_0_tmp_v = (0 != edge_idx_v) ? CLIP3(pu2_src[1] + pi1_sao_offset_v[edge_idx_v], 0, max_pixel_val)
                                             : pu2_src[1];
    }
    else
    {
        u2_pos_0_0_tmp_u = pu2_src[0];
        u2_pos_0_0_tmp_v = pu2_src[1];
    }

    if(0 != pu1_avail[7])
    {
        WORD32 br_u = wd - 2 + (ht - 1) * src_strd;
        WORD32 edge_idx_u = 2 + SIGN(pu2_src[br_u] - pu2_src[br_u - 2 - src_strd]) +
                                SIGN(pu2_src[br_u] - pu2_src[br_u + 2 + src_strd]);
        edge_idx_u = gi1_table_edge_idx[edge_idx_u];
        u2_pos_wd_ht_tmp_u = (0 != edge_idx_u) ? CLIP3(pu2_src[br_u] + pi1_sao_offset_u[edge_idx_u], 0, max_pixel_val)
                                               : pu2_src[br_u];

        WORD32 br_v = wd - 1 + (ht - 1) * src_strd;
        WORD32 edge_idx_v = 2 + SIGN(pu2_src[br_v] - pu2_src[br_v - 2 - src_strd]) +
                                SIGN(pu2_src[br_v] - pu2_src[br_v + 2 + src_strd]);
        edge_idx_v = gi1_table_edge_idx[edge_idx_v];
        u2_pos_wd_ht_tmp_v = (0 != edge_idx_v) ? CLIP3(pu2_src[br_v] + pi1_sao_offset_v[edge_idx_v], 0, max_pixel_val)
                                               : pu2_src[br_v];
    }
    else
    {
        u2_pos_wd_ht_tmp_u = pu2_src[wd - 2 + (ht - 1) * src_strd];
        u2_pos_wd_ht_tmp_v = pu2_src[wd - 1 + (ht - 1) * src_strd];
    }

    UWORD16 u2_top_left_u = pu2_src_top_left[0];
    UWORD16 u2_top_left_v = pu2_src_top_left[1];
    pu2_src_top_left[0]   = au2_src_top_left_tmp[0];
    pu2_src_top_left[1]   = au2_src_top_left_tmp[1];

    /* 2. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        u2_top_left_u   = pu2_src_left[0];
        u2_top_left_v   = pu2_src_left[1];
        pu2_src_left[0] = pu2_src[wd - 2];
        pu2_src_left[1] = pu2_src[wd - 1];
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 2;
    }
    if(0 == pu1_avail[3])
    {
        pu2_src_left[2 * (ht - 1)]     = pu2_src_org[(ht - 1) * src_strd + wd - 2];
        pu2_src_left[2 * (ht - 1) + 1] = pu2_src_org[(ht - 1) * src_strd + wd - 1];
        ht_tmp--;
    }

    int8x8_t edge_idx_tbl         = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u    = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v    = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);
    int8x8x2_t offset_edge_tbl_uv = { { offset_edge_tbl_u, offset_edge_tbl_v } };

    uint16_t u2_avail_left    = (0 == pu1_avail[0]) ? 0x2020 : 0x0A02;
    uint16_t u2_avail_right   = (0 == pu1_avail[1]) ? 0x2020 : 0x0A02;
    int8x16_t const_2_10_q = vreinterpretq_s8_u16(vdupq_n_u16(0x0A02));
    int8x8_t const_2_10    = vget_low_s8(const_2_10_q);
    int16x8_t v_zero_s16   = vdupq_n_s16(0);
    uint16x8_t v_max_val   = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 16; col += 16)
    {
        int8x16_t const2_cur = const_2_10_q;
        if(0 == col)
        {
            const2_cur = vreinterpretq_s8_u16(vsetq_lane_u16(u2_avail_left, vreinterpretq_u16_s8(const2_cur), 0));
        }
        if(col == wd - 16)
        {
            const2_cur = vreinterpretq_s8_u16(vsetq_lane_u16(u2_avail_right, vreinterpretq_u16_s8(const2_cur), 7));
        }

        UWORD16 *pu2_src_cpy  = pu2_src + col;
        uint16x8_t top_lo_raw = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_hi_raw = vld1q_u16(&pu2_src_top_cpy[col + 8]);
        uint16x8_t top_lo     = vsetq_lane_u16(u2_top_left_u, top_lo_raw, 6);
        top_lo                = vsetq_lane_u16(u2_top_left_v, top_lo, 7);
        top_lo                = vextq_u16(top_lo, top_lo_raw, 6);
        uint16x8_t top_hi     = vextq_u16(top_lo_raw, top_hi_raw, 6);
        u2_top_left_u         = pu2_src_top_cpy[col + 14];
        u2_top_left_v         = pu2_src_top_cpy[col + 15];

        vst1q_u16(&pu2_src_top[col],     vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
        vst1q_u16(&pu2_src_top[col + 8], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col + 8]));

        uint16x8_t cur0_lo = vld1q_u16(pu2_src_cpy);
        uint16x8_t cur0_hi = vld1q_u16(pu2_src_cpy + 8);
        int8x16_t sign_up0 = vreinterpretq_s8_u8(vcombine_u8(
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, top_lo), vcgtq_u16(cur0_lo, top_lo))),
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, top_hi), vcgtq_u16(cur0_hi, top_hi)))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t cur1_lo   = vld1q_u16(pu2_row1);
            uint16x8_t cur1_hi   = vld1q_u16(pu2_row1 + 8);
            uint16x8_t cur1_lo_s = vld1q_u16(pu2_row1 + 2);
            uint16x8_t cur1_hi_s = vld1q_u16(pu2_row1 + 10);

            UWORD16 u2_left0_u            = pu2_src_left_cpy[2 * row];
            UWORD16 u2_left0_v            = pu2_src_left_cpy[2 * row + 1];
            pu2_src_left_cpy[2 * row]     = pu2_row0[14];
            pu2_src_left_cpy[2 * row + 1] = pu2_row0[15];

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, cur1_lo_s), vcgtq_u16(cur0_lo, cur1_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, cur1_hi_s), vcgtq_u16(cur0_hi, cur1_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_row0,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_row0 + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));

            int8x16_t neg_down0 = vnegq_s8(sign_down0);
            int8x16_t sign_up1  = vsetq_lane_s8(SIGN(pu2_row1[0] - u2_left0_u), neg_down0, 14);
            sign_up1            = vsetq_lane_s8(SIGN(pu2_row1[1] - u2_left0_v), sign_up1, 15);
            sign_up1            = vextq_s8(sign_up1, neg_down0, 14);

            uint16x8_t next_lo   = vld1q_u16(pu2_row2);
            uint16x8_t next_hi   = vld1q_u16(pu2_row2 + 8);
            uint16x8_t next_lo_s = vld1q_u16(pu2_row2 + 2);
            uint16x8_t next_hi_s = vld1q_u16(pu2_row2 + 10);

            UWORD16 u2_left1_u                  = pu2_src_left_cpy[2 * (row + 1)];
            UWORD16 u2_left1_v                  = pu2_src_left_cpy[2 * (row + 1) + 1];
            pu2_src_left_cpy[2 * (row + 1)]     = pu2_row1[14];
            pu2_src_left_cpy[2 * (row + 1) + 1] = pu2_row1[15];

            int8x16_t sign_down1 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_lo, next_lo_s), vcgtq_u16(cur1_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_hi, next_hi_s), vcgtq_u16(cur1_hi, next_hi_s)))));

            int8x16_t idx1   = vaddq_s8(vaddq_s8(sign_down1, const2_cur), sign_up1);
            int8x8_t off1_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx1));
            int8x8_t off1_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx1));

            int16x8_t sum1_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_lo), off1_lo), v_zero_s16);
            int16x8_t sum1_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_hi), off1_hi), v_zero_s16);

            vst1q_u16(pu2_row1,     vminq_u16(vreinterpretq_u16_s16(sum1_lo), v_max_val));
            vst1q_u16(pu2_row1 + 8, vminq_u16(vreinterpretq_u16_s16(sum1_hi), v_max_val));

            int8x16_t neg_down1 = vnegq_s8(sign_down1);
            sign_up0            = vsetq_lane_s8(SIGN(pu2_row2[0] - u2_left1_u), neg_down1, 14);
            sign_up0            = vsetq_lane_s8(SIGN(pu2_row2[1] - u2_left1_v), sign_up0, 15);
            sign_up0            = vextq_s8(sign_up0, neg_down1, 14);

            cur0_lo     = next_lo;
            cur0_hi     = next_hi;
            pu2_src_cpy = pu2_row2;
        }

        if(row < ht_tmp)
        {
            UWORD16 *pu2_next    = pu2_src_cpy + src_strd;
            uint16x8_t next_lo_s = vld1q_u16(pu2_next + 2);
            uint16x8_t next_hi_s = vld1q_u16(pu2_next + 10);

            pu2_src_left_cpy[2 * row]     = pu2_src_cpy[14];
            pu2_src_left_cpy[2 * row + 1] = pu2_src_cpy[15];

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, next_lo_s), vcgtq_u16(cur0_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, next_hi_s), vcgtq_u16(cur0_hi, next_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_src_cpy,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_src_cpy + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));
        }
    }

    if(col < wd)
    {
        int8x8_t const2_cur = const_2_10;
        if(0 == col)
        {
            const2_cur = vreinterpret_s8_u16(vset_lane_u16(u2_avail_left, vreinterpret_u16_s8(const2_cur), 0));
        }
        const2_cur = vreinterpret_s8_u16(vset_lane_u16(u2_avail_right, vreinterpret_u16_s8(const2_cur), 3));

        UWORD16 *pu2_src_cpy   = pu2_src + col;
        uint16x8_t top_row_raw = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_row     = vsetq_lane_u16(u2_top_left_u, top_row_raw, 6);
        top_row                = vsetq_lane_u16(u2_top_left_v, top_row, 7);
        top_row                = vextq_u16(top_row, top_row_raw, 6);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(
            vsubq_u16(vcltq_u16(cur_row, top_row), vcgtq_u16(cur_row, top_row))));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1     = vld1q_u16(pu2_row1);
            uint16x8_t next_row1_tmp = vld1q_u16(pu2_row1 + 2);
            uint16x8_t next_row2     = vld1q_u16(pu2_row2);
            uint16x8_t next_row2_tmp = vld1q_u16(pu2_row2 + 2);

            UWORD16 u2_left0_u                  = pu2_src_left_cpy[2 * row];
            UWORD16 u2_left0_v                  = pu2_src_left_cpy[2 * row + 1];
            UWORD16 u2_left1_u                  = pu2_src_left_cpy[2 * (row + 1)];
            UWORD16 u2_left1_v                  = pu2_src_left_cpy[2 * (row + 1) + 1];
            pu2_src_left_cpy[2 * row]           = pu2_row0[6];
            pu2_src_left_cpy[2 * row + 1]       = pu2_row0[7];
            pu2_src_left_cpy[2 * (row + 1)]     = pu2_row1[6];
            pu2_src_left_cpy[2 * (row + 1) + 1] = pu2_row1[7];

            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row1_tmp), vcgtq_u16(cur_row, next_row1_tmp))));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(next_row1, next_row2_tmp), vcgtq_u16(next_row1, next_row2_tmp))));

            int8x8_t idx0 = vadd_s8(vadd_s8(sign_down0, const2_cur), sign_up0);

            int8x8_t neg_down0 = vneg_s8(sign_down0);
            int8x8_t sign_up1  = vset_lane_s8(SIGN(pu2_row1[0] - u2_left0_u), neg_down0, 6);
            sign_up1           = vset_lane_s8(SIGN(pu2_row1[1] - u2_left0_v), sign_up1, 7);
            sign_up1           = vext_s8(sign_up1, neg_down0, 6);
            int8x8_t idx1      = vadd_s8(vadd_s8(sign_down1, const2_cur), sign_up1);

            int8x8_t neg_down1 = vneg_s8(sign_down1);
            sign_up0           = vset_lane_s8(SIGN(pu2_row2[0] - u2_left1_u), neg_down1, 6);
            sign_up0           = vset_lane_s8(SIGN(pu2_row2[1] - u2_left1_v), sign_up0, 7);
            sign_up0           = vext_s8(sign_up0, neg_down1, 6);

            int8x8_t off0 = vtbl2_s8(offset_edge_tbl_uv, idx0);
            int8x8_t off1 = vtbl2_s8(offset_edge_tbl_uv, idx1);

            int16x8_t sum0 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), off0), v_zero_s16);
            int16x8_t sum1 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), off1), v_zero_s16);

            vst1q_u16(pu2_row0, vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val));
            vst1q_u16(pu2_row1, vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val));

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        if(row < ht_tmp)
        {
            uint16x8_t next_row_tmp       = vld1q_u16(pu2_src_cpy + src_strd + 2);
            pu2_src_left_cpy[2 * row]     = pu2_src_cpy[6];
            pu2_src_left_cpy[2 * row + 1] = pu2_src_cpy[7];

            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row_tmp), vcgtq_u16(cur_row, next_row_tmp))));

            int8x8_t idx0   = vadd_s8(vadd_s8(sign_down, const2_cur), sign_up0);
            int8x8_t offset = vtbl2_s8(offset_edge_tbl_uv, idx0);

            int16x8_t sum_s16 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            vst1q_u16(pu2_src_cpy, vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val));
        }
    }

    pu2_src_org[0] = u2_pos_0_0_tmp_u;
    pu2_src_org[1] = u2_pos_0_0_tmp_v;
    pu2_src_org[wd - 2 + (ht - 1) * src_strd] = u2_pos_wd_ht_tmp_u;
    pu2_src_org[wd - 1 + (ht - 1) * src_strd] = u2_pos_wd_ht_tmp_v;
}

/**
*******************************************************************************
*
* @brief
*  SAO edge offset class 3 (45-degree diagonal) filter for 10-bit interleaved
*  Chroma (U, V) blocks using ARM NEON intrinsics.
*
*******************************************************************************
*/
void ihevc_hbd_sao_edge_offset_class3_chroma_neonintr(UWORD16 *pu2_src,
                                                      WORD32 src_strd,
                                                      UWORD16 *pu2_src_left,
                                                      UWORD16 *pu2_src_top,
                                                      UWORD16 *pu2_src_top_left,
                                                      UWORD16 *pu2_src_top_right,
                                                      UWORD16 *pu2_src_bot_left,
                                                      UWORD8 *pu1_avail,
                                                      WORD8 *pi1_sao_offset_u,
                                                      WORD8 *pi1_sao_offset_v,
                                                      WORD32 wd,
                                                      WORD32 ht,
                                                      UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    WORD32 ht_tmp = ht;
    UWORD16 au2_src_top_left_tmp[2];
    UWORD16 u2_pos_wd_0_tmp_u, u2_pos_wd_0_tmp_v;
    UWORD16 u2_pos_0_ht_tmp_u, u2_pos_0_ht_tmp_v;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and corner pixels before any modifications */
    au2_src_top_left_tmp[0] = pu2_src_top[wd - 2];
    au2_src_top_left_tmp[1] = pu2_src_top[wd - 1];

    if(0 != pu1_avail[5])
    {
        WORD32 edge_idx_u = 2 + SIGN(pu2_src[wd - 2] - pu2_src_top_right[0]) +
                                SIGN(pu2_src[wd - 2] - pu2_src[wd - 4 + src_strd]);
        edge_idx_u = gi1_table_edge_idx[edge_idx_u];
        u2_pos_wd_0_tmp_u = (0 != edge_idx_u) ? CLIP3(pu2_src[wd - 2] + pi1_sao_offset_u[edge_idx_u], 0, max_pixel_val)
                                              : pu2_src[wd - 2];

        WORD32 edge_idx_v = 2 + SIGN(pu2_src[wd - 1] - pu2_src_top_right[1]) +
                                SIGN(pu2_src[wd - 1] - pu2_src[wd - 3 + src_strd]);
        edge_idx_v = gi1_table_edge_idx[edge_idx_v];
        u2_pos_wd_0_tmp_v = (0 != edge_idx_v) ? CLIP3(pu2_src[wd - 1] + pi1_sao_offset_v[edge_idx_v], 0, max_pixel_val)
                                              : pu2_src[wd - 1];
    }
    else
    {
        u2_pos_wd_0_tmp_u = pu2_src[wd - 2];
        u2_pos_wd_0_tmp_v = pu2_src[wd - 1];
    }

    if(0 != pu1_avail[6])
    {
        WORD32 bl_u = (ht - 1) * src_strd;
        WORD32 edge_idx_u = 2 + SIGN(pu2_src[bl_u] - pu2_src[bl_u + 2 - src_strd]) +
                                SIGN(pu2_src[bl_u] - pu2_src_bot_left[0]);
        edge_idx_u = gi1_table_edge_idx[edge_idx_u];
        u2_pos_0_ht_tmp_u = (0 != edge_idx_u) ? CLIP3(pu2_src[bl_u] + pi1_sao_offset_u[edge_idx_u], 0, max_pixel_val)
                                              : pu2_src[bl_u];

        WORD32 bl_v = (ht - 1) * src_strd + 1;
        WORD32 edge_idx_v = 2 + SIGN(pu2_src[bl_v] - pu2_src[bl_v + 2 - src_strd]) +
                                SIGN(pu2_src[bl_v] - pu2_src_bot_left[1]);
        edge_idx_v = gi1_table_edge_idx[edge_idx_v];
        u2_pos_0_ht_tmp_v = (0 != edge_idx_v) ? CLIP3(pu2_src[bl_v] + pi1_sao_offset_v[edge_idx_v], 0, max_pixel_val)
                                              : pu2_src[bl_v];
    }
    else
    {
        u2_pos_0_ht_tmp_u = pu2_src[(ht - 1) * src_strd];
        u2_pos_0_ht_tmp_v = pu2_src[(ht - 1) * src_strd + 1];
    }

    pu2_src_top_left[0] = au2_src_top_left_tmp[0];
    pu2_src_top_left[1] = au2_src_top_left_tmp[1];

    /* 2. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src_left[0] = pu2_src[wd - 2];
        pu2_src_left[1] = pu2_src[wd - 1];
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 2;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    int8x8_t edge_idx_tbl         = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u    = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v    = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);
    int8x8x2_t offset_edge_tbl_uv = { { offset_edge_tbl_u, offset_edge_tbl_v } };

    uint16_t u2_avail_left    = (0 == pu1_avail[0]) ? 0x2020 : 0x0A02;
    uint16_t u2_avail_right   = (0 == pu1_avail[1]) ? 0x2020 : 0x0A02;
    int8x16_t const_2_10_q = vreinterpretq_s8_u16(vdupq_n_u16(0x0A02));
    int8x8_t const_2_10    = vget_low_s8(const_2_10_q);
    int16x8_t v_zero_s16   = vdupq_n_s16(0);
    uint16x8_t v_max_val   = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 16; col += 16)
    {
        int8x16_t const2_cur = const_2_10_q;
        if(0 == col)
        {
            const2_cur = vreinterpretq_s8_u16(vsetq_lane_u16(u2_avail_left, vreinterpretq_u16_s8(const2_cur), 0));
        }
        if(col == wd - 16)
        {
            const2_cur = vreinterpretq_s8_u16(vsetq_lane_u16(u2_avail_right, vreinterpretq_u16_s8(const2_cur), 7));
        }

        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_lo    = vld1q_u16(&pu2_src_top_cpy[col + 2]);
        uint16x8_t top_hi    = vld1q_u16(&pu2_src_top_cpy[col + 10]);

        vst1q_u16(&pu2_src_top[col],     vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
        vst1q_u16(&pu2_src_top[col + 8], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col + 8]));

        uint16x8_t cur0_lo = vld1q_u16(pu2_src_cpy);
        uint16x8_t cur0_hi = vld1q_u16(pu2_src_cpy + 8);
        int8x16_t sign_up0 = vreinterpretq_s8_u8(vcombine_u8(
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, top_lo), vcgtq_u16(cur0_lo, top_lo))),
            vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, top_hi), vcgtq_u16(cur0_hi, top_hi)))));

        for(row = 0; row < ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t cur1_lo            = vld1q_u16(pu2_row1);
            uint16x8_t cur1_hi            = vld1q_u16(pu2_row1 + 8);
            UWORD16 u2_dl0_u              = pu2_src_left_cpy[2 * (row + 1)];
            UWORD16 u2_dl0_v              = pu2_src_left_cpy[2 * (row + 1) + 1];
            pu2_src_left_cpy[2 * row]     = pu2_row0[14];
            pu2_src_left_cpy[2 * row + 1] = pu2_row0[15];

            uint16x8_t cur1_lo_s = vsetq_lane_u16(u2_dl0_u, cur1_lo, 6);
            cur1_lo_s            = vsetq_lane_u16(u2_dl0_v, cur1_lo_s, 7);
            cur1_lo_s            = vextq_u16(cur1_lo_s, cur1_lo, 6);
            uint16x8_t cur1_hi_s = vextq_u16(cur1_lo, cur1_hi, 6);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, cur1_lo_s), vcgtq_u16(cur0_lo, cur1_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, cur1_hi_s), vcgtq_u16(cur0_hi, cur1_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_row0,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_row0 + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));

            UWORD16 u2_row1_14  = pu2_row1[14];
            UWORD16 u2_row1_15  = pu2_row1[15];
            int8x16_t neg_down0 = vnegq_s8(sign_down0);
            int8x16_t sign_up1  = vsetq_lane_s8(SIGN(u2_row1_14 - pu2_row0[16]), neg_down0, 0);
            sign_up1            = vsetq_lane_s8(SIGN(u2_row1_15 - pu2_row0[17]), sign_up1, 1);
            sign_up1            = vextq_s8(neg_down0, sign_up1, 2);

            uint16x8_t next_lo                  = vld1q_u16(pu2_row2);
            uint16x8_t next_hi                  = vld1q_u16(pu2_row2 + 8);
            UWORD16 u2_dl1_u                    = pu2_src_left_cpy[2 * (row + 2)];
            UWORD16 u2_dl1_v                    = pu2_src_left_cpy[2 * (row + 2) + 1];
            pu2_src_left_cpy[2 * (row + 1)]     = u2_row1_14;
            pu2_src_left_cpy[2 * (row + 1) + 1] = u2_row1_15;

            uint16x8_t next_lo_s = vsetq_lane_u16(u2_dl1_u, next_lo, 6);
            next_lo_s            = vsetq_lane_u16(u2_dl1_v, next_lo_s, 7);
            next_lo_s            = vextq_u16(next_lo_s, next_lo, 6);
            uint16x8_t next_hi_s = vextq_u16(next_lo, next_hi, 6);

            int8x16_t sign_down1 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_lo, next_lo_s), vcgtq_u16(cur1_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur1_hi, next_hi_s), vcgtq_u16(cur1_hi, next_hi_s)))));

            int8x16_t idx1   = vaddq_s8(vaddq_s8(sign_down1, const2_cur), sign_up1);
            int8x8_t off1_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx1));
            int8x8_t off1_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx1));

            int16x8_t sum1_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_lo), off1_lo), v_zero_s16);
            int16x8_t sum1_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur1_hi), off1_hi), v_zero_s16);

            vst1q_u16(pu2_row1,     vminq_u16(vreinterpretq_u16_s16(sum1_lo), v_max_val));
            vst1q_u16(pu2_row1 + 8, vminq_u16(vreinterpretq_u16_s16(sum1_hi), v_max_val));

            int8x16_t neg_down1 = vnegq_s8(sign_down1);
            sign_up0            = vsetq_lane_s8(SIGN(pu2_row2[14] - pu2_row1[16]), neg_down1, 0);
            sign_up0            = vsetq_lane_s8(SIGN(pu2_row2[15] - pu2_row1[17]), sign_up0, 1);
            sign_up0            = vextq_s8(neg_down1, sign_up0, 2);

            cur0_lo     = next_lo;
            cur0_hi     = next_hi;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            UWORD16 *pu2_next             = pu2_src_cpy + src_strd;
            uint16x8_t next_lo            = vld1q_u16(pu2_next);
            uint16x8_t next_hi            = vld1q_u16(pu2_next + 8);
            UWORD16 u2_dl_u               = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                            ? pu2_next[-2]
                                            : pu2_src_left_cpy[2 * (row + 1)];
            UWORD16 u2_dl_v               = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                            ? pu2_next[-1]
                                            : pu2_src_left_cpy[2 * (row + 1) + 1];
            pu2_src_left_cpy[2 * row]     = pu2_src_cpy[14];
            pu2_src_left_cpy[2 * row + 1] = pu2_src_cpy[15];

            uint16x8_t next_lo_s = vsetq_lane_u16(u2_dl_u, next_lo, 6);
            next_lo_s            = vsetq_lane_u16(u2_dl_v, next_lo_s, 7);
            next_lo_s            = vextq_u16(next_lo_s, next_lo, 6);
            uint16x8_t next_hi_s = vextq_u16(next_lo, next_hi, 6);

            int8x16_t sign_down0 = vreinterpretq_s8_u8(vcombine_u8(
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_lo, next_lo_s), vcgtq_u16(cur0_lo, next_lo_s))),
                vmovn_u16(vsubq_u16(vcltq_u16(cur0_hi, next_hi_s), vcgtq_u16(cur0_hi, next_hi_s)))));

            int8x16_t idx0   = vaddq_s8(vaddq_s8(sign_down0, const2_cur), sign_up0);
            int8x8_t off0_lo = vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(idx0));
            int8x8_t off0_hi = vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(idx0));

            int16x8_t sum0_lo = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_lo), off0_lo), v_zero_s16);
            int16x8_t sum0_hi = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur0_hi), off0_hi), v_zero_s16);

            vst1q_u16(pu2_src_cpy,     vminq_u16(vreinterpretq_u16_s16(sum0_lo), v_max_val));
            vst1q_u16(pu2_src_cpy + 8, vminq_u16(vreinterpretq_u16_s16(sum0_hi), v_max_val));

            int8x16_t neg_down0 = vnegq_s8(sign_down0);
            sign_up0            = vsetq_lane_s8(SIGN(pu2_next[14] - pu2_src_cpy[16]), neg_down0, 0);
            sign_up0            = vsetq_lane_s8(SIGN(pu2_next[15] - pu2_src_cpy[17]), sign_up0, 1);
            sign_up0            = vextq_s8(neg_down0, sign_up0, 2);

            cur0_lo     = next_lo;
            cur0_hi     = next_hi;
            pu2_src_cpy = pu2_next;
        }

        if(0 == pu1_avail[3])
        {
            pu2_src_left[2 * (ht - 1)]     = pu2_src_cpy[14];
            pu2_src_left[2 * (ht - 1) + 1] = pu2_src_cpy[15];
        }
    }

    if(col < wd)
    {
        int8x8_t const2_cur = const_2_10;
        if(0 == col)
        {
            const2_cur = vreinterpret_s8_u16(vset_lane_u16(u2_avail_left, vreinterpret_u16_s8(const2_cur), 0));
        }
        const2_cur = vreinterpret_s8_u16(vset_lane_u16(u2_avail_right, vreinterpret_u16_s8(const2_cur), 3));

        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col + 2]);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(
            vsubq_u16(vcltq_u16(cur_row, top_row), vcgtq_u16(cur_row, top_row))));

        for(row = 0; row < ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1          = vld1q_u16(pu2_row1);
            UWORD16 u2_dl0_u              = pu2_src_left_cpy[2 * (row + 1)];
            UWORD16 u2_dl0_v              = pu2_src_left_cpy[2 * (row + 1) + 1];
            pu2_src_left_cpy[2 * row]     = pu2_row0[6];
            pu2_src_left_cpy[2 * row + 1] = pu2_row0[7];
            uint16x8_t next_row1_tmp      = vsetq_lane_u16(u2_dl0_u, next_row1, 6);
            next_row1_tmp                 = vsetq_lane_u16(u2_dl0_v, next_row1_tmp, 7);
            next_row1_tmp                 = vextq_u16(next_row1_tmp, next_row1, 6);

            uint16x8_t next_row2                = vld1q_u16(pu2_row2);
            UWORD16 u2_dl1_u                    = pu2_src_left_cpy[2 * (row + 2)];
            UWORD16 u2_dl1_v                    = pu2_src_left_cpy[2 * (row + 2) + 1];
            UWORD16 u2_row1_6                   = pu2_row1[6];
            UWORD16 u2_row1_7                   = pu2_row1[7];
            pu2_src_left_cpy[2 * (row + 1)]     = u2_row1_6;
            pu2_src_left_cpy[2 * (row + 1) + 1] = u2_row1_7;
            uint16x8_t next_row2_tmp            = vsetq_lane_u16(u2_dl1_u, next_row2, 6);
            next_row2_tmp                       = vsetq_lane_u16(u2_dl1_v, next_row2_tmp, 7);
            next_row2_tmp                       = vextq_u16(next_row2_tmp, next_row2, 6);

            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row1_tmp), vcgtq_u16(cur_row, next_row1_tmp))));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(next_row1, next_row2_tmp), vcgtq_u16(next_row1, next_row2_tmp))));

            int8x8_t idx0 = vadd_s8(vadd_s8(sign_down0, const2_cur), sign_up0);

            int8x8_t neg_down0 = vneg_s8(sign_down0);
            int8x8_t sign_up1  = vset_lane_s8(SIGN(u2_row1_6 - pu2_row0[8]), neg_down0, 0);
            sign_up1           = vset_lane_s8(SIGN(u2_row1_7 - pu2_row0[9]), sign_up1, 1);
            sign_up1           = vext_s8(neg_down0, sign_up1, 2);
            int8x8_t idx1      = vadd_s8(vadd_s8(sign_down1, const2_cur), sign_up1);

            int8x8_t neg_down1 = vneg_s8(sign_down1);
            sign_up0           = vset_lane_s8(SIGN(pu2_row2[6] - pu2_row1[8]), neg_down1, 0);
            sign_up0           = vset_lane_s8(SIGN(pu2_row2[7] - pu2_row1[9]), sign_up0, 1);
            sign_up0           = vext_s8(neg_down1, sign_up0, 2);

            int8x8_t off0 = vtbl2_s8(offset_edge_tbl_uv, idx0);
            int8x8_t off1 = vtbl2_s8(offset_edge_tbl_uv, idx1);

            int16x8_t sum0 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), off0), v_zero_s16);
            int16x8_t sum1 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), off1), v_zero_s16);

            vst1q_u16(pu2_row0, vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val));
            vst1q_u16(pu2_row1, vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val));

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            UWORD16 *pu2_next             = pu2_src_cpy + src_strd;
            uint16x8_t next_row           = vld1q_u16(pu2_next);
            UWORD16 u2_dl_u               = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                            ? pu2_next[-2]
                                            : pu2_src_left_cpy[2 * (row + 1)];
            UWORD16 u2_dl_v               = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                            ? pu2_next[-1]
                                            : pu2_src_left_cpy[2 * (row + 1) + 1];
            pu2_src_left_cpy[2 * row]     = pu2_src_cpy[6];
            pu2_src_left_cpy[2 * row + 1] = pu2_src_cpy[7];
            uint16x8_t next_row_tmp       = vsetq_lane_u16(u2_dl_u, next_row, 6);
            next_row_tmp                  = vsetq_lane_u16(u2_dl_v, next_row_tmp, 7);
            next_row_tmp                  = vextq_u16(next_row_tmp, next_row, 6);

            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(
                vsubq_u16(vcltq_u16(cur_row, next_row_tmp), vcgtq_u16(cur_row, next_row_tmp))));

            int8x8_t idx0   = vadd_s8(vadd_s8(sign_down, const2_cur), sign_up0);
            int8x8_t offset = vtbl2_s8(offset_edge_tbl_uv, idx0);

            int16x8_t sum_s16 = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            vst1q_u16(pu2_src_cpy, vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val));

            int8x8_t neg_down = vneg_s8(sign_down);
            sign_up0          = vset_lane_s8(SIGN(pu2_next[6] - pu2_src_cpy[8]), neg_down, 0);
            sign_up0          = vset_lane_s8(SIGN(pu2_next[7] - pu2_src_cpy[9]), sign_up0, 1);
            sign_up0          = vext_s8(neg_down, sign_up0, 2);

            cur_row     = next_row;
            pu2_src_cpy = pu2_next;
        }

        if(0 == pu1_avail[3])
        {
            pu2_src_left[2 * (ht - 1)]     = pu2_src_cpy[6];
            pu2_src_left[2 * (ht - 1) + 1] = pu2_src_cpy[7];
        }
    }

    pu2_src_org[wd - 2] = u2_pos_wd_0_tmp_u;
    pu2_src_org[wd - 1] = u2_pos_wd_0_tmp_v;
    pu2_src_org[(ht - 1) * src_strd]     = u2_pos_0_ht_tmp_u;
    pu2_src_org[(ht - 1) * src_strd + 1] = u2_pos_0_ht_tmp_v;
}
