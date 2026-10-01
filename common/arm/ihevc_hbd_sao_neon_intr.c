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

    int8x8_t offset_tbl   = vld1_s8(pi1_sao_offset);
    uint8x16_t v_band_pos = vdupq_n_u8((uint8_t)(sao_band_pos - 1));
    uint8x16_t v_mask_31  = vdupq_n_u8(31);
    int16x8_t v_zero_s16 = vdupq_n_s16(0);
    uint16x8_t v_max_val = vdupq_n_u16((uint16_t)max_pixel_val);

    for(row = 0; row < ht; row +=2)
    {
        UWORD16 *pu2_row0 = pu2_src;
        UWORD16 *pu2_row1 = pu2_src + src_strd;

        for(col = 0; col <= wd - 8; col += 8)
        {
            uint16x8_t src0_u16 = vld1q_u16(&pu2_row0[col]);
            uint16x8_t src1_u16 = vld1q_u16(&pu2_row1[col]);

            uint8x16_t idx_u8   = vcombine_u8(vshrn_n_u16(src0_u16, 5), vshrn_n_u16(src1_u16, 5));

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

            uint8x16_t idx_u8   = vcombine_u8(vshrn_n_u16(src0_u16, 5), vshrn_n_u16(src1_u16, 5));

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
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Update top-left and top boundary buffers from unmodified pu2_src */
    *pu2_src_top_left = pu2_src_top[wd - 1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src[(ht - 1) * src_strd + col]));
    }

    int8x8_t offset_edge_tbl = vtbl1_s8(vld1_s8(pi1_sao_offset), vld1_s8(gi1_table_edge_idx));

    int8_t i1_avail_left  = (0 == pu1_avail[0]) ? 0 : -1;
    int8_t i1_avail_right = (0 == pu1_avail[1]) ? 0 : -1;
    int8x16_t const_2   = vdupq_n_s8(2);
    int16x8_t v_zero_s16  = vdupq_n_s16(0);
    uint16x8_t v_max_val  = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 8; col += 8)
    {
        int8x8_t cur_mask = vdup_n_s8(-1);
        if(0 == col)
        {
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 0);
        }
        if(col == wd - 8)
        {
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 7);
        }
        int8x16_t cur_mask_q = vcombine_s8(cur_mask, cur_mask);

        UWORD16 *pu2_src_cpy = pu2_src + col;

        for(row = 0; row < ht; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;

            uint16x8_t cur_row0   = vld1q_u16(pu2_row0);
            uint16x8_t right_row0 = vld1q_u16(pu2_row0 + 1);
            uint16x8_t cur_row1   = vld1q_u16(pu2_row1);
            uint16x8_t right_row1 = vld1q_u16(pu2_row1 + 1);

            uint16x8_t left_row0  = vsetq_lane_u16(pu2_src_left[row], cur_row0, 7);
            left_row0             = vextq_u16(left_row0, cur_row0, 7);
            uint16x8_t left_row1  = vsetq_lane_u16(pu2_src_left[row + 1], cur_row1, 7);
            left_row1             = vextq_u16(left_row1, cur_row1, 7);

            /* Save unmodified 8th pixel (P7) for the next 8-column strip */
            pu2_src_left[row]     = pu2_row0[7];
            pu2_src_left[row + 1] = pu2_row1[7];

            uint16x8_t diff_l0 = vsubq_u16(vcltq_u16(cur_row0, left_row0), vcgtq_u16(cur_row0, left_row0));
            uint16x8_t diff_l1 = vsubq_u16(vcltq_u16(cur_row1, left_row1), vcgtq_u16(cur_row1, left_row1));
            int8x16_t sign_left = vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(diff_l0), vmovn_u16(diff_l1)));


            uint16x8_t diff_r0 = vsubq_u16(vcltq_u16(cur_row0, right_row0), vcgtq_u16(cur_row0, right_row0));
            uint16x8_t diff_r1 = vsubq_u16(vcltq_u16(cur_row1, right_row1), vcgtq_u16(cur_row1, right_row1));
            int8x16_t sign_right = vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(diff_r0), vmovn_u16(diff_r1)));


            int8x16_t raw_idx = vaddq_s8(vaddq_s8(const_2, sign_left), sign_right);
            int8x16_t offset  = vandq_s8(vcombine_s8(vtbl1_s8(offset_edge_tbl, vget_low_s8(raw_idx)),
                                                     vtbl1_s8(offset_edge_tbl, vget_high_s8(raw_idx))),
                                         cur_mask_q);


            int16x8_t sum0  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), vget_low_s8(offset)), v_zero_s16);
            int16x8_t sum1  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row1), vget_high_s8(offset)), v_zero_s16);


            uint16x8_t res0 = vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val);
            uint16x8_t res1 = vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val);


            vst1q_u16(pu2_row0, res0);
            vst1q_u16(pu2_row1, res1);

            pu2_src_cpy += 2 * src_strd;

        }
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

    /* 1. Update top-left and left boundary buffers from unmodified pu2_src */
    *pu2_src_top_left = pu2_src_top[wd - 1];
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

    int8x8_t offset_edge_tbl = vtbl1_s8(vld1_s8(pi1_sao_offset), vld1_s8(gi1_table_edge_idx));

    int8x16_t const_2  = vdupq_n_s8(2);
    int16x8_t v_zero_s16 = vdupq_n_s16(0);
    uint16x8_t v_max_val = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 8; col += 8)
    {
        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col]);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row0 = vld1q_u16(pu2_src_cpy);
        uint16x8_t gt_u     = vcgtq_u16(cur_row0, top_row);
        uint16x8_t lt_u     = vcltq_u16(cur_row0, top_row);
        int8x8_t sign_up0   = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_u, gt_u)));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0   = pu2_src_cpy;
            UWORD16 *pu2_row1   = pu2_src_cpy + src_strd;
            uint16x8_t cur_row1 = vld1q_u16(pu2_row1);
            uint16x8_t next_row = vld1q_u16(pu2_row1 + src_strd);

            uint16x8_t diff_d0  = vsubq_u16(vcltq_u16(cur_row0, cur_row1), vcgtq_u16(cur_row0, cur_row1));
            uint16x8_t diff_d1  = vsubq_u16(vcltq_u16(cur_row1, next_row), vcgtq_u16(cur_row1, next_row));
            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(diff_d0));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(diff_d1));

            int8x16_t sign_up_q   = vcombine_s8(sign_up0, vneg_s8(sign_down0));
            int8x16_t sign_down_q = vcombine_s8(sign_down0, sign_down1);

            int8x16_t raw_idx = vaddq_s8(vaddq_s8(const_2, sign_up_q), sign_down_q);
            int8x16_t offset  = vcombine_s8(vtbl1_s8(offset_edge_tbl, vget_low_s8(raw_idx)),
                                            vtbl1_s8(offset_edge_tbl, vget_high_s8(raw_idx)));

            sign_up0 = vneg_s8(sign_down1);

            int16x8_t sum0  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), vget_low_s8(offset)), v_zero_s16);
            int16x8_t sum1  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row1), vget_high_s8(offset)), v_zero_s16);

            uint16x8_t res0 = vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val);
            uint16x8_t res1 = vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val);

            vst1q_u16(pu2_row0, res0);
            vst1q_u16(pu2_row1, res1);

            cur_row0 = next_row;
            pu2_src_cpy += 2 * src_strd;
        }
        for(; row < ht_tmp; row++)
        {
            uint16x8_t next_row = vld1q_u16(pu2_src_cpy + src_strd);

            uint16x8_t gt_d    = vcgtq_u16(cur_row0, next_row);
            uint16x8_t lt_d    = vcltq_u16(cur_row0, next_row);
            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d, gt_d)));

            int8x8_t raw_idx = vadd_s8(vadd_s8(vget_low_s8(const_2), sign_up0), sign_down);
            int8x8_t offset  = vtbl1_s8(offset_edge_tbl, raw_idx);

            sign_up0 = vneg_s8(sign_down);

            int16x8_t sum_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), offset), v_zero_s16);
            uint16x8_t res_u16 = vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val);

            vst1q_u16(pu2_src_cpy, res_u16);

            cur_row0 = next_row;
            pu2_src_cpy += src_strd;
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
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE];
    UWORD16 au2_src_top_tmp[MAX_CTB_SIZE];
    UWORD16 u2_src_top_left_tmp;
    UWORD16 u2_pos_0_0_tmp;
    UWORD16 u2_pos_wd_ht_tmp;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and unmodified bottom row for pu2_src_top */
    u2_src_top_left_tmp = pu2_src_top[wd - 1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&au2_src_top_tmp[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
    }

    if(0 != pu1_avail[4])
    {
        WORD32 edge_idx = 2 + SIGN(pu2_src[0] - pu2_src_top_left[0]) +
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

    /* 3. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 1;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    int8x8_t offset_edge_tbl = vtbl1_s8(vld1_s8(pi1_sao_offset), vld1_s8(gi1_table_edge_idx));

    int8_t i1_avail_left  = (0 == pu1_avail[0]) ? 0 : -1;
    int8_t i1_avail_right = (0 == pu1_avail[1]) ? 0 : -1;
    int8x8_t const_2      = vdup_n_s8(2);
    int8x16_t const_2_q   = vdupq_n_s8(2);
    int16x8_t v_zero_s16  = vdupq_n_s16(0);
    uint16x8_t v_max_val  = vdupq_n_u16((uint16_t)max_pixel_val);
    UWORD16 u2_top_left   = pu2_src_top_left[0];

    for(col = 0; col <= wd - 8; col += 8)
    {
        for(row = 0; row < ht; row++)
        {
            au2_src_left_tmp[row] = pu2_src_org[row * src_strd + col + 7];
        }

        int8x8_t cur_mask = vdup_n_s8(-1);
        if(0 == col)
        {
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 0);
        }
        if(col == wd - 8)
        {
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 7);
        }
        int8x16_t cur_mask_q = vcombine_s8(cur_mask, cur_mask);

        UWORD16 *pu2_src_cpy   = pu2_src + col;
        uint16x8_t top_row_raw = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_row     = vsetq_lane_u16(u2_top_left, top_row_raw, 7);
        top_row                = vextq_u16(top_row, top_row_raw, 7);
        u2_top_left            = pu2_src_top_cpy[col + 7];

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        uint16x8_t gt_u    = vcgtq_u16(cur_row, top_row);
        uint16x8_t lt_u    = vcltq_u16(cur_row, top_row);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_u, gt_u)));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1     = vld1q_u16(pu2_row1);
            uint16x8_t next_row1_tmp = vld1q_u16(pu2_row1 + 1);
            uint16x8_t next_row2     = vld1q_u16(pu2_row2);
            uint16x8_t next_row2_tmp = vld1q_u16(pu2_row2 + 1);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_row0[0] - pu2_src_left_cpy[row - 1]), sign_up0, 0);
            }

            uint16x8_t gt_d0    = vcgtq_u16(cur_row, next_row1_tmp);
            uint16x8_t lt_d0    = vcltq_u16(cur_row, next_row1_tmp);
            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d0, gt_d0)));

            uint16x8_t gt_d1    = vcgtq_u16(next_row1, next_row2_tmp);
            uint16x8_t lt_d1    = vcltq_u16(next_row1, next_row2_tmp);
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d1, gt_d1)));

            int8x8_t sign_up1 = vneg_s8(sign_down0);
            sign_up1          = vext_s8(sign_up1, sign_up1, 7);
            sign_up1          = vset_lane_s8(SIGN(pu2_row1[0] - pu2_src_left_cpy[row]), sign_up1, 0);

            int8x16_t sign_up_q   = vcombine_s8(sign_up0, sign_up1);
            int8x16_t sign_down_q = vcombine_s8(sign_down0, sign_down1);

            int8x16_t raw_idx_q = vaddq_s8(vaddq_s8(const_2_q, sign_up_q), sign_down_q);
            int8x16_t offset_q  = vandq_s8(vcombine_s8(vtbl1_s8(offset_edge_tbl, vget_low_s8(raw_idx_q)),
                                                       vtbl1_s8(offset_edge_tbl, vget_high_s8(raw_idx_q))),
                                           cur_mask_q);

            sign_up0 = vneg_s8(sign_down1);
            sign_up0 = vext_s8(sign_up0, sign_up0, 7);

            int16x8_t sum0_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), vget_low_s8(offset_q)), v_zero_s16);
            int16x8_t sum1_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), vget_high_s8(offset_q)), v_zero_s16);

            uint16x8_t res0_u16 = vminq_u16(vreinterpretq_u16_s16(sum0_s16), v_max_val);
            uint16x8_t res1_u16 = vminq_u16(vreinterpretq_u16_s16(sum1_s16), v_max_val);

            vst1q_u16(pu2_row0, res0_u16);
            vst1q_u16(pu2_row1, res1_u16);

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            uint16x8_t next_row     = vld1q_u16(pu2_src_cpy + src_strd);
            uint16x8_t next_row_tmp = vld1q_u16(pu2_src_cpy + src_strd + 1);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_src_cpy[0] - pu2_src_left_cpy[row - 1]), sign_up0, 0);
            }

            uint16x8_t gt_d    = vcgtq_u16(cur_row, next_row_tmp);
            uint16x8_t lt_d    = vcltq_u16(cur_row, next_row_tmp);
            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d, gt_d)));

            int8x8_t raw_idx = vadd_s8(vadd_s8(const_2, sign_up0), sign_down);
            int8x8_t offset  = vand_s8(vtbl1_s8(offset_edge_tbl, raw_idx), cur_mask);

            sign_up0 = vneg_s8(sign_down);
            sign_up0 = vext_s8(sign_up0, sign_up0, 7);

            int16x8_t sum_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            uint16x8_t res_u16 = vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val);

            vst1q_u16(pu2_src_cpy, res_u16);

            cur_row = next_row;
            pu2_src_cpy += src_strd;
        }

        for(row = 0; row < ht; row++)
        {
            pu2_src_left[row] = au2_src_left_tmp[row];
        }
    }

    pu2_src_org[0] = u2_pos_0_0_tmp;
    pu2_src_org[wd - 1 + (ht - 1) * src_strd] = u2_pos_wd_ht_tmp;

    *pu2_src_top_left = u2_src_top_left_tmp;
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&au2_src_top_tmp[col]));
    }
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
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE];
    UWORD16 au2_src_top_tmp[MAX_CTB_SIZE];
    UWORD16 u2_src_top_left_tmp;
    UWORD16 u2_pos_wd_0_tmp;
    UWORD16 u2_pos_0_ht_tmp;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and unmodified bottom row for pu2_src_top */
    u2_src_top_left_tmp = pu2_src_top[wd - 1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&au2_src_top_tmp[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
    }

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

    /* 3. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 1;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    int8x8_t offset_edge_tbl = vtbl1_s8(vld1_s8(pi1_sao_offset), vld1_s8(gi1_table_edge_idx));

    int8_t i1_avail_left  = (0 == pu1_avail[0]) ? 0 : -1;
    int8_t i1_avail_right = (0 == pu1_avail[1]) ? 0 : -1;
    int8x8_t const_2      = vdup_n_s8(2);
    int8x16_t const_2_q   = vdupq_n_s8(2);
    int16x8_t v_zero_s16  = vdupq_n_s16(0);
    uint16x8_t v_max_val  = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 8; col += 8)
    {
        for(row = 0; row < ht; row++)
        {
            au2_src_left_tmp[row] = pu2_src_org[row * src_strd + col + 7];
        }

        int8x8_t cur_mask = vdup_n_s8(-1);
        if(0 == col)
        {
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 0);
        }
        if(col == wd - 8)
        {
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 7);
        }
        int8x16_t cur_mask_q = vcombine_s8(cur_mask, cur_mask);

        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col + 1]);

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        uint16x8_t gt_u    = vcgtq_u16(cur_row, top_row);
        uint16x8_t lt_u    = vcltq_u16(cur_row, top_row);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_u, gt_u)));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1     = vld1q_u16(pu2_row1);
            uint16x8_t next_row1_tmp = vsetq_lane_u16(pu2_src_left_cpy[row + 1], next_row1, 7);
            next_row1_tmp            = vextq_u16(next_row1_tmp, next_row1, 7);

            uint16x8_t next_row2    = vld1q_u16(pu2_row2);
            UWORD16 u2_down_left1   = (0 != pu1_avail[3] && (row + 1) == ht_tmp - 1)
                                      ? pu2_row2[-1]
                                      : pu2_src_left_cpy[row + 2];
            uint16x8_t next_row2_tmp = vsetq_lane_u16(u2_down_left1, next_row2, 7);
            next_row2_tmp            = vextq_u16(next_row2_tmp, next_row2, 7);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_row0[7] - pu2_row0[8 - src_strd]), sign_up0, 7);
            }

            uint16x8_t gt_d0    = vcgtq_u16(cur_row, next_row1_tmp);
            uint16x8_t lt_d0    = vcltq_u16(cur_row, next_row1_tmp);
            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d0, gt_d0)));

            uint16x8_t gt_d1    = vcgtq_u16(next_row1, next_row2_tmp);
            uint16x8_t lt_d1    = vcltq_u16(next_row1, next_row2_tmp);
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d1, gt_d1)));

            int8x8_t sign_up1 = vneg_s8(sign_down0);
            sign_up1          = vext_s8(sign_up1, sign_up1, 1);
            sign_up1          = vset_lane_s8(SIGN(pu2_row1[7] - pu2_row0[8]), sign_up1, 7);

            int8x16_t sign_up_q   = vcombine_s8(sign_up0, sign_up1);
            int8x16_t sign_down_q = vcombine_s8(sign_down0, sign_down1);

            int8x16_t raw_idx_q = vaddq_s8(vaddq_s8(const_2_q, sign_up_q), sign_down_q);
            int8x16_t offset_q  = vandq_s8(vcombine_s8(vtbl1_s8(offset_edge_tbl, vget_low_s8(raw_idx_q)),
                                                       vtbl1_s8(offset_edge_tbl, vget_high_s8(raw_idx_q))),
                                           cur_mask_q);

            sign_up0 = vneg_s8(sign_down1);
            sign_up0 = vext_s8(sign_up0, sign_up0, 1);

            int16x8_t sum0_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), vget_low_s8(offset_q)), v_zero_s16);
            int16x8_t sum1_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), vget_high_s8(offset_q)), v_zero_s16);

            uint16x8_t res0_u16 = vminq_u16(vreinterpretq_u16_s16(sum0_s16), v_max_val);
            uint16x8_t res1_u16 = vminq_u16(vreinterpretq_u16_s16(sum1_s16), v_max_val);

            vst1q_u16(pu2_row0, res0_u16);
            vst1q_u16(pu2_row1, res1_u16);

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            uint16x8_t next_row = vld1q_u16(pu2_src_cpy + src_strd);
            UWORD16 u2_down_left = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                   ? pu2_src_cpy[src_strd - 1]
                                   : pu2_src_left_cpy[row + 1];
            uint16x8_t next_row_tmp = vsetq_lane_u16(u2_down_left, next_row, 7);
            next_row_tmp            = vextq_u16(next_row_tmp, next_row, 7);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_src_cpy[7] - pu2_src_cpy[8 - src_strd]), sign_up0, 7);
            }

            uint16x8_t gt_d    = vcgtq_u16(cur_row, next_row_tmp);
            uint16x8_t lt_d    = vcltq_u16(cur_row, next_row_tmp);
            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d, gt_d)));

            int8x8_t raw_idx = vadd_s8(vadd_s8(const_2, sign_up0), sign_down);
            int8x8_t offset  = vand_s8(vtbl1_s8(offset_edge_tbl, raw_idx), cur_mask);

            sign_up0 = vneg_s8(sign_down);
            sign_up0 = vext_s8(sign_up0, sign_up0, 1);

            int16x8_t sum_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            uint16x8_t res_u16 = vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val);

            vst1q_u16(pu2_src_cpy, res_u16);

            cur_row = next_row;
            pu2_src_cpy += src_strd;
        }

        for(row = 0; row < ht; row++)
        {
            pu2_src_left[row] = au2_src_left_tmp[row];
        }
    }

    pu2_src_org[wd - 1] = u2_pos_wd_0_tmp;
    pu2_src_org[(ht - 1) * src_strd] = u2_pos_0_ht_tmp;

    *pu2_src_top_left = u2_src_top_left_tmp;
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&au2_src_top_tmp[col]));
    }
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
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Update top-left and top boundary buffers from unmodified pu2_src */
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src[(ht - 1) * src_strd + col]));
    }

    int8x8_t edge_idx_tbl      = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);

    int8_t i1_avail_left  = (0 == pu1_avail[0]) ? 0 : -1;
    int8_t i1_avail_right = (0 == pu1_avail[1]) ? 0 : -1;
    uint8x8_t v_odd_lanes = vcreate_u8(0xFF00FF00FF00FF00ULL);
    int8x8_t const_2      = vdup_n_s8(2);
    int16x8_t v_zero_s16  = vdupq_n_s16(0);
    uint16x8_t v_max_val  = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 8; col += 8)
    {
        int8x8_t cur_mask = vdup_n_s8(-1);
        if(0 == col)
        {
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 0);
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 1);
        }
        if(col == wd - 8)
        {
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 6);
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 7);
        }

        UWORD16 *pu2_src_cpy = pu2_src + col;

        for(row = 0; row < ht; row++)
        {
            uint16x8_t cur_row   = vld1q_u16(pu2_src_cpy);
            uint16x8_t right_row = vld1q_u16(pu2_src_cpy + 2);

            uint16x8_t left_row  = vsetq_lane_u16(pu2_src_left[2 * row], cur_row, 6);
            left_row             = vsetq_lane_u16(pu2_src_left[2 * row + 1], left_row, 7);
            left_row             = vextq_u16(left_row, cur_row, 6);

            pu2_src_left[2 * row]     = pu2_src_cpy[6];
            pu2_src_left[2 * row + 1] = pu2_src_cpy[7];

            uint16x8_t gt_l = vcgtq_u16(cur_row, left_row);
            uint16x8_t lt_l = vcltq_u16(cur_row, left_row);
            int8x8_t sign_left = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_l, gt_l)));

            uint16x8_t gt_r = vcgtq_u16(cur_row, right_row);
            uint16x8_t lt_r = vcltq_u16(cur_row, right_row);
            int8x8_t sign_right = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_r, gt_r)));

            int8x8_t raw_idx = vadd_s8(vadd_s8(const_2, sign_left), sign_right);

            int8x8_t off_u  = vtbl1_s8(offset_edge_tbl_u, raw_idx);
            int8x8_t off_v  = vtbl1_s8(offset_edge_tbl_v, raw_idx);
            int8x8_t offset = vand_s8(vbsl_s8(v_odd_lanes, off_v, off_u), cur_mask);

            int16x8_t sum_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            uint16x8_t res_u16 = vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val);

            vst1q_u16(pu2_src_cpy, res_u16);

            pu2_src_cpy += src_strd;
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

    /* 1. Update top-left and left boundary buffers from unmodified pu2_src */
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];
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

    int8x8_t edge_idx_tbl        = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u   = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v   = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);
    int8x8x2_t offset_edge_tbl_uv = { { offset_edge_tbl_u, offset_edge_tbl_v } };

    int8x16_t const_2_10_q = vreinterpretq_s8_u16(vdupq_n_u16(0x0A02));
    int16x8_t v_zero_s16   = vdupq_n_s16(0);
    uint16x8_t v_max_val   = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 8; col += 8)
    {
        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col]);

        vst1q_u16(&pu2_src_top[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));

        uint16x8_t cur_row0 = vld1q_u16(pu2_src_cpy);
        uint16x8_t gt_u     = vcgtq_u16(cur_row0, top_row);
        uint16x8_t lt_u     = vcltq_u16(cur_row0, top_row);
        int8x8_t sign_up0   = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_u, gt_u)));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0   = pu2_src_cpy;
            UWORD16 *pu2_row1   = pu2_src_cpy + src_strd;
            uint16x8_t cur_row1 = vld1q_u16(pu2_row1);
            uint16x8_t next_row = vld1q_u16(pu2_row1 + src_strd);

            uint16x8_t diff_d0  = vsubq_u16(vcltq_u16(cur_row0, cur_row1), vcgtq_u16(cur_row0, cur_row1));
            uint16x8_t diff_d1  = vsubq_u16(vcltq_u16(cur_row1, next_row), vcgtq_u16(cur_row1, next_row));
            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(diff_d0));
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(diff_d1));

            int8x16_t sign_up_q   = vcombine_s8(sign_up0, vneg_s8(sign_down0));
            int8x16_t sign_down_q = vcombine_s8(sign_down0, sign_down1);

            int8x16_t raw_idx = vaddq_s8(vaddq_s8(const_2_10_q, sign_up_q), sign_down_q);
            int8x16_t offset  = vcombine_s8(vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(raw_idx)),
                                            vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(raw_idx)));

            sign_up0 = vneg_s8(sign_down1);

            int16x8_t sum0  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), vget_low_s8(offset)), v_zero_s16);
            int16x8_t sum1  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row1), vget_high_s8(offset)), v_zero_s16);

            uint16x8_t res0 = vminq_u16(vreinterpretq_u16_s16(sum0), v_max_val);
            uint16x8_t res1 = vminq_u16(vreinterpretq_u16_s16(sum1), v_max_val);

            vst1q_u16(pu2_row0, res0);
            vst1q_u16(pu2_row1, res1);

            cur_row0 = next_row;
            pu2_src_cpy += 2 * src_strd;
        }
        for(; row < ht_tmp; row++)
        {
            uint16x8_t next_row = vld1q_u16(pu2_src_cpy + src_strd);

            uint16x8_t gt_d    = vcgtq_u16(cur_row0, next_row);
            uint16x8_t lt_d    = vcltq_u16(cur_row0, next_row);
            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d, gt_d)));

            int8x8_t raw_idx = vadd_s8(vadd_s8(vget_low_s8(const_2_10_q), sign_up0), sign_down);
            int8x8_t offset  = vtbl2_s8(offset_edge_tbl_uv, raw_idx);

            sign_up0 = vneg_s8(sign_down);

            int16x8_t sum_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row0), offset), v_zero_s16);
            uint16x8_t res_u16 = vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val);

            vst1q_u16(pu2_src_cpy, res_u16);

            cur_row0 = next_row;
            pu2_src_cpy += src_strd;
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
    UWORD16 au2_src_left_tmp[2 * MAX_CTB_SIZE];
    UWORD16 au2_src_top_tmp[2 * MAX_CTB_SIZE];
    UWORD16 au2_src_top_left_tmp[2];
    UWORD16 u2_pos_0_0_tmp_u, u2_pos_0_0_tmp_v;
    UWORD16 u2_pos_wd_ht_tmp_u, u2_pos_wd_ht_tmp_v;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    (void)pu2_src_top_right;
    (void)pu2_src_bot_left;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and unmodified bottom row for pu2_src_top */
    au2_src_top_left_tmp[0] = pu2_src_top[wd - 2];
    au2_src_top_left_tmp[1] = pu2_src_top[wd - 1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&au2_src_top_tmp[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
    }

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

    /* 3. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 2;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    int8x8_t edge_idx_tbl          = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u     = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v     = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);
    int8x8x2_t offset_edge_tbl_uv  = { { offset_edge_tbl_u, offset_edge_tbl_v } };

    int8_t i1_avail_left   = (0 == pu1_avail[0]) ? 0 : -1;
    int8_t i1_avail_right  = (0 == pu1_avail[1]) ? 0 : -1;
    int8x16_t const_2_10_q = vreinterpretq_s8_u16(vdupq_n_u16(0x0A02));
    int8x8_t const_2_10    = vget_low_s8(const_2_10_q);
    int16x8_t v_zero_s16   = vdupq_n_s16(0);
    uint16x8_t v_max_val   = vdupq_n_u16((uint16_t)max_pixel_val);
    UWORD16 u2_top_left_u  = pu2_src_top_left[0];
    UWORD16 u2_top_left_v  = pu2_src_top_left[1];

    for(col = 0; col <= wd - 8; col += 8)
    {
        for(row = 0; row < ht; row++)
        {
            au2_src_left_tmp[2 * row]     = pu2_src_org[row * src_strd + col + 6];
            au2_src_left_tmp[2 * row + 1] = pu2_src_org[row * src_strd + col + 7];
        }

        int8x8_t cur_mask = vdup_n_s8(-1);
        if(0 == col)
        {
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 0);
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 1);
        }
        if(col == wd - 8)
        {
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 6);
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 7);
        }
        int8x16_t cur_mask_q = vcombine_s8(cur_mask, cur_mask);

        UWORD16 *pu2_src_cpy   = pu2_src + col;
        uint16x8_t top_row_raw = vld1q_u16(&pu2_src_top_cpy[col]);
        uint16x8_t top_row     = vsetq_lane_u16(u2_top_left_u, top_row_raw, 6);
        top_row                = vsetq_lane_u16(u2_top_left_v, top_row, 7);
        top_row                = vextq_u16(top_row, top_row_raw, 6);
        u2_top_left_u          = pu2_src_top_cpy[col + 6];
        u2_top_left_v          = pu2_src_top_cpy[col + 7];

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        uint16x8_t gt_u    = vcgtq_u16(cur_row, top_row);
        uint16x8_t lt_u    = vcltq_u16(cur_row, top_row);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_u, gt_u)));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1     = vld1q_u16(pu2_row1);
            uint16x8_t next_row1_tmp = vld1q_u16(pu2_row1 + 2);
            uint16x8_t next_row2     = vld1q_u16(pu2_row2);
            uint16x8_t next_row2_tmp = vld1q_u16(pu2_row2 + 2);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_row0[0] - pu2_src_left_cpy[(row - 1) * 2]), sign_up0, 0);
                sign_up0 = vset_lane_s8(SIGN(pu2_row0[1] - pu2_src_left_cpy[(row - 1) * 2 + 1]), sign_up0, 1);
            }

            uint16x8_t gt_d0    = vcgtq_u16(cur_row, next_row1_tmp);
            uint16x8_t lt_d0    = vcltq_u16(cur_row, next_row1_tmp);
            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d0, gt_d0)));

            uint16x8_t gt_d1    = vcgtq_u16(next_row1, next_row2_tmp);
            uint16x8_t lt_d1    = vcltq_u16(next_row1, next_row2_tmp);
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d1, gt_d1)));

            int8x8_t sign_up1 = vneg_s8(sign_down0);
            sign_up1          = vext_s8(sign_up1, sign_up1, 6);
            sign_up1          = vset_lane_s8(SIGN(pu2_row1[0] - pu2_src_left_cpy[row * 2]), sign_up1, 0);
            sign_up1          = vset_lane_s8(SIGN(pu2_row1[1] - pu2_src_left_cpy[row * 2 + 1]), sign_up1, 1);

            int8x16_t sign_up_q   = vcombine_s8(sign_up0, sign_up1);
            int8x16_t sign_down_q = vcombine_s8(sign_down0, sign_down1);

            int8x16_t raw_idx_q = vaddq_s8(vaddq_s8(const_2_10_q, sign_up_q), sign_down_q);
            int8x16_t tbl_res   = vcombine_s8(vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(raw_idx_q)),
                                              vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(raw_idx_q)));
            int8x16_t offset_q  = vandq_s8(tbl_res, cur_mask_q);

            sign_up0 = vneg_s8(sign_down1);
            sign_up0 = vext_s8(sign_up0, sign_up0, 6);

            int16x8_t sum0_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), vget_low_s8(offset_q)), v_zero_s16);
            int16x8_t sum1_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), vget_high_s8(offset_q)), v_zero_s16);

            uint16x8_t res0_u16 = vminq_u16(vreinterpretq_u16_s16(sum0_s16), v_max_val);
            uint16x8_t res1_u16 = vminq_u16(vreinterpretq_u16_s16(sum1_s16), v_max_val);

            vst1q_u16(pu2_row0, res0_u16);
            vst1q_u16(pu2_row1, res1_u16);

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            uint16x8_t next_row     = vld1q_u16(pu2_src_cpy + src_strd);
            uint16x8_t next_row_tmp = vld1q_u16(pu2_src_cpy + src_strd + 2);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_src_cpy[0] - pu2_src_left_cpy[(row - 1) * 2]), sign_up0, 0);
                sign_up0 = vset_lane_s8(SIGN(pu2_src_cpy[1] - pu2_src_left_cpy[(row - 1) * 2 + 1]), sign_up0, 1);
            }

            uint16x8_t gt_d    = vcgtq_u16(cur_row, next_row_tmp);
            uint16x8_t lt_d    = vcltq_u16(cur_row, next_row_tmp);
            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d, gt_d)));

            int8x8_t raw_idx = vadd_s8(vadd_s8(const_2_10, sign_up0), sign_down);
            int8x8_t offset  = vand_s8(vtbl2_s8(offset_edge_tbl_uv, raw_idx), cur_mask);

            sign_up0 = vneg_s8(sign_down);
            sign_up0 = vext_s8(sign_up0, sign_up0, 6);

            int16x8_t sum_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            uint16x8_t res_u16 = vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val);

            vst1q_u16(pu2_src_cpy, res_u16);

            cur_row = next_row;
            pu2_src_cpy += src_strd;
        }

        for(row = 0; row < 2 * ht; row++)
        {
            pu2_src_left[row] = au2_src_left_tmp[row];
        }
    }

    pu2_src_org[0] = u2_pos_0_0_tmp_u;
    pu2_src_org[1] = u2_pos_0_0_tmp_v;
    pu2_src_org[wd - 2 + (ht - 1) * src_strd] = u2_pos_wd_ht_tmp_u;
    pu2_src_org[wd - 1 + (ht - 1) * src_strd] = u2_pos_wd_ht_tmp_v;

    pu2_src_top_left[0] = au2_src_top_left_tmp[0];
    pu2_src_top_left[1] = au2_src_top_left_tmp[1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&au2_src_top_tmp[col]));
    }
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
    UWORD16 au2_src_left_tmp[2 * MAX_CTB_SIZE];
    UWORD16 au2_src_top_tmp[2 * MAX_CTB_SIZE];
    UWORD16 au2_src_top_left_tmp[2];
    UWORD16 u2_pos_wd_0_tmp_u, u2_pos_wd_0_tmp_v;
    UWORD16 u2_pos_0_ht_tmp_u, u2_pos_0_ht_tmp_v;
    UWORD16 *pu2_src_org = pu2_src;
    UWORD16 *pu2_src_left_cpy = pu2_src_left;
    UWORD16 *pu2_src_top_cpy = pu2_src_top;
    const WORD32 max_pixel_val = (1 << u4_bit_depth) - 1;

    /* 1. Save top-left and unmodified bottom row for pu2_src_top */
    au2_src_top_left_tmp[0] = pu2_src_top[wd - 2];
    au2_src_top_left_tmp[1] = pu2_src_top[wd - 1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&au2_src_top_tmp[col], vld1q_u16(&pu2_src_org[(ht - 1) * src_strd + col]));
    }

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

    /* 3. Adjust starting row and height based on top/bottom availability */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht_tmp--;
        pu2_src_left_cpy += 2;
    }
    if(0 == pu1_avail[3])
    {
        ht_tmp--;
    }

    int8x8_t edge_idx_tbl          = vld1_s8(gi1_table_edge_idx);
    int8x8_t offset_edge_tbl_u     = vtbl1_s8(vld1_s8(pi1_sao_offset_u), edge_idx_tbl);
    int8x8_t offset_edge_tbl_v     = vtbl1_s8(vld1_s8(pi1_sao_offset_v), edge_idx_tbl);
    int8x8x2_t offset_edge_tbl_uv  = { { offset_edge_tbl_u, offset_edge_tbl_v } };

    int8_t i1_avail_left   = (0 == pu1_avail[0]) ? 0 : -1;
    int8_t i1_avail_right  = (0 == pu1_avail[1]) ? 0 : -1;
    int8x16_t const_2_10_q = vreinterpretq_s8_u16(vdupq_n_u16(0x0A02));
    int8x8_t const_2_10    = vget_low_s8(const_2_10_q);
    int16x8_t v_zero_s16   = vdupq_n_s16(0);
    uint16x8_t v_max_val   = vdupq_n_u16((uint16_t)max_pixel_val);

    for(col = 0; col <= wd - 8; col += 8)
    {
        for(row = 0; row < ht; row++)
        {
            au2_src_left_tmp[2 * row]     = pu2_src_org[row * src_strd + col + 6];
            au2_src_left_tmp[2 * row + 1] = pu2_src_org[row * src_strd + col + 7];
        }

        int8x8_t cur_mask = vdup_n_s8(-1);
        if(0 == col)
        {
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 0);
            cur_mask = vset_lane_s8(i1_avail_left, cur_mask, 1);
        }
        if(col == wd - 8)
        {
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 6);
            cur_mask = vset_lane_s8(i1_avail_right, cur_mask, 7);
        }
        int8x16_t cur_mask_q = vcombine_s8(cur_mask, cur_mask);

        UWORD16 *pu2_src_cpy = pu2_src + col;
        uint16x8_t top_row   = vld1q_u16(&pu2_src_top_cpy[col + 2]);

        uint16x8_t cur_row = vld1q_u16(pu2_src_cpy);
        uint16x8_t gt_u    = vcgtq_u16(cur_row, top_row);
        uint16x8_t lt_u    = vcltq_u16(cur_row, top_row);
        int8x8_t sign_up0  = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_u, gt_u)));

        for(row = 0; row <= ht_tmp - 2; row += 2)
        {
            UWORD16 *pu2_row0 = pu2_src_cpy;
            UWORD16 *pu2_row1 = pu2_src_cpy + src_strd;
            UWORD16 *pu2_row2 = pu2_row1 + src_strd;

            uint16x8_t next_row1     = vld1q_u16(pu2_row1);
            uint16x8_t next_row1_tmp = vsetq_lane_u16(pu2_src_left_cpy[(row + 1) * 2], next_row1, 6);
            next_row1_tmp            = vsetq_lane_u16(pu2_src_left_cpy[(row + 1) * 2 + 1], next_row1_tmp, 7);
            next_row1_tmp            = vextq_u16(next_row1_tmp, next_row1, 6);

            uint16x8_t next_row2    = vld1q_u16(pu2_row2);
            UWORD16 u2_down_left1_u = (0 != pu1_avail[3] && (row + 1) == ht_tmp - 1)
                                      ? pu2_row2[-2]
                                      : pu2_src_left_cpy[(row + 2) * 2];
            UWORD16 u2_down_left1_v = (0 != pu1_avail[3] && (row + 1) == ht_tmp - 1)
                                      ? pu2_row2[-1]
                                      : pu2_src_left_cpy[(row + 2) * 2 + 1];
            uint16x8_t next_row2_tmp = vsetq_lane_u16(u2_down_left1_u, next_row2, 6);
            next_row2_tmp            = vsetq_lane_u16(u2_down_left1_v, next_row2_tmp, 7);
            next_row2_tmp            = vextq_u16(next_row2_tmp, next_row2, 6);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_row0[6] - pu2_row0[8 - src_strd]), sign_up0, 6);
                sign_up0 = vset_lane_s8(SIGN(pu2_row0[7] - pu2_row0[9 - src_strd]), sign_up0, 7);
            }

            uint16x8_t gt_d0    = vcgtq_u16(cur_row, next_row1_tmp);
            uint16x8_t lt_d0    = vcltq_u16(cur_row, next_row1_tmp);
            int8x8_t sign_down0 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d0, gt_d0)));

            uint16x8_t gt_d1    = vcgtq_u16(next_row1, next_row2_tmp);
            uint16x8_t lt_d1    = vcltq_u16(next_row1, next_row2_tmp);
            int8x8_t sign_down1 = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d1, gt_d1)));

            int8x8_t sign_up1 = vneg_s8(sign_down0);
            sign_up1          = vext_s8(sign_up1, sign_up1, 2);
            sign_up1          = vset_lane_s8(SIGN(pu2_row1[6] - pu2_row0[8]), sign_up1, 6);
            sign_up1          = vset_lane_s8(SIGN(pu2_row1[7] - pu2_row0[9]), sign_up1, 7);

            int8x16_t sign_up_q   = vcombine_s8(sign_up0, sign_up1);
            int8x16_t sign_down_q = vcombine_s8(sign_down0, sign_down1);

            int8x16_t raw_idx_q = vaddq_s8(vaddq_s8(const_2_10_q, sign_up_q), sign_down_q);
            int8x16_t tbl_res   = vcombine_s8(vtbl2_s8(offset_edge_tbl_uv, vget_low_s8(raw_idx_q)),
                                              vtbl2_s8(offset_edge_tbl_uv, vget_high_s8(raw_idx_q)));
            int8x16_t offset_q  = vandq_s8(tbl_res, cur_mask_q);

            sign_up0 = vneg_s8(sign_down1);
            sign_up0 = vext_s8(sign_up0, sign_up0, 2);

            int16x8_t sum0_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), vget_low_s8(offset_q)), v_zero_s16);
            int16x8_t sum1_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(next_row1), vget_high_s8(offset_q)), v_zero_s16);

            uint16x8_t res0_u16 = vminq_u16(vreinterpretq_u16_s16(sum0_s16), v_max_val);
            uint16x8_t res1_u16 = vminq_u16(vreinterpretq_u16_s16(sum1_s16), v_max_val);

            vst1q_u16(pu2_row0, res0_u16);
            vst1q_u16(pu2_row1, res1_u16);

            cur_row     = next_row2;
            pu2_src_cpy = pu2_row2;
        }

        for(; row < ht_tmp; row++)
        {
            uint16x8_t next_row = vld1q_u16(pu2_src_cpy + src_strd);
            UWORD16 u2_down_left_u = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                     ? pu2_src_cpy[src_strd - 2]
                                     : pu2_src_left_cpy[(row + 1) * 2];
            UWORD16 u2_down_left_v = (0 != pu1_avail[3] && row == ht_tmp - 1)
                                     ? pu2_src_cpy[src_strd - 1]
                                     : pu2_src_left_cpy[(row + 1) * 2 + 1];
            uint16x8_t next_row_tmp = vsetq_lane_u16(u2_down_left_u, next_row, 6);
            next_row_tmp            = vsetq_lane_u16(u2_down_left_v, next_row_tmp, 7);
            next_row_tmp            = vextq_u16(next_row_tmp, next_row, 6);

            if(row > 0 || 0 == pu1_avail[2])
            {
                sign_up0 = vset_lane_s8(SIGN(pu2_src_cpy[6] - pu2_src_cpy[8 - src_strd]), sign_up0, 6);
                sign_up0 = vset_lane_s8(SIGN(pu2_src_cpy[7] - pu2_src_cpy[9 - src_strd]), sign_up0, 7);
            }

            uint16x8_t gt_d    = vcgtq_u16(cur_row, next_row_tmp);
            uint16x8_t lt_d    = vcltq_u16(cur_row, next_row_tmp);
            int8x8_t sign_down = vreinterpret_s8_u8(vmovn_u16(vsubq_u16(lt_d, gt_d)));

            int8x8_t raw_idx = vadd_s8(vadd_s8(const_2_10, sign_up0), sign_down);
            int8x8_t offset  = vand_s8(vtbl2_s8(offset_edge_tbl_uv, raw_idx), cur_mask);

            sign_up0 = vneg_s8(sign_down);
            sign_up0 = vext_s8(sign_up0, sign_up0, 2);

            int16x8_t sum_s16  = vmaxq_s16(vaddw_s8(vreinterpretq_s16_u16(cur_row), offset), v_zero_s16);
            uint16x8_t res_u16 = vminq_u16(vreinterpretq_u16_s16(sum_s16), v_max_val);

            vst1q_u16(pu2_src_cpy, res_u16);

            cur_row = next_row;
            pu2_src_cpy += src_strd;
        }

        for(row = 0; row < 2 * ht; row++)
        {
            pu2_src_left[row] = au2_src_left_tmp[row];
        }
    }

    pu2_src_org[wd - 2] = u2_pos_wd_0_tmp_u;
    pu2_src_org[wd - 1] = u2_pos_wd_0_tmp_v;
    pu2_src_org[(ht - 1) * src_strd]     = u2_pos_0_ht_tmp_u;
    pu2_src_org[(ht - 1) * src_strd + 1] = u2_pos_0_ht_tmp_v;

    pu2_src_top_left[0] = au2_src_top_left_tmp[0];
    pu2_src_top_left[1] = au2_src_top_left_tmp[1];
    for(col = 0; col <= wd - 8; col += 8)
    {
        vst1q_u16(&pu2_src_top[col], vld1q_u16(&au2_src_top_tmp[col]));
    }
}