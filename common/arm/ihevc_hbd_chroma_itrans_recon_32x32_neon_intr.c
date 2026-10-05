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
*  ihevc_hbd_chroma_itrans_recon_32x32_neon_intr.c
*
* @brief
*  Contains function definitions for chroma inverse transform and reconstruction 32x32
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_chroma_itrans_recon_32x32_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_itrans_recon.h"
#include "ihevc_trans_macros.h"
#include <arm_neon.h>

void ihevc_hbd_chroma_itrans_recon_32x32_neonintr(WORD16 *pi2_src,
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
    /* Inverse Transform */
    /* Fast path: If either all rows or all columns are zero, copy pred to dst */
    if (((UWORD32)zero_cols == 0xFFFFFFFFU) || ((UWORD32)zero_rows == 0xFFFFFFFFU))
    {
        for (WORD32 i = 0; i < 32; i++)
        {
            uint16x8x2_t p0 = vld2q_u16(pu2_pred + 0);
            uint16x8x2_t d0 = vld2q_u16(pu2_dst + 0);
            d0.val[0] = p0.val[0];
            vst2q_u16(pu2_dst + 0, d0);

            uint16x8x2_t p1 = vld2q_u16(pu2_pred + 16);
            uint16x8x2_t d1 = vld2q_u16(pu2_dst + 16);
            d1.val[0] = p1.val[0];
            vst2q_u16(pu2_dst + 16, d1);

            uint16x8x2_t p2 = vld2q_u16(pu2_pred + 32);
            uint16x8x2_t d2 = vld2q_u16(pu2_dst + 32);
            d2.val[0] = p2.val[0];
            vst2q_u16(pu2_dst + 32, d2);

            uint16x8x2_t p3 = vld2q_u16(pu2_pred + 48);
            uint16x8x2_t d3 = vld2q_u16(pu2_dst + 48);
            d3.val[0] = p3.val[0];
            vst2q_u16(pu2_dst + 48, d3);

            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
        return;
    }

    int16x4_t ce0 = vld1_s16(&g_ai2_ihevc_trans_16[4][0]);
    int16x4_t ce1 = vld1_s16(&g_ai2_ihevc_trans_16[2][0]);
    int16x4_t co0 = vld1_s16(&g_ai2_ihevc_trans_16[1][0]);
    int16x4_t co1 = vld1_s16(&g_ai2_ihevc_trans_16[1][4]);
    int16x4_t c_o32_0 = vld1_s16(&g_ai2_ihevc_trans_32[1][0]);
    int16x4_t c_o32_1 = vld1_s16(&g_ai2_ihevc_trans_32[1][4]);
    int16x4_t c_o32_2 = vld1_s16(&g_ai2_ihevc_trans_32[1][8]);
    int16x4_t c_o32_3 = vld1_s16(&g_ai2_ihevc_trans_32[1][12]);

    /* Last 8 cols of 16x16 block are skipped based on the below flag */
    WORD32 zero_last24_cols_stg1 = ((zero_cols & 0xFFFFFF00) == 0xFFFFFF00) ? 1 : 0;
    WORD32 zero_last24_rows_stg1 = ((zero_rows & 0xFFFFFF00) == 0xFFFFFF00) ? 1 : 0;
    WORD32 zero_last28_rows_stg1 = ((zero_rows & 0xFFFFFFF0) == 0xFFFFFFF0) ? 1 : 0;

    WORD32 zero_last28_rows_stg2 = ((zero_cols & 0xFFFFFFF0) == 0xFFFFFFF0) ? 1 : 0;
    WORD32 zero_last24_rows_stg2 = zero_last24_cols_stg1;

    WORD32 trans_size_stg1 = ((zero_last28_rows_stg2) || (zero_last24_cols_stg1)) ? 8 : 32;
    WORD32 trans_size = 32;

    /* Inverse Transform 1st Stage */
    for (WORD32 i = 0; i < trans_size_stg1; i += 8)
    {
        int32x4_t e[2][16];
        int16x4_t y0[2][8], y1[2][8], y2[2][8], y3[2][8];
        int16x8_t m_temp_reg_70, m_temp_reg_71, m_temp_reg_72, m_temp_reg_73;
        int16x8_t m_temp_reg_74, m_temp_reg_75, m_temp_reg_76, m_temp_reg_77;
        int16x8_t m_temp_reg_80, m_temp_reg_81, m_temp_reg_82, m_temp_reg_83;
        int16x8_t m_temp_reg_84, m_temp_reg_85, m_temp_reg_86, m_temp_reg_87;

        /* Load even rows before even branching */
        {
            WORD16* pi2_tmp_src = pi2_src + i;

            m_temp_reg_70 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_71 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_72 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_73 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_74 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_75 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_76 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_77 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);

            m_temp_reg_80 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_81 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_82 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_83 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_84 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_85 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_86 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_87 = vld1q_s16(pi2_tmp_src);
        }

        if (zero_last28_rows_stg1)
        {
            /* eeo */
            /* eeeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
            /* eeeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
            {
                int16x4_t r0_low = vget_low_s16(m_temp_reg_70);
                int16x4_t r2_low = vget_low_s16(m_temp_reg_71);
                int16x4_t r0_high = vget_high_s16(m_temp_reg_70);
                int16x4_t r2_high = vget_high_s16(m_temp_reg_71);

                int32x4_t ee_low = vshll_n_s16(r0_low, 6);
                int32x4_t ee_high = vshll_n_s16(r0_high, 6);

                /* eo */
                /* eo0[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 0);

                    e[0][0] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][15] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo0[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 0);

                    e[1][0] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][15] = vsubq_s32(ee_high, m_temp_reg_30);
                }

                /* eo1[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 1);

                    e[0][1] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][14] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo1[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 1);

                    e[1][1] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][14] = vsubq_s32(ee_high, m_temp_reg_30);
                }

                /* eo2[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 2);

                    e[0][2] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][13] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo2[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 2);

                    e[1][2] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][13] = vsubq_s32(ee_high, m_temp_reg_30);
                }

                /**************************************************************************/

                /* eo3[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 3);

                    e[0][3] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][12] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo3[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 3);

                    e[1][3] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][12] = vsubq_s32(ee_high, m_temp_reg_30);
                }

                /* eo4[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 0);

                    e[0][4] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][11] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo4[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 0);

                    e[1][4] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][11] = vsubq_s32(ee_high, m_temp_reg_30);
                }

                /***********************************************************************/

                /* eo5[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 1);

                    e[0][5] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][10] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo5[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 1);

                    e[1][5] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][10] = vsubq_s32(ee_high, m_temp_reg_30);
                }

                /* eo6[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 2);

                    e[0][6] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][9] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo6[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 2);

                    e[1][6] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][9] = vsubq_s32(ee_high, m_temp_reg_30);
                }

                /* eo7[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 3);

                    e[0][7] = vaddq_s32(ee_low, m_temp_reg_30);
                    e[0][8] = vsubq_s32(ee_low, m_temp_reg_30);
                }

                /* eo7[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 3);

                    e[1][7] = vaddq_s32(ee_high, m_temp_reg_30);
                    e[1][8] = vsubq_s32(ee_high, m_temp_reg_30);
                }
            }
        }
        else if (zero_last24_rows_stg1)
        {
            /* eeo */
            /* eeeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
            /* eeeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
            {
                int16x4_t r0_low = vget_low_s16(m_temp_reg_70);
                int16x4_t r2_low = vget_low_s16(m_temp_reg_71);
                int16x4_t r4_low = vget_low_s16(m_temp_reg_72);
                int16x4_t r6_low = vget_low_s16(m_temp_reg_73);

                int16x4_t r0_high = vget_high_s16(m_temp_reg_70);
                int16x4_t r2_high = vget_high_s16(m_temp_reg_71);
                int16x4_t r4_high = vget_high_s16(m_temp_reg_72);
                int16x4_t r6_high = vget_high_s16(m_temp_reg_73);

                int32x4_t eee_low = vshll_n_s16(r0_low, 6);
                int32x4_t eee_high = vshll_n_s16(r0_high, 6);

                /* eeo */
                /* eeo0[0-3] */
                int32x4_t ee0_low = vaddq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 0));
                int32x4_t ee7_low = vsubq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 0));
                /* eeo0[4-7] */
                int32x4_t ee0_high = vaddq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 0));
                int32x4_t ee7_high = vsubq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 0));

                /* eeo1[0-3] */
                int32x4_t ee1_low = vaddq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 1));
                int32x4_t ee6_low = vsubq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 1));
                /* eo1[4-7] */
                int32x4_t ee1_high = vaddq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 1));
                int32x4_t ee6_high = vsubq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 1));

                /* eo2[0-3] */
                int32x4_t ee2_low = vaddq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 2));
                int32x4_t ee5_low = vsubq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 2));
                /* eo2[4-7] */
                int32x4_t ee2_high = vaddq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 2));
                int32x4_t ee5_high = vsubq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 2));

                /* eo3[0-3] */
                int32x4_t ee3_low = vaddq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 3));
                int32x4_t ee4_low = vsubq_s32(eee_low, vmull_lane_s16(r4_low, ce1, 3));
                /* eo3[4-7] */
                int32x4_t ee3_high = vaddq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 3));
                int32x4_t ee4_high = vsubq_s32(eee_high, vmull_lane_s16(r4_high, ce1, 3));

                /* eo */
                /* eo0[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlal_lane_s16(vmull_lane_s16(r2_low, co0, 0), r6_low, co0, 1);

                    e[0][0] = vaddq_s32(ee0_low, m_temp_reg_30);
                    e[0][15] = vsubq_s32(ee0_low, m_temp_reg_30);
                }
                /* eo0[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlal_lane_s16(vmull_lane_s16(r2_high, co0, 0), r6_high, co0, 1);

                    e[1][0] = vaddq_s32(ee0_high, m_temp_reg_30);
                    e[1][15] = vsubq_s32(ee0_high, m_temp_reg_30);
                }

                /* eo1[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlal_lane_s16(vmull_lane_s16(r2_low, co0, 1), r6_low, co1, 0);

                    e[0][1] = vaddq_s32(ee1_low, m_temp_reg_30);
                    e[0][14] = vsubq_s32(ee1_low, m_temp_reg_30);
                }
                /* eo1[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlal_lane_s16(vmull_lane_s16(r2_high, co0, 1), r6_high, co1, 0);

                    e[1][1] = vaddq_s32(ee1_high, m_temp_reg_30);
                    e[1][14] = vsubq_s32(ee1_high, m_temp_reg_30);
                }

                /* eo2[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlal_lane_s16(vmull_lane_s16(r2_low, co0, 2), r6_low, co1, 3);

                    e[0][2] = vaddq_s32(ee2_low, m_temp_reg_30);
                    e[0][13] = vsubq_s32(ee2_low, m_temp_reg_30);
                }
                /* eo2[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlal_lane_s16(vmull_lane_s16(r2_high, co0, 2), r6_high, co1, 3);

                    e[1][2] = vaddq_s32(ee2_high, m_temp_reg_30);
                    e[1][13] = vsubq_s32(ee2_high, m_temp_reg_30);
                }

                /**************************************************************************/

                /* eo3[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_low, co0, 3), r6_low, co1, 1);

                    e[0][3] = vaddq_s32(ee3_low, m_temp_reg_30);
                    e[0][12] = vsubq_s32(ee3_low, m_temp_reg_30);
                }
                /* eo3[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_high, co0, 3), r6_high, co1, 1);

                    e[1][3] = vaddq_s32(ee3_high, m_temp_reg_30);
                    e[1][12] = vsubq_s32(ee3_high, m_temp_reg_30);
                }

                /* eo4[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_low, co1, 0), r6_low, co0, 2);

                    e[0][4] = vaddq_s32(ee4_low, m_temp_reg_30);
                    e[0][11] = vsubq_s32(ee4_low, m_temp_reg_30);
                }
                /* eo4[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_high, co1, 0), r6_high, co0, 2);

                    e[1][4] = vaddq_s32(ee4_high, m_temp_reg_30);
                    e[1][11] = vsubq_s32(ee4_high, m_temp_reg_30);
                }

                /***********************************************************************/

                /* eo5[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_low, co1, 1), r6_low, co0, 0);

                    e[0][5] = vaddq_s32(ee5_low, m_temp_reg_30);
                    e[0][10] = vsubq_s32(ee5_low, m_temp_reg_30);
                }
                /* eo5[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_high, co1, 1), r6_high, co0, 0);

                    e[1][5] = vaddq_s32(ee5_high, m_temp_reg_30);
                    e[1][10] = vsubq_s32(ee5_high, m_temp_reg_30);
                }

                /* eo6[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_low, co1, 2), r6_low, co0, 3);

                    e[0][6] = vaddq_s32(ee6_low, m_temp_reg_30);
                    e[0][9] = vsubq_s32(ee6_low, m_temp_reg_30);
                }
                /* eo6[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_high, co1, 2), r6_high, co0, 3);

                    e[1][6] = vaddq_s32(ee6_high, m_temp_reg_30);
                    e[1][9] = vsubq_s32(ee6_high, m_temp_reg_30);
                }

                /* eo7[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_low, co1, 3), r6_low, co1, 2);

                    e[0][7] = vaddq_s32(ee7_low, m_temp_reg_30);
                    e[0][8] = vsubq_s32(ee7_low, m_temp_reg_30);
                }
                /* eo7[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmlsl_lane_s16(vmull_lane_s16(r2_high, co1, 3), r6_high, co1, 2);

                    e[1][7] = vaddq_s32(ee7_high, m_temp_reg_30);
                    e[1][8] = vsubq_s32(ee7_high, m_temp_reg_30);
                }
            }
        }
        else
        {
            /* eeo */
            /* eeeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
            /* eeeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
            {
                int16x4_t r0_low = vget_low_s16(m_temp_reg_70);
                int16x4_t r2_low = vget_low_s16(m_temp_reg_71);
                int16x4_t r4_low = vget_low_s16(m_temp_reg_72);
                int16x4_t r6_low = vget_low_s16(m_temp_reg_73);
                int16x4_t r8_low = vget_low_s16(m_temp_reg_74);
                int16x4_t r10_low = vget_low_s16(m_temp_reg_75);
                int16x4_t r12_low = vget_low_s16(m_temp_reg_76);
                int16x4_t r14_low = vget_low_s16(m_temp_reg_77);
                int16x4_t r16_low = vget_low_s16(m_temp_reg_80);
                int16x4_t r18_low = vget_low_s16(m_temp_reg_81);
                int16x4_t r20_low = vget_low_s16(m_temp_reg_82);
                int16x4_t r22_low = vget_low_s16(m_temp_reg_83);
                int16x4_t r24_low = vget_low_s16(m_temp_reg_84);
                int16x4_t r26_low = vget_low_s16(m_temp_reg_85);
                int16x4_t r28_low = vget_low_s16(m_temp_reg_86);
                int16x4_t r30_low = vget_low_s16(m_temp_reg_87);

                int16x4_t r0_high = vget_high_s16(m_temp_reg_70);
                int16x4_t r2_high = vget_high_s16(m_temp_reg_71);
                int16x4_t r4_high = vget_high_s16(m_temp_reg_72);
                int16x4_t r6_high = vget_high_s16(m_temp_reg_73);
                int16x4_t r8_high = vget_high_s16(m_temp_reg_74);
                int16x4_t r10_high = vget_high_s16(m_temp_reg_75);
                int16x4_t r12_high = vget_high_s16(m_temp_reg_76);
                int16x4_t r14_high = vget_high_s16(m_temp_reg_77);
                int16x4_t r16_high = vget_high_s16(m_temp_reg_80);
                int16x4_t r18_high = vget_high_s16(m_temp_reg_81);
                int16x4_t r20_high = vget_high_s16(m_temp_reg_82);
                int16x4_t r22_high = vget_high_s16(m_temp_reg_83);
                int16x4_t r24_high = vget_high_s16(m_temp_reg_84);
                int16x4_t r26_high = vget_high_s16(m_temp_reg_85);
                int16x4_t r28_high = vget_high_s16(m_temp_reg_86);
                int16x4_t r30_high = vget_high_s16(m_temp_reg_87);

                int32x4_t r0_shl_low = vshll_n_s16(r0_low, 6);
                int32x4_t r16_shl_low = vshll_n_s16(r16_low, 6);
                int32x4_t eeee0_low = vaddq_s32(r0_shl_low, r16_shl_low);
                int32x4_t eeee1_low = vsubq_s32(r0_shl_low, r16_shl_low);
                int32x4_t eeeo0_low = vmlal_lane_s16(vmull_lane_s16(r8_low, ce0, 0), r24_low, ce0, 1);
                int32x4_t eeeo1_low = vmlsl_lane_s16(vmull_lane_s16(r8_low, ce0, 1), r24_low, ce0, 0);
                int32x4_t eee0_low = vaddq_s32(eeee0_low, eeeo0_low);
                int32x4_t eee3_low = vsubq_s32(eeee0_low, eeeo0_low);
                int32x4_t eee1_low = vaddq_s32(eeee1_low, eeeo1_low);
                int32x4_t eee2_low = vsubq_s32(eeee1_low, eeeo1_low);

                int32x4_t r0_shl_high = vshll_n_s16(r0_high, 6);
                int32x4_t r16_shl_high = vshll_n_s16(r16_high, 6);
                int32x4_t eeee0_high = vaddq_s32(r0_shl_high, r16_shl_high);
                int32x4_t eeee1_high = vsubq_s32(r0_shl_high, r16_shl_high);
                int32x4_t eeeo0_high = vmlal_lane_s16(vmull_lane_s16(r8_high, ce0, 0), r24_high, ce0, 1);
                int32x4_t eeeo1_high = vmlsl_lane_s16(vmull_lane_s16(r8_high, ce0, 1), r24_high, ce0, 0);
                int32x4_t eee0_high = vaddq_s32(eeee0_high, eeeo0_high);
                int32x4_t eee3_high = vsubq_s32(eeee0_high, eeeo0_high);
                int32x4_t eee1_high = vaddq_s32(eeee1_high, eeeo1_high);
                int32x4_t eee2_high = vsubq_s32(eeee1_high, eeeo1_high);

                /* eeo */
                /* eeo0[0-3] */
                int32x4_t eeo0_low = vmull_lane_s16(r4_low, ce1, 0);
                eeo0_low = vmlal_lane_s16(eeo0_low, r12_low, ce1, 1);
                eeo0_low = vmlal_lane_s16(eeo0_low, r20_low, ce1, 2);
                eeo0_low = vmlal_lane_s16(eeo0_low, r28_low, ce1, 3);
                int32x4_t ee0_low = vaddq_s32(eee0_low, eeo0_low);
                int32x4_t ee7_low = vsubq_s32(eee0_low, eeo0_low);

                /* eeo0[4-7] */
                int32x4_t eeo0_high = vmull_lane_s16(r4_high, ce1, 0);
                eeo0_high = vmlal_lane_s16(eeo0_high, r12_high, ce1, 1);
                eeo0_high = vmlal_lane_s16(eeo0_high, r20_high, ce1, 2);
                eeo0_high = vmlal_lane_s16(eeo0_high, r28_high, ce1, 3);
                int32x4_t ee0_high = vaddq_s32(eee0_high, eeo0_high);
                int32x4_t ee7_high = vsubq_s32(eee0_high, eeo0_high);

                /* eeo1[0-3] */
                int32x4_t eeo1_low = vmull_lane_s16(r4_low, ce1, 1);
                eeo1_low = vmlsl_lane_s16(eeo1_low, r12_low, ce1, 3);
                eeo1_low = vmlsl_lane_s16(eeo1_low, r20_low, ce1, 0);
                eeo1_low = vmlsl_lane_s16(eeo1_low, r28_low, ce1, 2);
                int32x4_t ee1_low = vaddq_s32(eee1_low, eeo1_low);
                int32x4_t ee6_low = vsubq_s32(eee1_low, eeo1_low);

                /* eeo1[4-7] */
                int32x4_t eeo1_high = vmull_lane_s16(r4_high, ce1, 1);
                eeo1_high = vmlsl_lane_s16(eeo1_high, r12_high, ce1, 3);
                eeo1_high = vmlsl_lane_s16(eeo1_high, r20_high, ce1, 0);
                eeo1_high = vmlsl_lane_s16(eeo1_high, r28_high, ce1, 2);
                int32x4_t ee1_high = vaddq_s32(eee1_high, eeo1_high);
                int32x4_t ee6_high = vsubq_s32(eee1_high, eeo1_high);

                /* eeo2[0-3] */
                int32x4_t eeo2_low = vmull_lane_s16(r4_low, ce1, 2);
                eeo2_low = vmlsl_lane_s16(eeo2_low, r12_low, ce1, 0);
                eeo2_low = vmlal_lane_s16(eeo2_low, r20_low, ce1, 3);
                eeo2_low = vmlal_lane_s16(eeo2_low, r28_low, ce1, 1);
                int32x4_t ee2_low = vaddq_s32(eee2_low, eeo2_low);
                int32x4_t ee5_low = vsubq_s32(eee2_low, eeo2_low);

                /* eeo2[4-7] */
                int32x4_t eeo2_high = vmull_lane_s16(r4_high, ce1, 2);
                eeo2_high = vmlsl_lane_s16(eeo2_high, r12_high, ce1, 0);
                eeo2_high = vmlal_lane_s16(eeo2_high, r20_high, ce1, 3);
                eeo2_high = vmlal_lane_s16(eeo2_high, r28_high, ce1, 1);
                int32x4_t ee2_high = vaddq_s32(eee2_high, eeo2_high);
                int32x4_t ee5_high = vsubq_s32(eee2_high, eeo2_high);

                /* eeo3[0-3] */
                int32x4_t eeo3_low = vmull_lane_s16(r4_low, ce1, 3);
                eeo3_low = vmlsl_lane_s16(eeo3_low, r12_low, ce1, 2);
                eeo3_low = vmlal_lane_s16(eeo3_low, r20_low, ce1, 1);
                eeo3_low = vmlsl_lane_s16(eeo3_low, r28_low, ce1, 0);
                int32x4_t ee3_low = vaddq_s32(eee3_low, eeo3_low);
                int32x4_t ee4_low = vsubq_s32(eee3_low, eeo3_low);

                /* eeo3[4-7] */
                int32x4_t eeo3_high = vmull_lane_s16(r4_high, ce1, 3);
                eeo3_high = vmlsl_lane_s16(eeo3_high, r12_high, ce1, 2);
                eeo3_high = vmlal_lane_s16(eeo3_high, r20_high, ce1, 1);
                eeo3_high = vmlsl_lane_s16(eeo3_high, r28_high, ce1, 0);
                int32x4_t ee3_high = vaddq_s32(eee3_high, eeo3_high);
                int32x4_t ee4_high = vsubq_s32(eee3_high, eeo3_high);

                /* eo */
                /* eo0[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r6_low, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_low, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_low, co0, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_low, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_low, co1, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_low, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_low, co1, 3);

                    e[0][0] = vaddq_s32(ee0_low, m_temp_reg_30);
                    e[0][15] = vsubq_s32(ee0_low, m_temp_reg_30);
                }
                /* eo0[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r6_high, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_high, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_high, co0, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_high, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_high, co1, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_high, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_high, co1, 3);

                    e[1][0] = vaddq_s32(ee0_high, m_temp_reg_30);
                    e[1][15] = vsubq_s32(ee0_high, m_temp_reg_30);
                }

                /* eo1[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r6_low, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_low, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_low, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_low, co0, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r22_low, co0, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r26_low, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_low, co1, 2);

                    e[0][1] = vaddq_s32(ee1_low, m_temp_reg_30);
                    e[0][14] = vsubq_s32(ee1_low, m_temp_reg_30);
                }
                /* eo1[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r6_high, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_high, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_high, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_high, co0, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r22_high, co0, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r26_high, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_high, co1, 2);

                    e[1][1] = vaddq_s32(ee1_high, m_temp_reg_30);
                    e[1][14] = vsubq_s32(ee1_high, m_temp_reg_30);
                }

                /* eo2[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r6_low, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r10_low, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_low, co0, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_low, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_low, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_low, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_low, co1, 1);

                    e[0][2] = vaddq_s32(ee2_low, m_temp_reg_30);
                    e[0][13] = vsubq_s32(ee2_low, m_temp_reg_30);
                }
                /* eo2[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r6_high, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r10_high, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_high, co0, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_high, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_high, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_high, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_high, co1, 1);

                    e[1][2] = vaddq_s32(ee2_high, m_temp_reg_30);
                    e[1][13] = vsubq_s32(ee2_high, m_temp_reg_30);
                }

                /**************************************************************************/

                /* eo3[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_low, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r10_low, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_low, co1, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_low, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_low, co1, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r26_low, co0, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_low, co1, 0);

                    e[0][3] = vaddq_s32(ee3_low, m_temp_reg_30);
                    e[0][12] = vsubq_s32(ee3_low, m_temp_reg_30);
                }
                /* eo3[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_high, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r10_high, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_high, co1, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_high, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_high, co1, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r26_high, co0, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_high, co1, 0);

                    e[1][3] = vaddq_s32(ee3_high, m_temp_reg_30);
                    e[1][12] = vsubq_s32(ee3_high, m_temp_reg_30);
                }

                /* eo4[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_low, co0, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r10_low, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_low, co0, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_low, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r22_low, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_low, co1, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_low, co0, 3);

                    e[0][4] = vaddq_s32(ee4_low, m_temp_reg_30);
                    e[0][11] = vsubq_s32(ee4_low, m_temp_reg_30);
                }
                /* eo4[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_high, co0, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r10_high, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_high, co0, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_high, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r22_high, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_high, co1, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_high, co0, 3);

                    e[1][4] = vaddq_s32(ee4_high, m_temp_reg_30);
                    e[1][11] = vsubq_s32(ee4_high, m_temp_reg_30);
                }

                /***********************************************************************/

                /* eo5[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_low, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_low, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_low, co1, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_low, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_low, co0, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_low, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_low, co0, 2);

                    e[0][5] = vaddq_s32(ee5_low, m_temp_reg_30);
                    e[0][10] = vsubq_s32(ee5_low, m_temp_reg_30);
                }
                /* eo5[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_high, co0, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_high, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r14_high, co1, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r18_high, co0, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_high, co0, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_high, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_high, co0, 2);

                    e[1][5] = vaddq_s32(ee5_high, m_temp_reg_30);
                    e[1][10] = vsubq_s32(ee5_high, m_temp_reg_30);
                }

                /* eo6[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_low, co0, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_low, co0, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_low, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_low, co1, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_low, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r26_low, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_low, co0, 1);

                    e[0][6] = vaddq_s32(ee6_low, m_temp_reg_30);
                    e[0][9] = vsubq_s32(ee6_low, m_temp_reg_30);
                }
                /* eo6[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 2);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_high, co0, 3);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_high, co0, 0);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_high, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_high, co1, 1);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r22_high, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r26_high, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r30_high, co0, 1);

                    e[1][6] = vaddq_s32(ee6_high, m_temp_reg_30);
                    e[1][9] = vsubq_s32(ee6_high, m_temp_reg_30);
                }

                /* eo7[0-3] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_low, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_low, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_low, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_low, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_low, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r22_low, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_low, co0, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_low, co0, 0);

                    e[0][7] = vaddq_s32(ee7_low, m_temp_reg_30);
                    e[0][8] = vsubq_s32(ee7_low, m_temp_reg_30);
                }
                /* eo7[4-7] */
                {
                    int32x4_t m_temp_reg_30 = vmull_lane_s16(r2_high, co1, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r6_high, co1, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r10_high, co1, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r14_high, co1, 0);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r18_high, co0, 3);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r22_high, co0, 2);
                    m_temp_reg_30 = vmlal_lane_s16(m_temp_reg_30, r26_high, co0, 1);
                    m_temp_reg_30 = vmlsl_lane_s16(m_temp_reg_30, r30_high, co0, 0);

                    e[1][7] = vaddq_s32(ee7_high, m_temp_reg_30);
                    e[1][8] = vsubq_s32(ee7_high, m_temp_reg_30);
                }
            }
        }
        /* All e[] are done */
        /****************************/
        {
            WORD16* pi2_tmp_src = pi2_src + i + src_strd;

            m_temp_reg_70 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_71 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_72 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_73 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_74 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_75 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_76 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_77 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);

            m_temp_reg_80 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_81 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_82 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_83 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_84 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_85 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_86 = vld1q_s16(pi2_tmp_src);
            pi2_tmp_src += (src_strd << 1);
            m_temp_reg_87 = vld1q_s16(pi2_tmp_src);
        }

        if (zero_last28_rows_stg1)
        {
            /* o & stage 1 out */
            /* [0-3] */
            {
                int16x4_t r1 = vget_low_s16(m_temp_reg_70);
                int16x4_t r3 = vget_low_s16(m_temp_reg_71);
                /* o0[0-3] */
                int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 0), r3, c_o32_0, 1);
                y0[0][0] = vqrshrn_n_s32(vaddq_s32(e[0][0], o0), 7);
                y3[0][7] = vqrshrn_n_s32(vsubq_s32(e[0][0], o0), 7);

                /* o1[0-3] */
                int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 1), r3, c_o32_1, 0);
                y0[0][1] = vqrshrn_n_s32(vaddq_s32(e[0][1], o1), 7);
                y3[0][6] = vqrshrn_n_s32(vsubq_s32(e[0][1], o1), 7);

                /* o2[0-3] */
                int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 2), r3, c_o32_1, 3);
                y0[0][2] = vqrshrn_n_s32(vaddq_s32(e[0][2], o2), 7);
                y3[0][5] = vqrshrn_n_s32(vsubq_s32(e[0][2], o2), 7);

                /* o3[0-3] */
                int32x4_t o3 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 3), r3, c_o32_2, 2);
                y0[0][3] = vqrshrn_n_s32(vaddq_s32(e[0][3], o3), 7);
                y3[0][4] = vqrshrn_n_s32(vsubq_s32(e[0][3], o3), 7);

                /* o4[0-3] */
                int32x4_t o4 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_1, 0), r3, c_o32_3, 1);
                y0[0][4] = vqrshrn_n_s32(vaddq_s32(e[0][4], o4), 7);
                y3[0][3] = vqrshrn_n_s32(vsubq_s32(e[0][4], o4), 7);

                /* o5[0-3] */
                int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 1), r3, c_o32_3, 3);
                y0[0][5] = vqrshrn_n_s32(vaddq_s32(e[0][5], o5), 7);
                y3[0][2] = vqrshrn_n_s32(vsubq_s32(e[0][5], o5), 7);

                /* o6[0-3] */
                int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 2), r3, c_o32_3, 0);
                y0[0][6] = vqrshrn_n_s32(vaddq_s32(e[0][6], o6), 7);
                y3[0][1] = vqrshrn_n_s32(vsubq_s32(e[0][6], o6), 7);

                /* o7[0-3] */
                int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 3), r3, c_o32_2, 1);
                y0[0][7] = vqrshrn_n_s32(vaddq_s32(e[0][7], o7), 7);
                y3[0][0] = vqrshrn_n_s32(vsubq_s32(e[0][7], o7), 7);

                /* o8[0-3] */
                int32x4_t o8 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 0), r3, c_o32_1, 2);
                y1[0][0] = vqrshrn_n_s32(vaddq_s32(e[0][8], o8), 7);
                y2[0][7] = vqrshrn_n_s32(vsubq_s32(e[0][8], o8), 7);

                /* o9[0-3] */
                int32x4_t o9 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 1), r3, c_o32_0, 3);
                y1[0][1] = vqrshrn_n_s32(vaddq_s32(e[0][9], o9), 7);
                y2[0][6] = vqrshrn_n_s32(vsubq_s32(e[0][9], o9), 7);

                /* o10[0-3] */
                int32x4_t o10 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 2), r3, c_o32_0, 1);
                y1[0][2] = vqrshrn_n_s32(vaddq_s32(e[0][10], o10), 7);
                y2[0][5] = vqrshrn_n_s32(vsubq_s32(e[0][10], o10), 7);

                /* o11[0-3] */
                int32x4_t o11 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 3), r3, c_o32_0, 2);
                y1[0][3] = vqrshrn_n_s32(vaddq_s32(e[0][11], o11), 7);
                y2[0][4] = vqrshrn_n_s32(vsubq_s32(e[0][11], o11), 7);

                /* o12[0-3] */
                int32x4_t o12 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 0), r3, c_o32_1, 1);
                y1[0][4] = vqrshrn_n_s32(vaddq_s32(e[0][12], o12), 7);
                y2[0][3] = vqrshrn_n_s32(vsubq_s32(e[0][12], o12), 7);

                /* o13[0-3] */
                int32x4_t o13 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 1), r3, c_o32_2, 0);
                y1[0][5] = vqrshrn_n_s32(vaddq_s32(e[0][13], o13), 7);
                y2[0][2] = vqrshrn_n_s32(vsubq_s32(e[0][13], o13), 7);

                /* o14[0-3] */
                int32x4_t o14 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 2), r3, c_o32_2, 3);
                y1[0][6] = vqrshrn_n_s32(vaddq_s32(e[0][14], o14), 7);
                y2[0][1] = vqrshrn_n_s32(vsubq_s32(e[0][14], o14), 7);

                /* o15[0-3] */
                int32x4_t o15 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 3), r3, c_o32_3, 2);
                y1[0][7] = vqrshrn_n_s32(vaddq_s32(e[0][15], o15), 7);
                y2[0][0] = vqrshrn_n_s32(vsubq_s32(e[0][15], o15), 7);
            }

            /* [4-7] */
            {
                int16x4_t r1 = vget_high_s16(m_temp_reg_70);
                int16x4_t r3 = vget_high_s16(m_temp_reg_71);
                /* o0[4-7] */
                int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 0), r3, c_o32_0, 1);
                y0[1][0] = vqrshrn_n_s32(vaddq_s32(e[1][0], o0), 7);
                y3[1][7] = vqrshrn_n_s32(vsubq_s32(e[1][0], o0), 7);

                /* o1[4-7] */
                int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 1), r3, c_o32_1, 0);
                y0[1][1] = vqrshrn_n_s32(vaddq_s32(e[1][1], o1), 7);
                y3[1][6] = vqrshrn_n_s32(vsubq_s32(e[1][1], o1), 7);

                /* o2[4-7] */
                int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 2), r3, c_o32_1, 3);
                y0[1][2] = vqrshrn_n_s32(vaddq_s32(e[1][2], o2), 7);
                y3[1][5] = vqrshrn_n_s32(vsubq_s32(e[1][2], o2), 7);

                /* o3[4-7] */
                int32x4_t o3 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 3), r3, c_o32_2, 2);
                y0[1][3] = vqrshrn_n_s32(vaddq_s32(e[1][3], o3), 7);
                y3[1][4] = vqrshrn_n_s32(vsubq_s32(e[1][3], o3), 7);

                /* o4[4-7] */
                int32x4_t o4 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_1, 0), r3, c_o32_3, 1);
                y0[1][4] = vqrshrn_n_s32(vaddq_s32(e[1][4], o4), 7);
                y3[1][3] = vqrshrn_n_s32(vsubq_s32(e[1][4], o4), 7);

                /* o5[4-7] */
                int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 1), r3, c_o32_3, 3);
                y0[1][5] = vqrshrn_n_s32(vaddq_s32(e[1][5], o5), 7);
                y3[1][2] = vqrshrn_n_s32(vsubq_s32(e[1][5], o5), 7);

                /* o6[4-7] */
                int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 2), r3, c_o32_3, 0);
                y0[1][6] = vqrshrn_n_s32(vaddq_s32(e[1][6], o6), 7);
                y3[1][1] = vqrshrn_n_s32(vsubq_s32(e[1][6], o6), 7);

                /* o7[4-7] */
                int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 3), r3, c_o32_2, 1);
                y0[1][7] = vqrshrn_n_s32(vaddq_s32(e[1][7], o7), 7);
                y3[1][0] = vqrshrn_n_s32(vsubq_s32(e[1][7], o7), 7);

                /* o8[4-7] */
                int32x4_t o8 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 0), r3, c_o32_1, 2);
                y1[1][0] = vqrshrn_n_s32(vaddq_s32(e[1][8], o8), 7);
                y2[1][7] = vqrshrn_n_s32(vsubq_s32(e[1][8], o8), 7);

                /* o9[4-7] */
                int32x4_t o9 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 1), r3, c_o32_0, 3);
                y1[1][1] = vqrshrn_n_s32(vaddq_s32(e[1][9], o9), 7);
                y2[1][6] = vqrshrn_n_s32(vsubq_s32(e[1][9], o9), 7);

                /* o10[4-7] */
                int32x4_t o10 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 2), r3, c_o32_0, 1);
                y1[1][2] = vqrshrn_n_s32(vaddq_s32(e[1][10], o10), 7);
                y2[1][5] = vqrshrn_n_s32(vsubq_s32(e[1][10], o10), 7);

                /* o11[4-7] */
                int32x4_t o11 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 3), r3, c_o32_0, 2);
                y1[1][3] = vqrshrn_n_s32(vaddq_s32(e[1][11], o11), 7);
                y2[1][4] = vqrshrn_n_s32(vsubq_s32(e[1][11], o11), 7);

                /* o12[4-7] */
                int32x4_t o12 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 0), r3, c_o32_1, 1);
                y1[1][4] = vqrshrn_n_s32(vaddq_s32(e[1][12], o12), 7);
                y2[1][3] = vqrshrn_n_s32(vsubq_s32(e[1][12], o12), 7);

                /* o13[4-7] */
                int32x4_t o13 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 1), r3, c_o32_2, 0);
                y1[1][5] = vqrshrn_n_s32(vaddq_s32(e[1][13], o13), 7);
                y2[1][2] = vqrshrn_n_s32(vsubq_s32(e[1][13], o13), 7);

                /* o14[4-7] */
                int32x4_t o14 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 2), r3, c_o32_2, 3);
                y1[1][6] = vqrshrn_n_s32(vaddq_s32(e[1][14], o14), 7);
                y2[1][1] = vqrshrn_n_s32(vsubq_s32(e[1][14], o14), 7);

                /* o15[4-7] */
                int32x4_t o15 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 3), r3, c_o32_3, 2);
                y1[1][7] = vqrshrn_n_s32(vaddq_s32(e[1][15], o15), 7);
                y2[1][0] = vqrshrn_n_s32(vsubq_s32(e[1][15], o15), 7);
            }
        }
        else if (zero_last24_rows_stg1)
        {
            /* o & stage 1 out */
            /* [0-3] */
            {
                int16x4_t r1 = vget_low_s16(m_temp_reg_70);
                int16x4_t r3 = vget_low_s16(m_temp_reg_71);
                int16x4_t r5 = vget_low_s16(m_temp_reg_72);
                int16x4_t r7 = vget_low_s16(m_temp_reg_73);
                /* o0[0-3] */
                int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
                o0 = vmlal_lane_s16(o0, r3, c_o32_0, 1);
                o0 = vmlal_lane_s16(o0, r5, c_o32_0, 2);
                o0 = vmlal_lane_s16(o0, r7, c_o32_0, 3);
                y0[0][0] = vqrshrn_n_s32(vaddq_s32(e[0][0], o0), 7);
                y3[0][7] = vqrshrn_n_s32(vsubq_s32(e[0][0], o0), 7);

                /* o1[0-3] */
                int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
                o1 = vmlal_lane_s16(o1, r3, c_o32_1, 0);
                o1 = vmlal_lane_s16(o1, r5, c_o32_1, 3);
                o1 = vmlal_lane_s16(o1, r7, c_o32_2, 2);
                y0[0][1] = vqrshrn_n_s32(vaddq_s32(e[0][1], o1), 7);
                y3[0][6] = vqrshrn_n_s32(vsubq_s32(e[0][1], o1), 7);

                /* o2[0-3] */
                int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
                o2 = vmlal_lane_s16(o2, r3, c_o32_1, 3);
                o2 = vmlal_lane_s16(o2, r5, c_o32_3, 0);
                o2 = vmlsl_lane_s16(o2, r7, c_o32_3, 2);
                y0[0][2] = vqrshrn_n_s32(vaddq_s32(e[0][2], o2), 7);
                y3[0][5] = vqrshrn_n_s32(vsubq_s32(e[0][2], o2), 7);

                /* o3[0-3] */
                int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
                o3 = vmlal_lane_s16(o3, r3, c_o32_2, 2);
                o3 = vmlsl_lane_s16(o3, r5, c_o32_3, 2);
                o3 = vmlsl_lane_s16(o3, r7, c_o32_1, 3);
                y0[0][3] = vqrshrn_n_s32(vaddq_s32(e[0][3], o3), 7);
                y3[0][4] = vqrshrn_n_s32(vsubq_s32(e[0][3], o3), 7);

                /* o4[0-3] */
                int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
                o4 = vmlal_lane_s16(o4, r3, c_o32_3, 1);
                o4 = vmlsl_lane_s16(o4, r5, c_o32_2, 1);
                o4 = vmlsl_lane_s16(o4, r7, c_o32_0, 0);
                y0[0][4] = vqrshrn_n_s32(vaddq_s32(e[0][4], o4), 7);
                y3[0][3] = vqrshrn_n_s32(vsubq_s32(e[0][4], o4), 7);

                /* o5[0-3] */
                int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
                o5 = vmlsl_lane_s16(o5, r3, c_o32_3, 3);
                o5 = vmlsl_lane_s16(o5, r5, c_o32_1, 0);
                o5 = vmlsl_lane_s16(o5, r7, c_o32_1, 2);
                y0[0][5] = vqrshrn_n_s32(vaddq_s32(e[0][5], o5), 7);
                y3[0][2] = vqrshrn_n_s32(vsubq_s32(e[0][5], o5), 7);

                /* o6[0-3] */
                int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
                o6 = vmlsl_lane_s16(o6, r3, c_o32_3, 0);
                o6 = vmlsl_lane_s16(o6, r5, c_o32_0, 0);
                o6 = vmlsl_lane_s16(o6, r7, c_o32_3, 1);
                y0[0][6] = vqrshrn_n_s32(vaddq_s32(e[0][6], o6), 7);
                y3[0][1] = vqrshrn_n_s32(vsubq_s32(e[0][6], o6), 7);

                /* o7[0-3] */
                int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
                o7 = vmlsl_lane_s16(o7, r3, c_o32_2, 1);
                o7 = vmlsl_lane_s16(o7, r5, c_o32_1, 1);
                o7 = vmlal_lane_s16(o7, r7, c_o32_2, 3);
                y0[0][7] = vqrshrn_n_s32(vaddq_s32(e[0][7], o7), 7);
                y3[0][0] = vqrshrn_n_s32(vsubq_s32(e[0][7], o7), 7);

                /* o8[0-3] */
                int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
                o8 = vmlsl_lane_s16(o8, r3, c_o32_1, 2);
                o8 = vmlsl_lane_s16(o8, r5, c_o32_2, 2);
                o8 = vmlal_lane_s16(o8, r7, c_o32_1, 0);
                y1[0][0] = vqrshrn_n_s32(vaddq_s32(e[0][8], o8), 7);
                y2[0][7] = vqrshrn_n_s32(vsubq_s32(e[0][8], o8), 7);

                /* o9[0-3] */
                int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
                o9 = vmlsl_lane_s16(o9, r3, c_o32_0, 3);
                o9 = vmlsl_lane_s16(o9, r5, c_o32_3, 3);
                o9 = vmlal_lane_s16(o9, r7, c_o32_0, 2);
                y1[0][1] = vqrshrn_n_s32(vaddq_s32(e[0][9], o9), 7);
                y2[0][6] = vqrshrn_n_s32(vsubq_s32(e[0][9], o9), 7);

                /* o10[0-3] */
                int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
                o10 = vmlsl_lane_s16(o10, r3, c_o32_0, 1);
                o10 = vmlal_lane_s16(o10, r5, c_o32_2, 3);
                o10 = vmlal_lane_s16(o10, r7, c_o32_2, 1);
                y1[0][2] = vqrshrn_n_s32(vaddq_s32(e[0][10], o10), 7);
                y2[0][5] = vqrshrn_n_s32(vsubq_s32(e[0][10], o10), 7);

                /* o11[0-3] */
                int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
                o11 = vmlsl_lane_s16(o11, r3, c_o32_0, 2);
                o11 = vmlal_lane_s16(o11, r5, c_o32_1, 2);
                o11 = vmlsl_lane_s16(o11, r7, c_o32_3, 3);
                y1[0][3] = vqrshrn_n_s32(vaddq_s32(e[0][11], o11), 7);
                y2[0][4] = vqrshrn_n_s32(vsubq_s32(e[0][11], o11), 7);

                /* o12[0-3] */
                int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
                o12 = vmlsl_lane_s16(o12, r3, c_o32_1, 1);
                o12 = vmlal_lane_s16(o12, r5, c_o32_0, 0);
                o12 = vmlsl_lane_s16(o12, r7, c_o32_2, 0);
                y1[0][4] = vqrshrn_n_s32(vaddq_s32(e[0][12], o12), 7);
                y2[0][3] = vqrshrn_n_s32(vsubq_s32(e[0][12], o12), 7);

                /* o13[0-3] */
                int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
                o13 = vmlsl_lane_s16(o13, r3, c_o32_2, 0);
                o13 = vmlal_lane_s16(o13, r5, c_o32_0, 3);
                o13 = vmlsl_lane_s16(o13, r7, c_o32_0, 0);
                y1[0][5] = vqrshrn_n_s32(vaddq_s32(e[0][13], o13), 7);
                y2[0][2] = vqrshrn_n_s32(vsubq_s32(e[0][13], o13), 7);

                /* o14[0-3] */
                int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
                o14 = vmlsl_lane_s16(o14, r3, c_o32_2, 3);
                o14 = vmlal_lane_s16(o14, r5, c_o32_2, 0);
                o14 = vmlsl_lane_s16(o14, r7, c_o32_1, 1);
                y1[0][6] = vqrshrn_n_s32(vaddq_s32(e[0][14], o14), 7);
                y2[0][1] = vqrshrn_n_s32(vsubq_s32(e[0][14], o14), 7);

                /* o15[0-3] */
                int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
                o15 = vmlsl_lane_s16(o15, r3, c_o32_3, 2);
                o15 = vmlal_lane_s16(o15, r5, c_o32_3, 1);
                o15 = vmlsl_lane_s16(o15, r7, c_o32_3, 0);
                y1[0][7] = vqrshrn_n_s32(vaddq_s32(e[0][15], o15), 7);
                y2[0][0] = vqrshrn_n_s32(vsubq_s32(e[0][15], o15), 7);
            }

            /* [4-7] */
            {
                int16x4_t r1 = vget_high_s16(m_temp_reg_70);
                int16x4_t r3 = vget_high_s16(m_temp_reg_71);
                int16x4_t r5 = vget_high_s16(m_temp_reg_72);
                int16x4_t r7 = vget_high_s16(m_temp_reg_73);
                /* o0[4-7] */
                int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
                o0 = vmlal_lane_s16(o0, r3, c_o32_0, 1);
                o0 = vmlal_lane_s16(o0, r5, c_o32_0, 2);
                o0 = vmlal_lane_s16(o0, r7, c_o32_0, 3);
                y0[1][0] = vqrshrn_n_s32(vaddq_s32(e[1][0], o0), 7);
                y3[1][7] = vqrshrn_n_s32(vsubq_s32(e[1][0], o0), 7);

                /* o1[4-7] */
                int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
                o1 = vmlal_lane_s16(o1, r3, c_o32_1, 0);
                o1 = vmlal_lane_s16(o1, r5, c_o32_1, 3);
                o1 = vmlal_lane_s16(o1, r7, c_o32_2, 2);
                y0[1][1] = vqrshrn_n_s32(vaddq_s32(e[1][1], o1), 7);
                y3[1][6] = vqrshrn_n_s32(vsubq_s32(e[1][1], o1), 7);

                /* o2[4-7] */
                int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
                o2 = vmlal_lane_s16(o2, r3, c_o32_1, 3);
                o2 = vmlal_lane_s16(o2, r5, c_o32_3, 0);
                o2 = vmlsl_lane_s16(o2, r7, c_o32_3, 2);
                y0[1][2] = vqrshrn_n_s32(vaddq_s32(e[1][2], o2), 7);
                y3[1][5] = vqrshrn_n_s32(vsubq_s32(e[1][2], o2), 7);

                /* o3[4-7] */
                int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
                o3 = vmlal_lane_s16(o3, r3, c_o32_2, 2);
                o3 = vmlsl_lane_s16(o3, r5, c_o32_3, 2);
                o3 = vmlsl_lane_s16(o3, r7, c_o32_1, 3);
                y0[1][3] = vqrshrn_n_s32(vaddq_s32(e[1][3], o3), 7);
                y3[1][4] = vqrshrn_n_s32(vsubq_s32(e[1][3], o3), 7);

                /* o4[4-7] */
                int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
                o4 = vmlal_lane_s16(o4, r3, c_o32_3, 1);
                o4 = vmlsl_lane_s16(o4, r5, c_o32_2, 1);
                o4 = vmlsl_lane_s16(o4, r7, c_o32_0, 0);
                y0[1][4] = vqrshrn_n_s32(vaddq_s32(e[1][4], o4), 7);
                y3[1][3] = vqrshrn_n_s32(vsubq_s32(e[1][4], o4), 7);

                /* o5[4-7] */
                int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
                o5 = vmlsl_lane_s16(o5, r3, c_o32_3, 3);
                o5 = vmlsl_lane_s16(o5, r5, c_o32_1, 0);
                o5 = vmlsl_lane_s16(o5, r7, c_o32_1, 2);
                y0[1][5] = vqrshrn_n_s32(vaddq_s32(e[1][5], o5), 7);
                y3[1][2] = vqrshrn_n_s32(vsubq_s32(e[1][5], o5), 7);

                /* o6[4-7] */
                int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
                o6 = vmlsl_lane_s16(o6, r3, c_o32_3, 0);
                o6 = vmlsl_lane_s16(o6, r5, c_o32_0, 0);
                o6 = vmlsl_lane_s16(o6, r7, c_o32_3, 1);
                y0[1][6] = vqrshrn_n_s32(vaddq_s32(e[1][6], o6), 7);
                y3[1][1] = vqrshrn_n_s32(vsubq_s32(e[1][6], o6), 7);

                /* o7[4-7] */
                int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
                o7 = vmlsl_lane_s16(o7, r3, c_o32_2, 1);
                o7 = vmlsl_lane_s16(o7, r5, c_o32_1, 1);
                o7 = vmlal_lane_s16(o7, r7, c_o32_2, 3);
                y0[1][7] = vqrshrn_n_s32(vaddq_s32(e[1][7], o7), 7);
                y3[1][0] = vqrshrn_n_s32(vsubq_s32(e[1][7], o7), 7);

                /* o8[4-7] */
                int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
                o8 = vmlsl_lane_s16(o8, r3, c_o32_1, 2);
                o8 = vmlsl_lane_s16(o8, r5, c_o32_2, 2);
                o8 = vmlal_lane_s16(o8, r7, c_o32_1, 0);
                y1[1][0] = vqrshrn_n_s32(vaddq_s32(e[1][8], o8), 7);
                y2[1][7] = vqrshrn_n_s32(vsubq_s32(e[1][8], o8), 7);

                /* o9[4-7] */
                int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
                o9 = vmlsl_lane_s16(o9, r3, c_o32_0, 3);
                o9 = vmlsl_lane_s16(o9, r5, c_o32_3, 3);
                o9 = vmlal_lane_s16(o9, r7, c_o32_0, 2);
                y1[1][1] = vqrshrn_n_s32(vaddq_s32(e[1][9], o9), 7);
                y2[1][6] = vqrshrn_n_s32(vsubq_s32(e[1][9], o9), 7);

                /* o10[4-7] */
                int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
                o10 = vmlsl_lane_s16(o10, r3, c_o32_0, 1);
                o10 = vmlal_lane_s16(o10, r5, c_o32_2, 3);
                o10 = vmlal_lane_s16(o10, r7, c_o32_2, 1);
                y1[1][2] = vqrshrn_n_s32(vaddq_s32(e[1][10], o10), 7);
                y2[1][5] = vqrshrn_n_s32(vsubq_s32(e[1][10], o10), 7);

                /* o11[4-7] */
                int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
                o11 = vmlsl_lane_s16(o11, r3, c_o32_0, 2);
                o11 = vmlal_lane_s16(o11, r5, c_o32_1, 2);
                o11 = vmlsl_lane_s16(o11, r7, c_o32_3, 3);
                y1[1][3] = vqrshrn_n_s32(vaddq_s32(e[1][11], o11), 7);
                y2[1][4] = vqrshrn_n_s32(vsubq_s32(e[1][11], o11), 7);

                /* o12[4-7] */
                int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
                o12 = vmlsl_lane_s16(o12, r3, c_o32_1, 1);
                o12 = vmlal_lane_s16(o12, r5, c_o32_0, 0);
                o12 = vmlsl_lane_s16(o12, r7, c_o32_2, 0);
                y1[1][4] = vqrshrn_n_s32(vaddq_s32(e[1][12], o12), 7);
                y2[1][3] = vqrshrn_n_s32(vsubq_s32(e[1][12], o12), 7);

                /* o13[4-7] */
                int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
                o13 = vmlsl_lane_s16(o13, r3, c_o32_2, 0);
                o13 = vmlal_lane_s16(o13, r5, c_o32_0, 3);
                o13 = vmlsl_lane_s16(o13, r7, c_o32_0, 0);
                y1[1][5] = vqrshrn_n_s32(vaddq_s32(e[1][13], o13), 7);
                y2[1][2] = vqrshrn_n_s32(vsubq_s32(e[1][13], o13), 7);

                /* o14[4-7] */
                int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
                o14 = vmlsl_lane_s16(o14, r3, c_o32_2, 3);
                o14 = vmlal_lane_s16(o14, r5, c_o32_2, 0);
                o14 = vmlsl_lane_s16(o14, r7, c_o32_1, 1);
                y1[1][6] = vqrshrn_n_s32(vaddq_s32(e[1][14], o14), 7);
                y2[1][1] = vqrshrn_n_s32(vsubq_s32(e[1][14], o14), 7);

                /* o15[4-7] */
                int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
                o15 = vmlsl_lane_s16(o15, r3, c_o32_3, 2);
                o15 = vmlal_lane_s16(o15, r5, c_o32_3, 1);
                o15 = vmlsl_lane_s16(o15, r7, c_o32_3, 0);
                y1[1][7] = vqrshrn_n_s32(vaddq_s32(e[1][15], o15), 7);
                y2[1][0] = vqrshrn_n_s32(vsubq_s32(e[1][15], o15), 7);
            }
        }
        else
        {
            /* o & stage 1 out */
            /* [0-3] */
            {
                int16x4_t r1 = vget_low_s16(m_temp_reg_70);
                int16x4_t r3 = vget_low_s16(m_temp_reg_71);
                int16x4_t r5 = vget_low_s16(m_temp_reg_72);
                int16x4_t r7 = vget_low_s16(m_temp_reg_73);
                int16x4_t r9 = vget_low_s16(m_temp_reg_74);
                int16x4_t r11 = vget_low_s16(m_temp_reg_75);
                int16x4_t r13 = vget_low_s16(m_temp_reg_76);
                int16x4_t r15 = vget_low_s16(m_temp_reg_77);
                int16x4_t r17 = vget_low_s16(m_temp_reg_80);
                int16x4_t r19 = vget_low_s16(m_temp_reg_81);
                int16x4_t r21 = vget_low_s16(m_temp_reg_82);
                int16x4_t r23 = vget_low_s16(m_temp_reg_83);
                int16x4_t r25 = vget_low_s16(m_temp_reg_84);
                int16x4_t r27 = vget_low_s16(m_temp_reg_85);
                int16x4_t r29 = vget_low_s16(m_temp_reg_86);
                int16x4_t r31 = vget_low_s16(m_temp_reg_87);
                /* o0[0-3] */
                int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
                o0 = vmlal_lane_s16(o0, r3, c_o32_0, 1);
                o0 = vmlal_lane_s16(o0, r5, c_o32_0, 2);
                o0 = vmlal_lane_s16(o0, r7, c_o32_0, 3);
                o0 = vmlal_lane_s16(o0, r9, c_o32_1, 0);
                o0 = vmlal_lane_s16(o0, r11, c_o32_1, 1);
                o0 = vmlal_lane_s16(o0, r13, c_o32_1, 2);
                o0 = vmlal_lane_s16(o0, r15, c_o32_1, 3);
                o0 = vmlal_lane_s16(o0, r17, c_o32_2, 0);
                o0 = vmlal_lane_s16(o0, r19, c_o32_2, 1);
                o0 = vmlal_lane_s16(o0, r21, c_o32_2, 2);
                o0 = vmlal_lane_s16(o0, r23, c_o32_2, 3);
                o0 = vmlal_lane_s16(o0, r25, c_o32_3, 0);
                o0 = vmlal_lane_s16(o0, r27, c_o32_3, 1);
                o0 = vmlal_lane_s16(o0, r29, c_o32_3, 2);
                o0 = vmlal_lane_s16(o0, r31, c_o32_3, 3);
                y0[0][0] = vqrshrn_n_s32(vaddq_s32(e[0][0], o0), 7);
                y3[0][7] = vqrshrn_n_s32(vsubq_s32(e[0][0], o0), 7);

                /* o1[0-3] */
                int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
                o1 = vmlal_lane_s16(o1, r3, c_o32_1, 0);
                o1 = vmlal_lane_s16(o1, r5, c_o32_1, 3);
                o1 = vmlal_lane_s16(o1, r7, c_o32_2, 2);
                o1 = vmlal_lane_s16(o1, r9, c_o32_3, 1);
                o1 = vmlsl_lane_s16(o1, r11, c_o32_3, 3);
                o1 = vmlsl_lane_s16(o1, r13, c_o32_3, 0);
                o1 = vmlsl_lane_s16(o1, r15, c_o32_2, 1);
                o1 = vmlsl_lane_s16(o1, r17, c_o32_1, 2);
                o1 = vmlsl_lane_s16(o1, r19, c_o32_0, 3);
                o1 = vmlsl_lane_s16(o1, r21, c_o32_0, 0);
                o1 = vmlsl_lane_s16(o1, r23, c_o32_0, 2);
                o1 = vmlsl_lane_s16(o1, r25, c_o32_1, 1);
                o1 = vmlsl_lane_s16(o1, r27, c_o32_2, 0);
                o1 = vmlsl_lane_s16(o1, r29, c_o32_2, 3);
                o1 = vmlsl_lane_s16(o1, r31, c_o32_3, 2);
                y0[0][1] = vqrshrn_n_s32(vaddq_s32(e[0][1], o1), 7);
                y3[0][6] = vqrshrn_n_s32(vsubq_s32(e[0][1], o1), 7);

                /* o2[0-3] */
                int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
                o2 = vmlal_lane_s16(o2, r3, c_o32_1, 3);
                o2 = vmlal_lane_s16(o2, r5, c_o32_3, 0);
                o2 = vmlsl_lane_s16(o2, r7, c_o32_3, 2);
                o2 = vmlsl_lane_s16(o2, r9, c_o32_2, 1);
                o2 = vmlsl_lane_s16(o2, r11, c_o32_1, 0);
                o2 = vmlsl_lane_s16(o2, r13, c_o32_0, 0);
                o2 = vmlsl_lane_s16(o2, r15, c_o32_1, 1);
                o2 = vmlsl_lane_s16(o2, r17, c_o32_2, 2);
                o2 = vmlsl_lane_s16(o2, r19, c_o32_3, 3);
                o2 = vmlal_lane_s16(o2, r21, c_o32_2, 3);
                o2 = vmlal_lane_s16(o2, r23, c_o32_1, 2);
                o2 = vmlal_lane_s16(o2, r25, c_o32_0, 0);
                o2 = vmlal_lane_s16(o2, r27, c_o32_0, 3);
                o2 = vmlal_lane_s16(o2, r29, c_o32_2, 0);
                o2 = vmlal_lane_s16(o2, r31, c_o32_3, 1);
                y0[0][2] = vqrshrn_n_s32(vaddq_s32(e[0][2], o2), 7);
                y3[0][5] = vqrshrn_n_s32(vsubq_s32(e[0][2], o2), 7);

                /* o3[0-3] */
                int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
                o3 = vmlal_lane_s16(o3, r3, c_o32_2, 2);
                o3 = vmlsl_lane_s16(o3, r5, c_o32_3, 2);
                o3 = vmlsl_lane_s16(o3, r7, c_o32_1, 3);
                o3 = vmlsl_lane_s16(o3, r9, c_o32_0, 0);
                o3 = vmlsl_lane_s16(o3, r11, c_o32_1, 2);
                o3 = vmlsl_lane_s16(o3, r13, c_o32_3, 1);
                o3 = vmlal_lane_s16(o3, r15, c_o32_2, 3);
                o3 = vmlal_lane_s16(o3, r17, c_o32_1, 0);
                o3 = vmlal_lane_s16(o3, r19, c_o32_0, 2);
                o3 = vmlal_lane_s16(o3, r21, c_o32_2, 1);
                o3 = vmlsl_lane_s16(o3, r23, c_o32_3, 3);
                o3 = vmlsl_lane_s16(o3, r25, c_o32_2, 0);
                o3 = vmlsl_lane_s16(o3, r27, c_o32_0, 0);
                o3 = vmlsl_lane_s16(o3, r29, c_o32_1, 1);
                o3 = vmlsl_lane_s16(o3, r31, c_o32_3, 0);
                y0[0][3] = vqrshrn_n_s32(vaddq_s32(e[0][3], o3), 7);
                y3[0][4] = vqrshrn_n_s32(vsubq_s32(e[0][3], o3), 7);

                /* o4[0-3] */
                int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
                o4 = vmlal_lane_s16(o4, r3, c_o32_3, 1);
                o4 = vmlsl_lane_s16(o4, r5, c_o32_2, 1);
                o4 = vmlsl_lane_s16(o4, r7, c_o32_0, 0);
                o4 = vmlsl_lane_s16(o4, r9, c_o32_2, 0);
                o4 = vmlal_lane_s16(o4, r11, c_o32_3, 2);
                o4 = vmlal_lane_s16(o4, r13, c_o32_1, 1);
                o4 = vmlal_lane_s16(o4, r15, c_o32_0, 3);
                o4 = vmlal_lane_s16(o4, r17, c_o32_3, 0);
                o4 = vmlsl_lane_s16(o4, r19, c_o32_2, 2);
                o4 = vmlsl_lane_s16(o4, r21, c_o32_0, 0);
                o4 = vmlsl_lane_s16(o4, r23, c_o32_1, 3);
                o4 = vmlal_lane_s16(o4, r25, c_o32_3, 3);
                o4 = vmlal_lane_s16(o4, r27, c_o32_1, 2);
                o4 = vmlal_lane_s16(o4, r29, c_o32_0, 2);
                o4 = vmlal_lane_s16(o4, r31, c_o32_2, 3);
                y0[0][4] = vqrshrn_n_s32(vaddq_s32(e[0][4], o4), 7);
                y3[0][3] = vqrshrn_n_s32(vsubq_s32(e[0][4], o4), 7);

                /* o5[0-3] */
                int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
                o5 = vmlsl_lane_s16(o5, r3, c_o32_3, 3);
                o5 = vmlsl_lane_s16(o5, r5, c_o32_1, 0);
                o5 = vmlsl_lane_s16(o5, r7, c_o32_1, 2);
                o5 = vmlal_lane_s16(o5, r9, c_o32_3, 2);
                o5 = vmlal_lane_s16(o5, r11, c_o32_0, 3);
                o5 = vmlal_lane_s16(o5, r13, c_o32_1, 3);
                o5 = vmlsl_lane_s16(o5, r15, c_o32_3, 1);
                o5 = vmlsl_lane_s16(o5, r17, c_o32_0, 2);
                o5 = vmlsl_lane_s16(o5, r19, c_o32_2, 0);
                o5 = vmlal_lane_s16(o5, r21, c_o32_3, 0);
                o5 = vmlal_lane_s16(o5, r23, c_o32_0, 0);
                o5 = vmlal_lane_s16(o5, r25, c_o32_2, 1);
                o5 = vmlsl_lane_s16(o5, r27, c_o32_2, 3);
                o5 = vmlsl_lane_s16(o5, r29, c_o32_0, 0);
                o5 = vmlsl_lane_s16(o5, r31, c_o32_2, 2);
                y0[0][5] = vqrshrn_n_s32(vaddq_s32(e[0][5], o5), 7);
                y3[0][2] = vqrshrn_n_s32(vsubq_s32(e[0][5], o5), 7);

                /* o6[0-3] */
                int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
                o6 = vmlsl_lane_s16(o6, r3, c_o32_3, 0);
                o6 = vmlsl_lane_s16(o6, r5, c_o32_0, 0);
                o6 = vmlsl_lane_s16(o6, r7, c_o32_3, 1);
                o6 = vmlal_lane_s16(o6, r9, c_o32_1, 1);
                o6 = vmlal_lane_s16(o6, r11, c_o32_1, 3);
                o6 = vmlsl_lane_s16(o6, r13, c_o32_2, 3);
                o6 = vmlsl_lane_s16(o6, r15, c_o32_0, 0);
                o6 = vmlsl_lane_s16(o6, r17, c_o32_3, 2);
                o6 = vmlal_lane_s16(o6, r19, c_o32_1, 0);
                o6 = vmlal_lane_s16(o6, r21, c_o32_2, 0);
                o6 = vmlsl_lane_s16(o6, r23, c_o32_2, 2);
                o6 = vmlsl_lane_s16(o6, r25, c_o32_0, 2);
                o6 = vmlsl_lane_s16(o6, r27, c_o32_3, 3);
                o6 = vmlal_lane_s16(o6, r29, c_o32_0, 3);
                o6 = vmlal_lane_s16(o6, r31, c_o32_2, 1);
                y0[0][6] = vqrshrn_n_s32(vaddq_s32(e[0][6], o6), 7);
                y3[0][1] = vqrshrn_n_s32(vsubq_s32(e[0][6], o6), 7);

                /* o7[0-3] */
                int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
                o7 = vmlsl_lane_s16(o7, r3, c_o32_2, 1);
                o7 = vmlsl_lane_s16(o7, r5, c_o32_1, 1);
                o7 = vmlal_lane_s16(o7, r7, c_o32_2, 3);
                o7 = vmlal_lane_s16(o7, r9, c_o32_0, 3);
                o7 = vmlsl_lane_s16(o7, r11, c_o32_3, 1);
                o7 = vmlsl_lane_s16(o7, r13, c_o32_0, 0);
                o7 = vmlal_lane_s16(o7, r15, c_o32_3, 3);
                o7 = vmlal_lane_s16(o7, r17, c_o32_0, 0);
                o7 = vmlal_lane_s16(o7, r19, c_o32_3, 2);
                o7 = vmlsl_lane_s16(o7, r21, c_o32_0, 2);
                o7 = vmlsl_lane_s16(o7, r23, c_o32_3, 0);
                o7 = vmlal_lane_s16(o7, r25, c_o32_1, 0);
                o7 = vmlal_lane_s16(o7, r27, c_o32_2, 2);
                o7 = vmlsl_lane_s16(o7, r29, c_o32_1, 2);
                o7 = vmlsl_lane_s16(o7, r31, c_o32_2, 0);
                y0[0][7] = vqrshrn_n_s32(vaddq_s32(e[0][7], o7), 7);
                y3[0][0] = vqrshrn_n_s32(vsubq_s32(e[0][7], o7), 7);

                /* o8[0-3] */
                int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
                o8 = vmlsl_lane_s16(o8, r3, c_o32_1, 2);
                o8 = vmlsl_lane_s16(o8, r5, c_o32_2, 2);
                o8 = vmlal_lane_s16(o8, r7, c_o32_1, 0);
                o8 = vmlal_lane_s16(o8, r9, c_o32_3, 0);
                o8 = vmlsl_lane_s16(o8, r11, c_o32_0, 2);
                o8 = vmlsl_lane_s16(o8, r13, c_o32_3, 2);
                o8 = vmlal_lane_s16(o8, r15, c_o32_0, 0);
                o8 = vmlsl_lane_s16(o8, r17, c_o32_3, 3);
                o8 = vmlsl_lane_s16(o8, r19, c_o32_0, 0);
                o8 = vmlal_lane_s16(o8, r21, c_o32_3, 1);
                o8 = vmlal_lane_s16(o8, r23, c_o32_0, 3);
                o8 = vmlsl_lane_s16(o8, r25, c_o32_2, 3);
                o8 = vmlsl_lane_s16(o8, r27, c_o32_1, 1);
                o8 = vmlal_lane_s16(o8, r29, c_o32_2, 1);
                o8 = vmlal_lane_s16(o8, r31, c_o32_1, 3);
                y1[0][0] = vqrshrn_n_s32(vaddq_s32(e[0][8], o8), 7);
                y2[0][7] = vqrshrn_n_s32(vsubq_s32(e[0][8], o8), 7);

                /* o9[0-3] */
                int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
                o9 = vmlsl_lane_s16(o9, r3, c_o32_0, 3);
                o9 = vmlsl_lane_s16(o9, r5, c_o32_3, 3);
                o9 = vmlal_lane_s16(o9, r7, c_o32_0, 2);
                o9 = vmlsl_lane_s16(o9, r9, c_o32_2, 2);
                o9 = vmlsl_lane_s16(o9, r11, c_o32_2, 0);
                o9 = vmlal_lane_s16(o9, r13, c_o32_1, 0);
                o9 = vmlal_lane_s16(o9, r15, c_o32_3, 2);
                o9 = vmlsl_lane_s16(o9, r17, c_o32_0, 0);
                o9 = vmlal_lane_s16(o9, r19, c_o32_2, 3);
                o9 = vmlal_lane_s16(o9, r21, c_o32_1, 3);
                o9 = vmlsl_lane_s16(o9, r23, c_o32_1, 1);
                o9 = vmlsl_lane_s16(o9, r25, c_o32_3, 1);
                o9 = vmlal_lane_s16(o9, r27, c_o32_0, 0);
                o9 = vmlsl_lane_s16(o9, r29, c_o32_3, 0);
                o9 = vmlsl_lane_s16(o9, r31, c_o32_1, 2);
                y1[0][1] = vqrshrn_n_s32(vaddq_s32(e[0][9], o9), 7);
                y2[0][6] = vqrshrn_n_s32(vsubq_s32(e[0][9], o9), 7);

                /* o10[0-3] */
                int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
                o10 = vmlsl_lane_s16(o10, r3, c_o32_0, 1);
                o10 = vmlal_lane_s16(o10, r5, c_o32_2, 3);
                o10 = vmlal_lane_s16(o10, r7, c_o32_2, 1);
                o10 = vmlsl_lane_s16(o10, r9, c_o32_0, 0);
                o10 = vmlal_lane_s16(o10, r11, c_o32_3, 0);
                o10 = vmlal_lane_s16(o10, r13, c_o32_2, 0);
                o10 = vmlsl_lane_s16(o10, r15, c_o32_0, 2);
                o10 = vmlal_lane_s16(o10, r17, c_o32_3, 1);
                o10 = vmlal_lane_s16(o10, r19, c_o32_1, 3);
                o10 = vmlsl_lane_s16(o10, r21, c_o32_0, 3);
                o10 = vmlal_lane_s16(o10, r23, c_o32_3, 2);
                o10 = vmlal_lane_s16(o10, r25, c_o32_1, 2);
                o10 = vmlsl_lane_s16(o10, r27, c_o32_1, 0);
                o10 = vmlal_lane_s16(o10, r29, c_o32_3, 3);
                o10 = vmlal_lane_s16(o10, r31, c_o32_1, 1);
                y1[0][2] = vqrshrn_n_s32(vaddq_s32(e[0][10], o10), 7);
                y2[0][5] = vqrshrn_n_s32(vsubq_s32(e[0][10], o10), 7);

                /* o11[0-3] */
                int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
                o11 = vmlsl_lane_s16(o11, r3, c_o32_0, 2);
                o11 = vmlal_lane_s16(o11, r5, c_o32_1, 2);
                o11 = vmlsl_lane_s16(o11, r7, c_o32_3, 3);
                o11 = vmlsl_lane_s16(o11, r9, c_o32_1, 3);
                o11 = vmlal_lane_s16(o11, r11, c_o32_0, 0);
                o11 = vmlsl_lane_s16(o11, r13, c_o32_2, 2);
                o11 = vmlsl_lane_s16(o11, r15, c_o32_3, 0);
                o11 = vmlal_lane_s16(o11, r17, c_o32_0, 3);
                o11 = vmlsl_lane_s16(o11, r19, c_o32_1, 1);
                o11 = vmlal_lane_s16(o11, r21, c_o32_3, 2);
                o11 = vmlal_lane_s16(o11, r23, c_o32_2, 0);
                o11 = vmlsl_lane_s16(o11, r25, c_o32_0, 0);
                o11 = vmlal_lane_s16(o11, r27, c_o32_2, 1);
                o11 = vmlal_lane_s16(o11, r29, c_o32_3, 1);
                o11 = vmlsl_lane_s16(o11, r31, c_o32_1, 0);
                y1[0][3] = vqrshrn_n_s32(vaddq_s32(e[0][11], o11), 7);
                y2[0][4] = vqrshrn_n_s32(vsubq_s32(e[0][11], o11), 7);

                /* o12[0-3] */
                int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
                o12 = vmlsl_lane_s16(o12, r3, c_o32_1, 1);
                o12 = vmlal_lane_s16(o12, r5, c_o32_0, 0);
                o12 = vmlsl_lane_s16(o12, r7, c_o32_2, 0);
                o12 = vmlal_lane_s16(o12, r9, c_o32_3, 3);
                o12 = vmlal_lane_s16(o12, r11, c_o32_2, 1);
                o12 = vmlsl_lane_s16(o12, r13, c_o32_0, 2);
                o12 = vmlal_lane_s16(o12, r15, c_o32_1, 0);
                o12 = vmlsl_lane_s16(o12, r17, c_o32_2, 3);
                o12 = vmlsl_lane_s16(o12, r19, c_o32_3, 1);
                o12 = vmlal_lane_s16(o12, r21, c_o32_1, 2);
                o12 = vmlsl_lane_s16(o12, r23, c_o32_0, 0);
                o12 = vmlal_lane_s16(o12, r25, c_o32_1, 3);
                o12 = vmlsl_lane_s16(o12, r27, c_o32_3, 2);
                o12 = vmlsl_lane_s16(o12, r29, c_o32_2, 2);
                o12 = vmlal_lane_s16(o12, r31, c_o32_0, 3);
                y1[0][4] = vqrshrn_n_s32(vaddq_s32(e[0][12], o12), 7);
                y2[0][3] = vqrshrn_n_s32(vsubq_s32(e[0][12], o12), 7);

                /* o13[0-3] */
                int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
                o13 = vmlsl_lane_s16(o13, r3, c_o32_2, 0);
                o13 = vmlal_lane_s16(o13, r5, c_o32_0, 3);
                o13 = vmlsl_lane_s16(o13, r7, c_o32_0, 0);
                o13 = vmlal_lane_s16(o13, r9, c_o32_1, 2);
                o13 = vmlsl_lane_s16(o13, r11, c_o32_2, 3);
                o13 = vmlsl_lane_s16(o13, r13, c_o32_3, 3);
                o13 = vmlal_lane_s16(o13, r15, c_o32_2, 2);
                o13 = vmlsl_lane_s16(o13, r17, c_o32_1, 1);
                o13 = vmlal_lane_s16(o13, r19, c_o32_0, 0);
                o13 = vmlsl_lane_s16(o13, r21, c_o32_1, 0);
                o13 = vmlal_lane_s16(o13, r23, c_o32_2, 1);
                o13 = vmlsl_lane_s16(o13, r25, c_o32_3, 2);
                o13 = vmlsl_lane_s16(o13, r27, c_o32_3, 0);
                o13 = vmlal_lane_s16(o13, r29, c_o32_1, 3);
                o13 = vmlsl_lane_s16(o13, r31, c_o32_0, 2);
                y1[0][5] = vqrshrn_n_s32(vaddq_s32(e[0][13], o13), 7);
                y2[0][2] = vqrshrn_n_s32(vsubq_s32(e[0][13], o13), 7);

                /* o14[0-3] */
                int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
                o14 = vmlsl_lane_s16(o14, r3, c_o32_2, 3);
                o14 = vmlal_lane_s16(o14, r5, c_o32_2, 0);
                o14 = vmlsl_lane_s16(o14, r7, c_o32_1, 1);
                o14 = vmlal_lane_s16(o14, r9, c_o32_0, 2);
                o14 = vmlsl_lane_s16(o14, r11, c_o32_0, 0);
                o14 = vmlal_lane_s16(o14, r13, c_o32_0, 3);
                o14 = vmlsl_lane_s16(o14, r15, c_o32_1, 2);
                o14 = vmlal_lane_s16(o14, r17, c_o32_2, 1);
                o14 = vmlsl_lane_s16(o14, r19, c_o32_3, 0);
                o14 = vmlal_lane_s16(o14, r21, c_o32_3, 3);
                o14 = vmlal_lane_s16(o14, r23, c_o32_3, 1);
                o14 = vmlsl_lane_s16(o14, r25, c_o32_2, 2);
                o14 = vmlal_lane_s16(o14, r27, c_o32_1, 3);
                o14 = vmlsl_lane_s16(o14, r29, c_o32_1, 0);
                o14 = vmlal_lane_s16(o14, r31, c_o32_0, 0);
                y1[0][6] = vqrshrn_n_s32(vaddq_s32(e[0][14], o14), 7);
                y2[0][1] = vqrshrn_n_s32(vsubq_s32(e[0][14], o14), 7);

                /* o15[0-3] */
                int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
                o15 = vmlsl_lane_s16(o15, r3, c_o32_3, 2);
                o15 = vmlal_lane_s16(o15, r5, c_o32_3, 1);
                o15 = vmlsl_lane_s16(o15, r7, c_o32_3, 0);
                o15 = vmlal_lane_s16(o15, r9, c_o32_2, 3);
                o15 = vmlsl_lane_s16(o15, r11, c_o32_2, 2);
                o15 = vmlal_lane_s16(o15, r13, c_o32_2, 1);
                o15 = vmlsl_lane_s16(o15, r15, c_o32_2, 0);
                o15 = vmlal_lane_s16(o15, r17, c_o32_1, 3);
                o15 = vmlsl_lane_s16(o15, r19, c_o32_1, 2);
                o15 = vmlal_lane_s16(o15, r21, c_o32_1, 1);
                o15 = vmlsl_lane_s16(o15, r23, c_o32_1, 0);
                o15 = vmlal_lane_s16(o15, r25, c_o32_0, 3);
                o15 = vmlsl_lane_s16(o15, r27, c_o32_0, 2);
                o15 = vmlal_lane_s16(o15, r29, c_o32_0, 0);
                o15 = vmlsl_lane_s16(o15, r31, c_o32_0, 0);
                y1[0][7] = vqrshrn_n_s32(vaddq_s32(e[0][15], o15), 7);
                y2[0][0] = vqrshrn_n_s32(vsubq_s32(e[0][15], o15), 7);
            }

            /* [4-7] */
            {
                int16x4_t r1 = vget_high_s16(m_temp_reg_70);
                int16x4_t r3 = vget_high_s16(m_temp_reg_71);
                int16x4_t r5 = vget_high_s16(m_temp_reg_72);
                int16x4_t r7 = vget_high_s16(m_temp_reg_73);
                int16x4_t r9 = vget_high_s16(m_temp_reg_74);
                int16x4_t r11 = vget_high_s16(m_temp_reg_75);
                int16x4_t r13 = vget_high_s16(m_temp_reg_76);
                int16x4_t r15 = vget_high_s16(m_temp_reg_77);
                int16x4_t r17 = vget_high_s16(m_temp_reg_80);
                int16x4_t r19 = vget_high_s16(m_temp_reg_81);
                int16x4_t r21 = vget_high_s16(m_temp_reg_82);
                int16x4_t r23 = vget_high_s16(m_temp_reg_83);
                int16x4_t r25 = vget_high_s16(m_temp_reg_84);
                int16x4_t r27 = vget_high_s16(m_temp_reg_85);
                int16x4_t r29 = vget_high_s16(m_temp_reg_86);
                int16x4_t r31 = vget_high_s16(m_temp_reg_87);
                /* o0[4-7] */
                int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
                o0 = vmlal_lane_s16(o0, r3, c_o32_0, 1);
                o0 = vmlal_lane_s16(o0, r5, c_o32_0, 2);
                o0 = vmlal_lane_s16(o0, r7, c_o32_0, 3);
                o0 = vmlal_lane_s16(o0, r9, c_o32_1, 0);
                o0 = vmlal_lane_s16(o0, r11, c_o32_1, 1);
                o0 = vmlal_lane_s16(o0, r13, c_o32_1, 2);
                o0 = vmlal_lane_s16(o0, r15, c_o32_1, 3);
                o0 = vmlal_lane_s16(o0, r17, c_o32_2, 0);
                o0 = vmlal_lane_s16(o0, r19, c_o32_2, 1);
                o0 = vmlal_lane_s16(o0, r21, c_o32_2, 2);
                o0 = vmlal_lane_s16(o0, r23, c_o32_2, 3);
                o0 = vmlal_lane_s16(o0, r25, c_o32_3, 0);
                o0 = vmlal_lane_s16(o0, r27, c_o32_3, 1);
                o0 = vmlal_lane_s16(o0, r29, c_o32_3, 2);
                o0 = vmlal_lane_s16(o0, r31, c_o32_3, 3);
                y0[1][0] = vqrshrn_n_s32(vaddq_s32(e[1][0], o0), 7);
                y3[1][7] = vqrshrn_n_s32(vsubq_s32(e[1][0], o0), 7);

                /* o1[4-7] */
                int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
                o1 = vmlal_lane_s16(o1, r3, c_o32_1, 0);
                o1 = vmlal_lane_s16(o1, r5, c_o32_1, 3);
                o1 = vmlal_lane_s16(o1, r7, c_o32_2, 2);
                o1 = vmlal_lane_s16(o1, r9, c_o32_3, 1);
                o1 = vmlsl_lane_s16(o1, r11, c_o32_3, 3);
                o1 = vmlsl_lane_s16(o1, r13, c_o32_3, 0);
                o1 = vmlsl_lane_s16(o1, r15, c_o32_2, 1);
                o1 = vmlsl_lane_s16(o1, r17, c_o32_1, 2);
                o1 = vmlsl_lane_s16(o1, r19, c_o32_0, 3);
                o1 = vmlsl_lane_s16(o1, r21, c_o32_0, 0);
                o1 = vmlsl_lane_s16(o1, r23, c_o32_0, 2);
                o1 = vmlsl_lane_s16(o1, r25, c_o32_1, 1);
                o1 = vmlsl_lane_s16(o1, r27, c_o32_2, 0);
                o1 = vmlsl_lane_s16(o1, r29, c_o32_2, 3);
                o1 = vmlsl_lane_s16(o1, r31, c_o32_3, 2);
                y0[1][1] = vqrshrn_n_s32(vaddq_s32(e[1][1], o1), 7);
                y3[1][6] = vqrshrn_n_s32(vsubq_s32(e[1][1], o1), 7);

                /* o2[4-7] */
                int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
                o2 = vmlal_lane_s16(o2, r3, c_o32_1, 3);
                o2 = vmlal_lane_s16(o2, r5, c_o32_3, 0);
                o2 = vmlsl_lane_s16(o2, r7, c_o32_3, 2);
                o2 = vmlsl_lane_s16(o2, r9, c_o32_2, 1);
                o2 = vmlsl_lane_s16(o2, r11, c_o32_1, 0);
                o2 = vmlsl_lane_s16(o2, r13, c_o32_0, 0);
                o2 = vmlsl_lane_s16(o2, r15, c_o32_1, 1);
                o2 = vmlsl_lane_s16(o2, r17, c_o32_2, 2);
                o2 = vmlsl_lane_s16(o2, r19, c_o32_3, 3);
                o2 = vmlal_lane_s16(o2, r21, c_o32_2, 3);
                o2 = vmlal_lane_s16(o2, r23, c_o32_1, 2);
                o2 = vmlal_lane_s16(o2, r25, c_o32_0, 0);
                o2 = vmlal_lane_s16(o2, r27, c_o32_0, 3);
                o2 = vmlal_lane_s16(o2, r29, c_o32_2, 0);
                o2 = vmlal_lane_s16(o2, r31, c_o32_3, 1);
                y0[1][2] = vqrshrn_n_s32(vaddq_s32(e[1][2], o2), 7);
                y3[1][5] = vqrshrn_n_s32(vsubq_s32(e[1][2], o2), 7);

                /* o3[4-7] */
                int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
                o3 = vmlal_lane_s16(o3, r3, c_o32_2, 2);
                o3 = vmlsl_lane_s16(o3, r5, c_o32_3, 2);
                o3 = vmlsl_lane_s16(o3, r7, c_o32_1, 3);
                o3 = vmlsl_lane_s16(o3, r9, c_o32_0, 0);
                o3 = vmlsl_lane_s16(o3, r11, c_o32_1, 2);
                o3 = vmlsl_lane_s16(o3, r13, c_o32_3, 1);
                o3 = vmlal_lane_s16(o3, r15, c_o32_2, 3);
                o3 = vmlal_lane_s16(o3, r17, c_o32_1, 0);
                o3 = vmlal_lane_s16(o3, r19, c_o32_0, 2);
                o3 = vmlal_lane_s16(o3, r21, c_o32_2, 1);
                o3 = vmlsl_lane_s16(o3, r23, c_o32_3, 3);
                o3 = vmlsl_lane_s16(o3, r25, c_o32_2, 0);
                o3 = vmlsl_lane_s16(o3, r27, c_o32_0, 0);
                o3 = vmlsl_lane_s16(o3, r29, c_o32_1, 1);
                o3 = vmlsl_lane_s16(o3, r31, c_o32_3, 0);
                y0[1][3] = vqrshrn_n_s32(vaddq_s32(e[1][3], o3), 7);
                y3[1][4] = vqrshrn_n_s32(vsubq_s32(e[1][3], o3), 7);

                /* o4[4-7] */
                int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
                o4 = vmlal_lane_s16(o4, r3, c_o32_3, 1);
                o4 = vmlsl_lane_s16(o4, r5, c_o32_2, 1);
                o4 = vmlsl_lane_s16(o4, r7, c_o32_0, 0);
                o4 = vmlsl_lane_s16(o4, r9, c_o32_2, 0);
                o4 = vmlal_lane_s16(o4, r11, c_o32_3, 2);
                o4 = vmlal_lane_s16(o4, r13, c_o32_1, 1);
                o4 = vmlal_lane_s16(o4, r15, c_o32_0, 3);
                o4 = vmlal_lane_s16(o4, r17, c_o32_3, 0);
                o4 = vmlsl_lane_s16(o4, r19, c_o32_2, 2);
                o4 = vmlsl_lane_s16(o4, r21, c_o32_0, 0);
                o4 = vmlsl_lane_s16(o4, r23, c_o32_1, 3);
                o4 = vmlal_lane_s16(o4, r25, c_o32_3, 3);
                o4 = vmlal_lane_s16(o4, r27, c_o32_1, 2);
                o4 = vmlal_lane_s16(o4, r29, c_o32_0, 2);
                o4 = vmlal_lane_s16(o4, r31, c_o32_2, 3);
                y0[1][4] = vqrshrn_n_s32(vaddq_s32(e[1][4], o4), 7);
                y3[1][3] = vqrshrn_n_s32(vsubq_s32(e[1][4], o4), 7);

                /* o5[4-7] */
                int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
                o5 = vmlsl_lane_s16(o5, r3, c_o32_3, 3);
                o5 = vmlsl_lane_s16(o5, r5, c_o32_1, 0);
                o5 = vmlsl_lane_s16(o5, r7, c_o32_1, 2);
                o5 = vmlal_lane_s16(o5, r9, c_o32_3, 2);
                o5 = vmlal_lane_s16(o5, r11, c_o32_0, 3);
                o5 = vmlal_lane_s16(o5, r13, c_o32_1, 3);
                o5 = vmlsl_lane_s16(o5, r15, c_o32_3, 1);
                o5 = vmlsl_lane_s16(o5, r17, c_o32_0, 2);
                o5 = vmlsl_lane_s16(o5, r19, c_o32_2, 0);
                o5 = vmlal_lane_s16(o5, r21, c_o32_3, 0);
                o5 = vmlal_lane_s16(o5, r23, c_o32_0, 0);
                o5 = vmlal_lane_s16(o5, r25, c_o32_2, 1);
                o5 = vmlsl_lane_s16(o5, r27, c_o32_2, 3);
                o5 = vmlsl_lane_s16(o5, r29, c_o32_0, 0);
                o5 = vmlsl_lane_s16(o5, r31, c_o32_2, 2);
                y0[1][5] = vqrshrn_n_s32(vaddq_s32(e[1][5], o5), 7);
                y3[1][2] = vqrshrn_n_s32(vsubq_s32(e[1][5], o5), 7);

                /* o6[4-7] */
                int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
                o6 = vmlsl_lane_s16(o6, r3, c_o32_3, 0);
                o6 = vmlsl_lane_s16(o6, r5, c_o32_0, 0);
                o6 = vmlsl_lane_s16(o6, r7, c_o32_3, 1);
                o6 = vmlal_lane_s16(o6, r9, c_o32_1, 1);
                o6 = vmlal_lane_s16(o6, r11, c_o32_1, 3);
                o6 = vmlsl_lane_s16(o6, r13, c_o32_2, 3);
                o6 = vmlsl_lane_s16(o6, r15, c_o32_0, 0);
                o6 = vmlsl_lane_s16(o6, r17, c_o32_3, 2);
                o6 = vmlal_lane_s16(o6, r19, c_o32_1, 0);
                o6 = vmlal_lane_s16(o6, r21, c_o32_2, 0);
                o6 = vmlsl_lane_s16(o6, r23, c_o32_2, 2);
                o6 = vmlsl_lane_s16(o6, r25, c_o32_0, 2);
                o6 = vmlsl_lane_s16(o6, r27, c_o32_3, 3);
                o6 = vmlal_lane_s16(o6, r29, c_o32_0, 3);
                o6 = vmlal_lane_s16(o6, r31, c_o32_2, 1);
                y0[1][6] = vqrshrn_n_s32(vaddq_s32(e[1][6], o6), 7);
                y3[1][1] = vqrshrn_n_s32(vsubq_s32(e[1][6], o6), 7);

                /* o7[4-7] */
                int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
                o7 = vmlsl_lane_s16(o7, r3, c_o32_2, 1);
                o7 = vmlsl_lane_s16(o7, r5, c_o32_1, 1);
                o7 = vmlal_lane_s16(o7, r7, c_o32_2, 3);
                o7 = vmlal_lane_s16(o7, r9, c_o32_0, 3);
                o7 = vmlsl_lane_s16(o7, r11, c_o32_3, 1);
                o7 = vmlsl_lane_s16(o7, r13, c_o32_0, 0);
                o7 = vmlal_lane_s16(o7, r15, c_o32_3, 3);
                o7 = vmlal_lane_s16(o7, r17, c_o32_0, 0);
                o7 = vmlal_lane_s16(o7, r19, c_o32_3, 2);
                o7 = vmlsl_lane_s16(o7, r21, c_o32_0, 2);
                o7 = vmlsl_lane_s16(o7, r23, c_o32_3, 0);
                o7 = vmlal_lane_s16(o7, r25, c_o32_1, 0);
                o7 = vmlal_lane_s16(o7, r27, c_o32_2, 2);
                o7 = vmlsl_lane_s16(o7, r29, c_o32_1, 2);
                o7 = vmlsl_lane_s16(o7, r31, c_o32_2, 0);
                y0[1][7] = vqrshrn_n_s32(vaddq_s32(e[1][7], o7), 7);
                y3[1][0] = vqrshrn_n_s32(vsubq_s32(e[1][7], o7), 7);

                /* o8[4-7] */
                int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
                o8 = vmlsl_lane_s16(o8, r3, c_o32_1, 2);
                o8 = vmlsl_lane_s16(o8, r5, c_o32_2, 2);
                o8 = vmlal_lane_s16(o8, r7, c_o32_1, 0);
                o8 = vmlal_lane_s16(o8, r9, c_o32_3, 0);
                o8 = vmlsl_lane_s16(o8, r11, c_o32_0, 2);
                o8 = vmlsl_lane_s16(o8, r13, c_o32_3, 2);
                o8 = vmlal_lane_s16(o8, r15, c_o32_0, 0);
                o8 = vmlsl_lane_s16(o8, r17, c_o32_3, 3);
                o8 = vmlsl_lane_s16(o8, r19, c_o32_0, 0);
                o8 = vmlal_lane_s16(o8, r21, c_o32_3, 1);
                o8 = vmlal_lane_s16(o8, r23, c_o32_0, 3);
                o8 = vmlsl_lane_s16(o8, r25, c_o32_2, 3);
                o8 = vmlsl_lane_s16(o8, r27, c_o32_1, 1);
                o8 = vmlal_lane_s16(o8, r29, c_o32_2, 1);
                o8 = vmlal_lane_s16(o8, r31, c_o32_1, 3);
                y1[1][0] = vqrshrn_n_s32(vaddq_s32(e[1][8], o8), 7);
                y2[1][7] = vqrshrn_n_s32(vsubq_s32(e[1][8], o8), 7);

                /* o9[4-7] */
                int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
                o9 = vmlsl_lane_s16(o9, r3, c_o32_0, 3);
                o9 = vmlsl_lane_s16(o9, r5, c_o32_3, 3);
                o9 = vmlal_lane_s16(o9, r7, c_o32_0, 2);
                o9 = vmlsl_lane_s16(o9, r9, c_o32_2, 2);
                o9 = vmlsl_lane_s16(o9, r11, c_o32_2, 0);
                o9 = vmlal_lane_s16(o9, r13, c_o32_1, 0);
                o9 = vmlal_lane_s16(o9, r15, c_o32_3, 2);
                o9 = vmlsl_lane_s16(o9, r17, c_o32_0, 0);
                o9 = vmlal_lane_s16(o9, r19, c_o32_2, 3);
                o9 = vmlal_lane_s16(o9, r21, c_o32_1, 3);
                o9 = vmlsl_lane_s16(o9, r23, c_o32_1, 1);
                o9 = vmlsl_lane_s16(o9, r25, c_o32_3, 1);
                o9 = vmlal_lane_s16(o9, r27, c_o32_0, 0);
                o9 = vmlsl_lane_s16(o9, r29, c_o32_3, 0);
                o9 = vmlsl_lane_s16(o9, r31, c_o32_1, 2);
                y1[1][1] = vqrshrn_n_s32(vaddq_s32(e[1][9], o9), 7);
                y2[1][6] = vqrshrn_n_s32(vsubq_s32(e[1][9], o9), 7);

                /* o10[4-7] */
                int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
                o10 = vmlsl_lane_s16(o10, r3, c_o32_0, 1);
                o10 = vmlal_lane_s16(o10, r5, c_o32_2, 3);
                o10 = vmlal_lane_s16(o10, r7, c_o32_2, 1);
                o10 = vmlsl_lane_s16(o10, r9, c_o32_0, 0);
                o10 = vmlal_lane_s16(o10, r11, c_o32_3, 0);
                o10 = vmlal_lane_s16(o10, r13, c_o32_2, 0);
                o10 = vmlsl_lane_s16(o10, r15, c_o32_0, 2);
                o10 = vmlal_lane_s16(o10, r17, c_o32_3, 1);
                o10 = vmlal_lane_s16(o10, r19, c_o32_1, 3);
                o10 = vmlsl_lane_s16(o10, r21, c_o32_0, 3);
                o10 = vmlal_lane_s16(o10, r23, c_o32_3, 2);
                o10 = vmlal_lane_s16(o10, r25, c_o32_1, 2);
                o10 = vmlsl_lane_s16(o10, r27, c_o32_1, 0);
                o10 = vmlal_lane_s16(o10, r29, c_o32_3, 3);
                o10 = vmlal_lane_s16(o10, r31, c_o32_1, 1);
                y1[1][2] = vqrshrn_n_s32(vaddq_s32(e[1][10], o10), 7);
                y2[1][5] = vqrshrn_n_s32(vsubq_s32(e[1][10], o10), 7);

                /* o11[4-7] */
                int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
                o11 = vmlsl_lane_s16(o11, r3, c_o32_0, 2);
                o11 = vmlal_lane_s16(o11, r5, c_o32_1, 2);
                o11 = vmlsl_lane_s16(o11, r7, c_o32_3, 3);
                o11 = vmlsl_lane_s16(o11, r9, c_o32_1, 3);
                o11 = vmlal_lane_s16(o11, r11, c_o32_0, 0);
                o11 = vmlsl_lane_s16(o11, r13, c_o32_2, 2);
                o11 = vmlsl_lane_s16(o11, r15, c_o32_3, 0);
                o11 = vmlal_lane_s16(o11, r17, c_o32_0, 3);
                o11 = vmlsl_lane_s16(o11, r19, c_o32_1, 1);
                o11 = vmlal_lane_s16(o11, r21, c_o32_3, 2);
                o11 = vmlal_lane_s16(o11, r23, c_o32_2, 0);
                o11 = vmlsl_lane_s16(o11, r25, c_o32_0, 0);
                o11 = vmlal_lane_s16(o11, r27, c_o32_2, 1);
                o11 = vmlal_lane_s16(o11, r29, c_o32_3, 1);
                o11 = vmlsl_lane_s16(o11, r31, c_o32_1, 0);
                y1[1][3] = vqrshrn_n_s32(vaddq_s32(e[1][11], o11), 7);
                y2[1][4] = vqrshrn_n_s32(vsubq_s32(e[1][11], o11), 7);

                /* o12[4-7] */
                int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
                o12 = vmlsl_lane_s16(o12, r3, c_o32_1, 1);
                o12 = vmlal_lane_s16(o12, r5, c_o32_0, 0);
                o12 = vmlsl_lane_s16(o12, r7, c_o32_2, 0);
                o12 = vmlal_lane_s16(o12, r9, c_o32_3, 3);
                o12 = vmlal_lane_s16(o12, r11, c_o32_2, 1);
                o12 = vmlsl_lane_s16(o12, r13, c_o32_0, 2);
                o12 = vmlal_lane_s16(o12, r15, c_o32_1, 0);
                o12 = vmlsl_lane_s16(o12, r17, c_o32_2, 3);
                o12 = vmlsl_lane_s16(o12, r19, c_o32_3, 1);
                o12 = vmlal_lane_s16(o12, r21, c_o32_1, 2);
                o12 = vmlsl_lane_s16(o12, r23, c_o32_0, 0);
                o12 = vmlal_lane_s16(o12, r25, c_o32_1, 3);
                o12 = vmlsl_lane_s16(o12, r27, c_o32_3, 2);
                o12 = vmlsl_lane_s16(o12, r29, c_o32_2, 2);
                o12 = vmlal_lane_s16(o12, r31, c_o32_0, 3);
                y1[1][4] = vqrshrn_n_s32(vaddq_s32(e[1][12], o12), 7);
                y2[1][3] = vqrshrn_n_s32(vsubq_s32(e[1][12], o12), 7);

                /* o13[4-7] */
                int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
                o13 = vmlsl_lane_s16(o13, r3, c_o32_2, 0);
                o13 = vmlal_lane_s16(o13, r5, c_o32_0, 3);
                o13 = vmlsl_lane_s16(o13, r7, c_o32_0, 0);
                o13 = vmlal_lane_s16(o13, r9, c_o32_1, 2);
                o13 = vmlsl_lane_s16(o13, r11, c_o32_2, 3);
                o13 = vmlsl_lane_s16(o13, r13, c_o32_3, 3);
                o13 = vmlal_lane_s16(o13, r15, c_o32_2, 2);
                o13 = vmlsl_lane_s16(o13, r17, c_o32_1, 1);
                o13 = vmlal_lane_s16(o13, r19, c_o32_0, 0);
                o13 = vmlsl_lane_s16(o13, r21, c_o32_1, 0);
                o13 = vmlal_lane_s16(o13, r23, c_o32_2, 1);
                o13 = vmlsl_lane_s16(o13, r25, c_o32_3, 2);
                o13 = vmlsl_lane_s16(o13, r27, c_o32_3, 0);
                o13 = vmlal_lane_s16(o13, r29, c_o32_1, 3);
                o13 = vmlsl_lane_s16(o13, r31, c_o32_0, 2);
                y1[1][5] = vqrshrn_n_s32(vaddq_s32(e[1][13], o13), 7);
                y2[1][2] = vqrshrn_n_s32(vsubq_s32(e[1][13], o13), 7);

                /* o14[4-7] */
                int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
                o14 = vmlsl_lane_s16(o14, r3, c_o32_2, 3);
                o14 = vmlal_lane_s16(o14, r5, c_o32_2, 0);
                o14 = vmlsl_lane_s16(o14, r7, c_o32_1, 1);
                o14 = vmlal_lane_s16(o14, r9, c_o32_0, 2);
                o14 = vmlsl_lane_s16(o14, r11, c_o32_0, 0);
                o14 = vmlal_lane_s16(o14, r13, c_o32_0, 3);
                o14 = vmlsl_lane_s16(o14, r15, c_o32_1, 2);
                o14 = vmlal_lane_s16(o14, r17, c_o32_2, 1);
                o14 = vmlsl_lane_s16(o14, r19, c_o32_3, 0);
                o14 = vmlal_lane_s16(o14, r21, c_o32_3, 3);
                o14 = vmlal_lane_s16(o14, r23, c_o32_3, 1);
                o14 = vmlsl_lane_s16(o14, r25, c_o32_2, 2);
                o14 = vmlal_lane_s16(o14, r27, c_o32_1, 3);
                o14 = vmlsl_lane_s16(o14, r29, c_o32_1, 0);
                o14 = vmlal_lane_s16(o14, r31, c_o32_0, 0);
                y1[1][6] = vqrshrn_n_s32(vaddq_s32(e[1][14], o14), 7);
                y2[1][1] = vqrshrn_n_s32(vsubq_s32(e[1][14], o14), 7);

                /* o15[4-7] */
                int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
                o15 = vmlsl_lane_s16(o15, r3, c_o32_3, 2);
                o15 = vmlal_lane_s16(o15, r5, c_o32_3, 1);
                o15 = vmlsl_lane_s16(o15, r7, c_o32_3, 0);
                o15 = vmlal_lane_s16(o15, r9, c_o32_2, 3);
                o15 = vmlsl_lane_s16(o15, r11, c_o32_2, 2);
                o15 = vmlal_lane_s16(o15, r13, c_o32_2, 1);
                o15 = vmlsl_lane_s16(o15, r15, c_o32_2, 0);
                o15 = vmlal_lane_s16(o15, r17, c_o32_1, 3);
                o15 = vmlsl_lane_s16(o15, r19, c_o32_1, 2);
                o15 = vmlal_lane_s16(o15, r21, c_o32_1, 1);
                o15 = vmlsl_lane_s16(o15, r23, c_o32_1, 0);
                o15 = vmlal_lane_s16(o15, r25, c_o32_0, 3);
                o15 = vmlsl_lane_s16(o15, r27, c_o32_0, 2);
                o15 = vmlal_lane_s16(o15, r29, c_o32_0, 0);
                o15 = vmlsl_lane_s16(o15, r31, c_o32_0, 0);
                y1[1][7] = vqrshrn_n_s32(vaddq_s32(e[1][15], o15), 7);
                y2[1][0] = vqrshrn_n_s32(vsubq_s32(e[1][15], o15), 7);
            }
        }

        /* Transpose */
        {
            WORD32 cb = i;

            /* Chunk 0 (y0): store to pi2_tmp at col 0 */
            int16x4x2_t t0 = vtrn_s16(y0[0][0], y0[0][1]);
            int16x4x2_t t1 = vtrn_s16(y0[0][2], y0[0][3]);
            int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
            int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
            int16x4x2_t t2 = vtrn_s16(y0[0][4], y0[0][5]);
            int16x4x2_t t3 = vtrn_s16(y0[0][6], y0[0][7]);
            int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
            int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1])));

            /* Chunk 1 (y1): store to pi2_tmp at col 8 */
            int16x4x2_t t4 = vtrn_s16(y1[0][0], y1[0][1]);
            int16x4x2_t t5 = vtrn_s16(y1[0][2], y1[0][3]);
            int32x2x2_t u4 = vtrn_s32(vreinterpret_s32_s16(t4.val[0]), vreinterpret_s32_s16(t5.val[0]));
            int32x2x2_t u5 = vtrn_s32(vreinterpret_s32_s16(t4.val[1]), vreinterpret_s32_s16(t5.val[1]));
            int16x4x2_t t6 = vtrn_s16(y1[0][4], y1[0][5]);
            int16x4x2_t t7 = vtrn_s16(y1[0][6], y1[0][7]);
            int32x2x2_t u6 = vtrn_s32(vreinterpret_s32_s16(t6.val[0]), vreinterpret_s32_s16(t7.val[0]));
            int32x2x2_t u7 = vtrn_s32(vreinterpret_s32_s16(t6.val[1]), vreinterpret_s32_s16(t7.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u4.val[0]), vreinterpret_s16_s32(u6.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u5.val[0]), vreinterpret_s16_s32(u7.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u4.val[1]), vreinterpret_s16_s32(u6.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u5.val[1]), vreinterpret_s16_s32(u7.val[1])));

            /* Chunk 2 (y2): store to pi2_tmp at col 16 */
            int16x4x2_t t8 = vtrn_s16(y2[0][0], y2[0][1]);
            int16x4x2_t t9 = vtrn_s16(y2[0][2], y2[0][3]);
            int32x2x2_t u8 = vtrn_s32(vreinterpret_s32_s16(t8.val[0]), vreinterpret_s32_s16(t9.val[0]));
            int32x2x2_t u9 = vtrn_s32(vreinterpret_s32_s16(t8.val[1]), vreinterpret_s32_s16(t9.val[1]));
            int16x4x2_t ta = vtrn_s16(y2[0][4], y2[0][5]);
            int16x4x2_t tb = vtrn_s16(y2[0][6], y2[0][7]);
            int32x2x2_t ua = vtrn_s32(vreinterpret_s32_s16(ta.val[0]), vreinterpret_s32_s16(tb.val[0]));
            int32x2x2_t ub = vtrn_s32(vreinterpret_s32_s16(ta.val[1]), vreinterpret_s32_s16(tb.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u8.val[0]), vreinterpret_s16_s32(ua.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u9.val[0]), vreinterpret_s16_s32(ub.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u8.val[1]), vreinterpret_s16_s32(ua.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u9.val[1]), vreinterpret_s16_s32(ub.val[1])));

            /* Chunk 3 (y3): store to pi2_tmp at col 24 */
            int16x4x2_t tc = vtrn_s16(y3[0][0], y3[0][1]);
            int16x4x2_t td = vtrn_s16(y3[0][2], y3[0][3]);
            int32x2x2_t uc = vtrn_s32(vreinterpret_s32_s16(tc.val[0]), vreinterpret_s32_s16(td.val[0]));
            int32x2x2_t ud = vtrn_s32(vreinterpret_s32_s16(tc.val[1]), vreinterpret_s32_s16(td.val[1]));
            int16x4x2_t te = vtrn_s16(y3[0][4], y3[0][5]);
            int16x4x2_t tf = vtrn_s16(y3[0][6], y3[0][7]);
            int32x2x2_t ue = vtrn_s32(vreinterpret_s32_s16(te.val[0]), vreinterpret_s32_s16(tf.val[0]));
            int32x2x2_t uf = vtrn_s32(vreinterpret_s32_s16(te.val[1]), vreinterpret_s32_s16(tf.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(uc.val[0]), vreinterpret_s16_s32(ue.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(ud.val[0]), vreinterpret_s16_s32(uf.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(uc.val[1]), vreinterpret_s16_s32(ue.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(ud.val[1]), vreinterpret_s16_s32(uf.val[1])));
        }
        {
            WORD32 cb = i + 4;

            /* Chunk 0 (y0): store to pi2_tmp at col 0 */
            int16x4x2_t t0 = vtrn_s16(y0[1][0], y0[1][1]);
            int16x4x2_t t1 = vtrn_s16(y0[1][2], y0[1][3]);
            int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
            int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
            int16x4x2_t t2 = vtrn_s16(y0[1][4], y0[1][5]);
            int16x4x2_t t3 = vtrn_s16(y0[1][6], y0[1][7]);
            int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
            int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 0, vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1])));

            /* Chunk 1 (y1): store to pi2_tmp at col 8 */
            int16x4x2_t t4 = vtrn_s16(y1[1][0], y1[1][1]);
            int16x4x2_t t5 = vtrn_s16(y1[1][2], y1[1][3]);
            int32x2x2_t u4 = vtrn_s32(vreinterpret_s32_s16(t4.val[0]), vreinterpret_s32_s16(t5.val[0]));
            int32x2x2_t u5 = vtrn_s32(vreinterpret_s32_s16(t4.val[1]), vreinterpret_s32_s16(t5.val[1]));
            int16x4x2_t t6 = vtrn_s16(y1[1][4], y1[1][5]);
            int16x4x2_t t7 = vtrn_s16(y1[1][6], y1[1][7]);
            int32x2x2_t u6 = vtrn_s32(vreinterpret_s32_s16(t6.val[0]), vreinterpret_s32_s16(t7.val[0]));
            int32x2x2_t u7 = vtrn_s32(vreinterpret_s32_s16(t6.val[1]), vreinterpret_s32_s16(t7.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u4.val[0]), vreinterpret_s16_s32(u6.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u5.val[0]), vreinterpret_s16_s32(u7.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u4.val[1]), vreinterpret_s16_s32(u6.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 8, vcombine_s16(vreinterpret_s16_s32(u5.val[1]), vreinterpret_s16_s32(u7.val[1])));

            /* Chunk 2 (y2): store to pi2_tmp at col 16 */
            int16x4x2_t t8 = vtrn_s16(y2[1][0], y2[1][1]);
            int16x4x2_t t9 = vtrn_s16(y2[1][2], y2[1][3]);
            int32x2x2_t u8 = vtrn_s32(vreinterpret_s32_s16(t8.val[0]), vreinterpret_s32_s16(t9.val[0]));
            int32x2x2_t u9 = vtrn_s32(vreinterpret_s32_s16(t8.val[1]), vreinterpret_s32_s16(t9.val[1]));
            int16x4x2_t ta = vtrn_s16(y2[1][4], y2[1][5]);
            int16x4x2_t tb = vtrn_s16(y2[1][6], y2[1][7]);
            int32x2x2_t ua = vtrn_s32(vreinterpret_s32_s16(ta.val[0]), vreinterpret_s32_s16(tb.val[0]));
            int32x2x2_t ub = vtrn_s32(vreinterpret_s32_s16(ta.val[1]), vreinterpret_s32_s16(tb.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u8.val[0]), vreinterpret_s16_s32(ua.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u9.val[0]), vreinterpret_s16_s32(ub.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u8.val[1]), vreinterpret_s16_s32(ua.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 16, vcombine_s16(vreinterpret_s16_s32(u9.val[1]), vreinterpret_s16_s32(ub.val[1])));

            /* Chunk 3 (y3): store to pi2_tmp at col 24 */
            int16x4x2_t tc = vtrn_s16(y3[1][0], y3[1][1]);
            int16x4x2_t td = vtrn_s16(y3[1][2], y3[1][3]);
            int32x2x2_t uc = vtrn_s32(vreinterpret_s32_s16(tc.val[0]), vreinterpret_s32_s16(td.val[0]));
            int32x2x2_t ud = vtrn_s32(vreinterpret_s32_s16(tc.val[1]), vreinterpret_s32_s16(td.val[1]));
            int16x4x2_t te = vtrn_s16(y3[1][4], y3[1][5]);
            int16x4x2_t tf = vtrn_s16(y3[1][6], y3[1][7]);
            int32x2x2_t ue = vtrn_s32(vreinterpret_s32_s16(te.val[0]), vreinterpret_s32_s16(tf.val[0]));
            int32x2x2_t uf = vtrn_s32(vreinterpret_s32_s16(te.val[1]), vreinterpret_s32_s16(tf.val[1]));
            vst1q_s16(pi2_tmp + (cb + 0) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(uc.val[0]), vreinterpret_s16_s32(ue.val[0])));
            vst1q_s16(pi2_tmp + (cb + 1) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(ud.val[0]), vreinterpret_s16_s32(uf.val[0])));
            vst1q_s16(pi2_tmp + (cb + 2) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(uc.val[1]), vreinterpret_s16_s32(ue.val[1])));
            vst1q_s16(pi2_tmp + (cb + 3) * 32 + 24, vcombine_s16(vreinterpret_s16_s32(ud.val[1]), vreinterpret_s16_s32(uf.val[1])));
        }
    }

    /* Inverse Transform 2nd stage */
    int32x4_t neg_shift = vdupq_n_s32(-(20 - u1_bit_depth));
    int16x8_t clip_min = vdupq_n_s16(0);
    int16x8_t clip_max = vdupq_n_s16((1 << u1_bit_depth) - 1);

    for (WORD32 j = 0; j < trans_size; j += 4)
    {
        WORD32 col = j;
        int16x4_t y0[8], y1[8], y2[8], y3[8];

        /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
        if (zero_last28_rows_stg2)
        {
            int16x4_t r0 = vld1_s16(pi2_tmp + 0 * 32 + col);
            int16x4_t r1 = vld1_s16(pi2_tmp + 1 * 32 + col);
            int16x4_t r2 = vld1_s16(pi2_tmp + 2 * 32 + col);
            int16x4_t r3 = vld1_s16(pi2_tmp + 3 * 32 + col);
            int32x4_t ee = vshll_n_s16(r0, 6);
            /* eo0[0-3] */
            int32x4_t eo0 = vmull_lane_s16(r2, co0, 0);
            /* eo1[0-3] */
            int32x4_t eo1 = vmull_lane_s16(r2, co0, 1);
            /* eo2[0-3] */
            int32x4_t eo2 = vmull_lane_s16(r2, co0, 2);
            /* eo3[0-3] */
            int32x4_t eo3 = vmull_lane_s16(r2, co0, 3);
            /* eo4[0-3] */
            int32x4_t eo4 = vmull_lane_s16(r2, co1, 0);
            /* eo5[0-3] */
            int32x4_t eo5 = vmull_lane_s16(r2, co1, 1);
            /* eo6[0-3] */
            int32x4_t eo6 = vmull_lane_s16(r2, co1, 2);
            /* eo7[0-3] */
            int32x4_t eo7 = vmull_lane_s16(r2, co1, 3);

            /* o0[0-3] */
            int32x4_t e0 = vaddq_s32(ee, eo0);
            int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 0), r3, c_o32_0, 1);
            y0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
            y3[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));

            /* o1[0-3] */
            int32x4_t e1 = vaddq_s32(ee, eo1);
            int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 1), r3, c_o32_1, 0);
            y0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
            y3[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));

            /* o2[0-3] */
            int32x4_t e2 = vaddq_s32(ee, eo2);
            int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 2), r3, c_o32_1, 3);
            y0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
            y3[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));

            /* o3[0-3] */
            int32x4_t e3 = vaddq_s32(ee, eo3);
            int32x4_t o3 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 3), r3, c_o32_2, 2);
            y0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
            y3[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));

            /* o4[0-3] */
            int32x4_t e4 = vaddq_s32(ee, eo4);
            int32x4_t o4 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_1, 0), r3, c_o32_3, 1);
            y0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
            y3[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));

            /* o5[0-3] */
            int32x4_t e5 = vaddq_s32(ee, eo5);
            int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 1), r3, c_o32_3, 3);
            y0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
            y3[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));

            /* o6[0-3] */
            int32x4_t e6 = vaddq_s32(ee, eo6);
            int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 2), r3, c_o32_3, 0);
            y0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
            y3[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));

            /* o7[0-3] */
            int32x4_t e7 = vaddq_s32(ee, eo7);
            int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 3), r3, c_o32_2, 1);
            y0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
            y3[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));

            /* o8[0-3] */
            int32x4_t e8 = vsubq_s32(ee, eo7);
            int32x4_t o8 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 0), r3, c_o32_1, 2);
            y1[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e8, o8), neg_shift));
            y2[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e8, o8), neg_shift));

            /* o9[0-3] */
            int32x4_t e9 = vsubq_s32(ee, eo6);
            int32x4_t o9 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 1), r3, c_o32_0, 3);
            y1[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e9, o9), neg_shift));
            y2[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e9, o9), neg_shift));

            /* o10[0-3] */
            int32x4_t e10 = vsubq_s32(ee, eo5);
            int32x4_t o10 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 2), r3, c_o32_0, 1);
            y1[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e10, o10), neg_shift));
            y2[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e10, o10), neg_shift));

            /* o11[0-3] */
            int32x4_t e11 = vsubq_s32(ee, eo4);
            int32x4_t o11 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 3), r3, c_o32_0, 2);
            y1[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e11, o11), neg_shift));
            y2[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e11, o11), neg_shift));

            /* o12[0-3] */
            int32x4_t e12 = vsubq_s32(ee, eo3);
            int32x4_t o12 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 0), r3, c_o32_1, 1);
            y1[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e12, o12), neg_shift));
            y2[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e12, o12), neg_shift));

            /* o13[0-3] */
            int32x4_t e13 = vsubq_s32(ee, eo2);
            int32x4_t o13 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 1), r3, c_o32_2, 0);
            y1[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e13, o13), neg_shift));
            y2[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e13, o13), neg_shift));

            /* o14[0-3] */
            int32x4_t e14 = vsubq_s32(ee, eo1);
            int32x4_t o14 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 2), r3, c_o32_2, 3);
            y1[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e14, o14), neg_shift));
            y2[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e14, o14), neg_shift));

            /* o15[0-3] */
            int32x4_t e15 = vsubq_s32(ee, eo0);
            int32x4_t o15 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 3), r3, c_o32_3, 2);
            y1[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e15, o15), neg_shift));
            y2[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e15, o15), neg_shift));
        }
        else if (zero_last24_rows_stg2)
        {
            int16x4_t r0 = vld1_s16(pi2_tmp + 0 * 32 + col);
            int16x4_t r1 = vld1_s16(pi2_tmp + 1 * 32 + col);
            int16x4_t r2 = vld1_s16(pi2_tmp + 2 * 32 + col);
            int16x4_t r3 = vld1_s16(pi2_tmp + 3 * 32 + col);
            int16x4_t r4 = vld1_s16(pi2_tmp + 4 * 32 + col);
            int16x4_t r5 = vld1_s16(pi2_tmp + 5 * 32 + col);
            int16x4_t r6 = vld1_s16(pi2_tmp + 6 * 32 + col);
            int16x4_t r7 = vld1_s16(pi2_tmp + 7 * 32 + col);
            int32x4_t eee = vshll_n_s16(r0, 6);
            int32x4_t ee[8];
            /* eo */
            /* eo0[0-3] */
            int32x4_t eo0 = vmlal_lane_s16(vmull_lane_s16(r2, co0, 0), r6, co0, 1);
            /* eo1[0-3] */
            int32x4_t eo1 = vmlal_lane_s16(vmull_lane_s16(r2, co0, 1), r6, co1, 0);
            /* eo2[0-3] */
            int32x4_t eo2 = vmlal_lane_s16(vmull_lane_s16(r2, co0, 2), r6, co1, 3);
            /* eo3[0-3] */
            int32x4_t eo3 = vmlsl_lane_s16(vmull_lane_s16(r2, co0, 3), r6, co1, 1);
            /* eo4[0-3] */
            int32x4_t eo4 = vmlsl_lane_s16(vmull_lane_s16(r2, co1, 0), r6, co0, 2);
            /* eo5[0-3] */
            int32x4_t eo5 = vmlsl_lane_s16(vmull_lane_s16(r2, co1, 1), r6, co0, 0);
            /* eo6[0-3] */
            int32x4_t eo6 = vmlsl_lane_s16(vmull_lane_s16(r2, co1, 2), r6, co0, 3);
            /* eo7[0-3] */
            int32x4_t eo7 = vmlsl_lane_s16(vmull_lane_s16(r2, co1, 3), r6, co1, 2);

            /* eeo */
            /* eeo0[0-3] */
            int32x4_t eeo0 = vmull_lane_s16(r4, ce1, 0);
            ee[0] = vaddq_s32(eee, eeo0);
            ee[7] = vsubq_s32(eee, eeo0);
            /* eeo1[0-3] */
            int32x4_t eeo1 = vmull_lane_s16(r4, ce1, 1);
            ee[1] = vaddq_s32(eee, eeo1);
            ee[6] = vsubq_s32(eee, eeo1);
            /* eeo2[0-3] */
            int32x4_t eeo2 = vmull_lane_s16(r4, ce1, 2);
            ee[2] = vaddq_s32(eee, eeo2);
            ee[5] = vsubq_s32(eee, eeo2);
            /* eeo3[0-3] */
            int32x4_t eeo3 = vmull_lane_s16(r4, ce1, 3);
            ee[3] = vaddq_s32(eee, eeo3);
            ee[4] = vsubq_s32(eee, eeo3);
            /* o0[0-3] */
            int32x4_t e0 = vaddq_s32(ee[0], eo0);
            int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
            o0 = vmlal_lane_s16(o0, r3, c_o32_0, 1);
            o0 = vmlal_lane_s16(o0, r5, c_o32_0, 2);
            o0 = vmlal_lane_s16(o0, r7, c_o32_0, 3);
            y0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
            y3[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));

            /* o1[0-3] */
            int32x4_t e1 = vaddq_s32(ee[1], eo1);
            int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
            o1 = vmlal_lane_s16(o1, r3, c_o32_1, 0);
            o1 = vmlal_lane_s16(o1, r5, c_o32_1, 3);
            o1 = vmlal_lane_s16(o1, r7, c_o32_2, 2);
            y0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
            y3[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));

            /* o2[0-3] */
            int32x4_t e2 = vaddq_s32(ee[2], eo2);
            int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
            o2 = vmlal_lane_s16(o2, r3, c_o32_1, 3);
            o2 = vmlal_lane_s16(o2, r5, c_o32_3, 0);
            o2 = vmlsl_lane_s16(o2, r7, c_o32_3, 2);
            y0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
            y3[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));

            /* o3[0-3] */
            int32x4_t e3 = vaddq_s32(ee[3], eo3);
            int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
            o3 = vmlal_lane_s16(o3, r3, c_o32_2, 2);
            o3 = vmlsl_lane_s16(o3, r5, c_o32_3, 2);
            o3 = vmlsl_lane_s16(o3, r7, c_o32_1, 3);
            y0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
            y3[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));

            /* o4[0-3] */
            int32x4_t e4 = vaddq_s32(ee[4], eo4);
            int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
            o4 = vmlal_lane_s16(o4, r3, c_o32_3, 1);
            o4 = vmlsl_lane_s16(o4, r5, c_o32_2, 1);
            o4 = vmlsl_lane_s16(o4, r7, c_o32_0, 0);
            y0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
            y3[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));

            /* o5[0-3] */
            int32x4_t e5 = vaddq_s32(ee[5], eo5);
            int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
            o5 = vmlsl_lane_s16(o5, r3, c_o32_3, 3);
            o5 = vmlsl_lane_s16(o5, r5, c_o32_1, 0);
            o5 = vmlsl_lane_s16(o5, r7, c_o32_1, 2);
            y0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
            y3[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));

            /* o6[0-3] */
            int32x4_t e6 = vaddq_s32(ee[6], eo6);
            int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
            o6 = vmlsl_lane_s16(o6, r3, c_o32_3, 0);
            o6 = vmlsl_lane_s16(o6, r5, c_o32_0, 0);
            o6 = vmlsl_lane_s16(o6, r7, c_o32_3, 1);
            y0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
            y3[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));

            /* o7[0-3] */
            int32x4_t e7 = vaddq_s32(ee[7], eo7);
            int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
            o7 = vmlsl_lane_s16(o7, r3, c_o32_2, 1);
            o7 = vmlsl_lane_s16(o7, r5, c_o32_1, 1);
            o7 = vmlal_lane_s16(o7, r7, c_o32_2, 3);
            y0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
            y3[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));

            /* o8[0-3] */
            int32x4_t e8 = vsubq_s32(ee[7], eo7);
            int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
            o8 = vmlsl_lane_s16(o8, r3, c_o32_1, 2);
            o8 = vmlsl_lane_s16(o8, r5, c_o32_2, 2);
            o8 = vmlal_lane_s16(o8, r7, c_o32_1, 0);
            y1[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e8, o8), neg_shift));
            y2[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e8, o8), neg_shift));

            /* o9[0-3] */
            int32x4_t e9 = vsubq_s32(ee[6], eo6);
            int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
            o9 = vmlsl_lane_s16(o9, r3, c_o32_0, 3);
            o9 = vmlsl_lane_s16(o9, r5, c_o32_3, 3);
            o9 = vmlal_lane_s16(o9, r7, c_o32_0, 2);
            y1[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e9, o9), neg_shift));
            y2[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e9, o9), neg_shift));

            /* o10[0-3] */
            int32x4_t e10 = vsubq_s32(ee[5], eo5);
            int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
            o10 = vmlsl_lane_s16(o10, r3, c_o32_0, 1);
            o10 = vmlal_lane_s16(o10, r5, c_o32_2, 3);
            o10 = vmlal_lane_s16(o10, r7, c_o32_2, 1);
            y1[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e10, o10), neg_shift));
            y2[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e10, o10), neg_shift));

            /* o11[0-3] */
            int32x4_t e11 = vsubq_s32(ee[4], eo4);
            int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
            o11 = vmlsl_lane_s16(o11, r3, c_o32_0, 2);
            o11 = vmlal_lane_s16(o11, r5, c_o32_1, 2);
            o11 = vmlsl_lane_s16(o11, r7, c_o32_3, 3);
            y1[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e11, o11), neg_shift));
            y2[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e11, o11), neg_shift));

            /* o12[0-3] */
            int32x4_t e12 = vsubq_s32(ee[3], eo3);
            int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
            o12 = vmlsl_lane_s16(o12, r3, c_o32_1, 1);
            o12 = vmlal_lane_s16(o12, r5, c_o32_0, 0);
            o12 = vmlsl_lane_s16(o12, r7, c_o32_2, 0);
            y1[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e12, o12), neg_shift));
            y2[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e12, o12), neg_shift));

            /* o13[0-3] */
            int32x4_t e13 = vsubq_s32(ee[2], eo2);
            int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
            o13 = vmlsl_lane_s16(o13, r3, c_o32_2, 0);
            o13 = vmlal_lane_s16(o13, r5, c_o32_0, 3);
            o13 = vmlsl_lane_s16(o13, r7, c_o32_0, 0);
            y1[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e13, o13), neg_shift));
            y2[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e13, o13), neg_shift));

            /* o14[0-3] */
            int32x4_t e14 = vsubq_s32(ee[1], eo1);
            int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
            o14 = vmlsl_lane_s16(o14, r3, c_o32_2, 3);
            o14 = vmlal_lane_s16(o14, r5, c_o32_2, 0);
            o14 = vmlsl_lane_s16(o14, r7, c_o32_1, 1);
            y1[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e14, o14), neg_shift));
            y2[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e14, o14), neg_shift));

            /* o15[0-3] */
            int32x4_t e15 = vsubq_s32(ee[0], eo0);
            int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
            o15 = vmlsl_lane_s16(o15, r3, c_o32_3, 2);
            o15 = vmlal_lane_s16(o15, r5, c_o32_3, 1);
            o15 = vmlsl_lane_s16(o15, r7, c_o32_3, 0);
            y1[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e15, o15), neg_shift));
            y2[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e15, o15), neg_shift));
        }
        else
        {
            int32x4_t ee[8];
            int32x4_t eo[8];
            {
                int16x4_t r0 = vld1_s16(pi2_tmp + 0 * 32 + col);
                int16x4_t r16 = vld1_s16(pi2_tmp + 16 * 32 + col);
                int16x4_t r8 = vld1_s16(pi2_tmp + 8 * 32 + col);
                int16x4_t r24 = vld1_s16(pi2_tmp + 24 * 32 + col);
                int16x4_t r4 = vld1_s16(pi2_tmp + 4 * 32 + col);
                int16x4_t r12 = vld1_s16(pi2_tmp + 12 * 32 + col);
                int16x4_t r20 = vld1_s16(pi2_tmp + 20 * 32 + col);
                int16x4_t r28 = vld1_s16(pi2_tmp + 28 * 32 + col);
                int16x4_t r2 = vld1_s16(pi2_tmp + 2 * 32 + col);
                int16x4_t r6 = vld1_s16(pi2_tmp + 6 * 32 + col);
                int16x4_t r10 = vld1_s16(pi2_tmp + 10 * 32 + col);
                int16x4_t r14 = vld1_s16(pi2_tmp + 14 * 32 + col);
                int16x4_t r18 = vld1_s16(pi2_tmp + 18 * 32 + col);
                int16x4_t r22 = vld1_s16(pi2_tmp + 22 * 32 + col);
                int16x4_t r26 = vld1_s16(pi2_tmp + 26 * 32 + col);
                int16x4_t r30 = vld1_s16(pi2_tmp + 30 * 32 + col);
                int32x4_t r0_shl = vshll_n_s16(r0, 6);
                int32x4_t r16_shl = vshll_n_s16(r16, 6);
                int32x4_t eeee0 = vaddq_s32(r0_shl, r16_shl);
                int32x4_t eeee1 = vsubq_s32(r0_shl, r16_shl);
                int32x4_t eeeo0 = vmlal_lane_s16(vmull_lane_s16(r8, ce0, 0), r24, ce0, 1);
                int32x4_t eeeo1 = vmlsl_lane_s16(vmull_lane_s16(r8, ce0, 1), r24, ce0, 0);
                /* eee[0] = eeee[0] + eeeo[0]; */
                int32x4_t eee0 = vaddq_s32(eeee0, eeeo0);
                /* eee[3] = eeee[0] - eeeo[0]; */
                int32x4_t eee3 = vsubq_s32(eeee0, eeeo0);
                /* eee[1] = eeee[1] + eeeo[1]; */
                int32x4_t eee1 = vaddq_s32(eeee1, eeeo1);
                /* eee[2] = eeee[1] - eeeo[1]; */
                int32x4_t eee2 = vsubq_s32(eeee1, eeeo1);

                /* eeo */
                /* eeo0[0-3] */
                int32x4_t eeo0 = vmull_lane_s16(r4, ce1, 0);
                eeo0 = vmlal_lane_s16(eeo0, r12, ce1, 1);
                eeo0 = vmlal_lane_s16(eeo0, r20, ce1, 2);
                eeo0 = vmlal_lane_s16(eeo0, r28, ce1, 3);
                /* eeo1[0-3] */
                int32x4_t eeo1 = vmull_lane_s16(r4, ce1, 1);
                eeo1 = vmlsl_lane_s16(eeo1, r12, ce1, 3);
                eeo1 = vmlsl_lane_s16(eeo1, r20, ce1, 0);
                eeo1 = vmlsl_lane_s16(eeo1, r28, ce1, 2);
                /* eeo2[0-3] */
                int32x4_t eeo2 = vmull_lane_s16(r4, ce1, 2);
                eeo2 = vmlsl_lane_s16(eeo2, r12, ce1, 0);
                eeo2 = vmlal_lane_s16(eeo2, r20, ce1, 3);
                eeo2 = vmlal_lane_s16(eeo2, r28, ce1, 1);
                /* eeo3[0-3] */
                int32x4_t eeo3 = vmull_lane_s16(r4, ce1, 3);
                eeo3 = vmlsl_lane_s16(eeo3, r12, ce1, 2);
                eeo3 = vmlal_lane_s16(eeo3, r20, ce1, 1);
                eeo3 = vmlsl_lane_s16(eeo3, r28, ce1, 0);

                ee[0] = vaddq_s32(eee0, eeo0);
                ee[7] = vsubq_s32(eee0, eeo0);
                ee[1] = vaddq_s32(eee1, eeo1);
                ee[6] = vsubq_s32(eee1, eeo1);
                ee[2] = vaddq_s32(eee2, eeo2);
                ee[5] = vsubq_s32(eee2, eeo2);
                ee[3] = vaddq_s32(eee3, eeo3);
                ee[4] = vsubq_s32(eee3, eeo3);

                /* eo */
                /* eo0[0-3] */
                eo[0] = vmull_lane_s16(r2, co0, 0);
                eo[0] = vmlal_lane_s16(eo[0], r6, co0, 1);
                eo[0] = vmlal_lane_s16(eo[0], r10, co0, 2);
                eo[0] = vmlal_lane_s16(eo[0], r14, co0, 3);
                eo[0] = vmlal_lane_s16(eo[0], r18, co1, 0);
                eo[0] = vmlal_lane_s16(eo[0], r22, co1, 1);
                eo[0] = vmlal_lane_s16(eo[0], r26, co1, 2);
                eo[0] = vmlal_lane_s16(eo[0], r30, co1, 3);

                /* eo1[0-3] */
                eo[1] = vmull_lane_s16(r2, co0, 1);
                eo[1] = vmlal_lane_s16(eo[1], r6, co1, 0);
                eo[1] = vmlal_lane_s16(eo[1], r10, co1, 3);
                eo[1] = vmlsl_lane_s16(eo[1], r14, co1, 1);
                eo[1] = vmlsl_lane_s16(eo[1], r18, co0, 2);
                eo[1] = vmlsl_lane_s16(eo[1], r22, co0, 0);
                eo[1] = vmlsl_lane_s16(eo[1], r26, co0, 3);
                eo[1] = vmlsl_lane_s16(eo[1], r30, co1, 2);

                /* eo2[0-3] */
                eo[2] = vmull_lane_s16(r2, co0, 2);
                eo[2] = vmlal_lane_s16(eo[2], r6, co1, 3);
                eo[2] = vmlsl_lane_s16(eo[2], r10, co0, 3);
                eo[2] = vmlsl_lane_s16(eo[2], r14, co0, 1);
                eo[2] = vmlsl_lane_s16(eo[2], r18, co1, 2);
                eo[2] = vmlal_lane_s16(eo[2], r22, co1, 0);
                eo[2] = vmlal_lane_s16(eo[2], r26, co0, 0);
                eo[2] = vmlal_lane_s16(eo[2], r30, co1, 1);

                /* eo3[0-3] */
                eo[3] = vmull_lane_s16(r2, co0, 3);
                eo[3] = vmlsl_lane_s16(eo[3], r6, co1, 1);
                eo[3] = vmlsl_lane_s16(eo[3], r10, co0, 1);
                eo[3] = vmlal_lane_s16(eo[3], r14, co1, 3);
                eo[3] = vmlal_lane_s16(eo[3], r18, co0, 0);
                eo[3] = vmlal_lane_s16(eo[3], r22, co1, 2);
                eo[3] = vmlsl_lane_s16(eo[3], r26, co0, 2);
                eo[3] = vmlsl_lane_s16(eo[3], r30, co1, 0);

                /* eo4[0-3] */
                eo[4] = vmull_lane_s16(r2, co1, 0);
                eo[4] = vmlsl_lane_s16(eo[4], r6, co0, 2);
                eo[4] = vmlsl_lane_s16(eo[4], r10, co1, 2);
                eo[4] = vmlal_lane_s16(eo[4], r14, co0, 0);
                eo[4] = vmlsl_lane_s16(eo[4], r18, co1, 3);
                eo[4] = vmlsl_lane_s16(eo[4], r22, co0, 1);
                eo[4] = vmlal_lane_s16(eo[4], r26, co1, 1);
                eo[4] = vmlal_lane_s16(eo[4], r30, co0, 3);

                /* eo5[0-3] */
                eo[5] = vmull_lane_s16(r2, co1, 1);
                eo[5] = vmlsl_lane_s16(eo[5], r6, co0, 0);
                eo[5] = vmlal_lane_s16(eo[5], r10, co1, 0);
                eo[5] = vmlal_lane_s16(eo[5], r14, co1, 2);
                eo[5] = vmlsl_lane_s16(eo[5], r18, co0, 1);
                eo[5] = vmlal_lane_s16(eo[5], r22, co0, 3);
                eo[5] = vmlal_lane_s16(eo[5], r26, co1, 3);
                eo[5] = vmlsl_lane_s16(eo[5], r30, co0, 2);

                /* eo6[0-3] */
                eo[6] = vmull_lane_s16(r2, co1, 2);
                eo[6] = vmlsl_lane_s16(eo[6], r6, co0, 3);
                eo[6] = vmlal_lane_s16(eo[6], r10, co0, 0);
                eo[6] = vmlsl_lane_s16(eo[6], r14, co0, 2);
                eo[6] = vmlal_lane_s16(eo[6], r18, co1, 1);
                eo[6] = vmlal_lane_s16(eo[6], r22, co1, 3);
                eo[6] = vmlsl_lane_s16(eo[6], r26, co1, 0);
                eo[6] = vmlal_lane_s16(eo[6], r30, co0, 1);

                /* eo7[0-3] */
                eo[7] = vmull_lane_s16(r2, co1, 3);
                eo[7] = vmlsl_lane_s16(eo[7], r6, co1, 2);
                eo[7] = vmlal_lane_s16(eo[7], r10, co1, 1);
                eo[7] = vmlsl_lane_s16(eo[7], r14, co1, 0);
                eo[7] = vmlal_lane_s16(eo[7], r18, co0, 3);
                eo[7] = vmlsl_lane_s16(eo[7], r22, co0, 2);
                eo[7] = vmlal_lane_s16(eo[7], r26, co0, 1);
                eo[7] = vmlsl_lane_s16(eo[7], r30, co0, 0);
            }
            /* Odd stage */
            int16x4_t r1 = vld1_s16(pi2_tmp + 1 * 32 + col);
            int16x4_t r3 = vld1_s16(pi2_tmp + 3 * 32 + col);
            int16x4_t r5 = vld1_s16(pi2_tmp + 5 * 32 + col);
            int16x4_t r7 = vld1_s16(pi2_tmp + 7 * 32 + col);
            int16x4_t r9 = vld1_s16(pi2_tmp + 9 * 32 + col);
            int16x4_t r11 = vld1_s16(pi2_tmp + 11 * 32 + col);
            int16x4_t r13 = vld1_s16(pi2_tmp + 13 * 32 + col);
            int16x4_t r15 = vld1_s16(pi2_tmp + 15 * 32 + col);
            int16x4_t r17 = vld1_s16(pi2_tmp + 17 * 32 + col);
            int16x4_t r19 = vld1_s16(pi2_tmp + 19 * 32 + col);
            int16x4_t r21 = vld1_s16(pi2_tmp + 21 * 32 + col);
            int16x4_t r23 = vld1_s16(pi2_tmp + 23 * 32 + col);
            int16x4_t r25 = vld1_s16(pi2_tmp + 25 * 32 + col);
            int16x4_t r27 = vld1_s16(pi2_tmp + 27 * 32 + col);
            int16x4_t r29 = vld1_s16(pi2_tmp + 29 * 32 + col);
            int16x4_t r31 = vld1_s16(pi2_tmp + 31 * 32 + col);
            /* o0[0-3] */
            int32x4_t e0 = vaddq_s32(ee[0], eo[0]);
            int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
            o0 = vmlal_lane_s16(o0, r3, c_o32_0, 1);
            o0 = vmlal_lane_s16(o0, r5, c_o32_0, 2);
            o0 = vmlal_lane_s16(o0, r7, c_o32_0, 3);
            o0 = vmlal_lane_s16(o0, r9, c_o32_1, 0);
            o0 = vmlal_lane_s16(o0, r11, c_o32_1, 1);
            o0 = vmlal_lane_s16(o0, r13, c_o32_1, 2);
            o0 = vmlal_lane_s16(o0, r15, c_o32_1, 3);
            o0 = vmlal_lane_s16(o0, r17, c_o32_2, 0);
            o0 = vmlal_lane_s16(o0, r19, c_o32_2, 1);
            o0 = vmlal_lane_s16(o0, r21, c_o32_2, 2);
            o0 = vmlal_lane_s16(o0, r23, c_o32_2, 3);
            o0 = vmlal_lane_s16(o0, r25, c_o32_3, 0);
            o0 = vmlal_lane_s16(o0, r27, c_o32_3, 1);
            o0 = vmlal_lane_s16(o0, r29, c_o32_3, 2);
            o0 = vmlal_lane_s16(o0, r31, c_o32_3, 3);
            y0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
            y3[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));

            /* o1[0-3] */
            int32x4_t e1 = vaddq_s32(ee[1], eo[1]);
            int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
            o1 = vmlal_lane_s16(o1, r3, c_o32_1, 0);
            o1 = vmlal_lane_s16(o1, r5, c_o32_1, 3);
            o1 = vmlal_lane_s16(o1, r7, c_o32_2, 2);
            o1 = vmlal_lane_s16(o1, r9, c_o32_3, 1);
            o1 = vmlsl_lane_s16(o1, r11, c_o32_3, 3);
            o1 = vmlsl_lane_s16(o1, r13, c_o32_3, 0);
            o1 = vmlsl_lane_s16(o1, r15, c_o32_2, 1);
            o1 = vmlsl_lane_s16(o1, r17, c_o32_1, 2);
            o1 = vmlsl_lane_s16(o1, r19, c_o32_0, 3);
            o1 = vmlsl_lane_s16(o1, r21, c_o32_0, 0);
            o1 = vmlsl_lane_s16(o1, r23, c_o32_0, 2);
            o1 = vmlsl_lane_s16(o1, r25, c_o32_1, 1);
            o1 = vmlsl_lane_s16(o1, r27, c_o32_2, 0);
            o1 = vmlsl_lane_s16(o1, r29, c_o32_2, 3);
            o1 = vmlsl_lane_s16(o1, r31, c_o32_3, 2);
            y0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
            y3[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));

            /* o2[0-3] */
            int32x4_t e2 = vaddq_s32(ee[2], eo[2]);
            int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
            o2 = vmlal_lane_s16(o2, r3, c_o32_1, 3);
            o2 = vmlal_lane_s16(o2, r5, c_o32_3, 0);
            o2 = vmlsl_lane_s16(o2, r7, c_o32_3, 2);
            o2 = vmlsl_lane_s16(o2, r9, c_o32_2, 1);
            o2 = vmlsl_lane_s16(o2, r11, c_o32_1, 0);
            o2 = vmlsl_lane_s16(o2, r13, c_o32_0, 0);
            o2 = vmlsl_lane_s16(o2, r15, c_o32_1, 1);
            o2 = vmlsl_lane_s16(o2, r17, c_o32_2, 2);
            o2 = vmlsl_lane_s16(o2, r19, c_o32_3, 3);
            o2 = vmlal_lane_s16(o2, r21, c_o32_2, 3);
            o2 = vmlal_lane_s16(o2, r23, c_o32_1, 2);
            o2 = vmlal_lane_s16(o2, r25, c_o32_0, 0);
            o2 = vmlal_lane_s16(o2, r27, c_o32_0, 3);
            o2 = vmlal_lane_s16(o2, r29, c_o32_2, 0);
            o2 = vmlal_lane_s16(o2, r31, c_o32_3, 1);
            y0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
            y3[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));

            /* o3[0-3] */
            int32x4_t e3 = vaddq_s32(ee[3], eo[3]);
            int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
            o3 = vmlal_lane_s16(o3, r3, c_o32_2, 2);
            o3 = vmlsl_lane_s16(o3, r5, c_o32_3, 2);
            o3 = vmlsl_lane_s16(o3, r7, c_o32_1, 3);
            o3 = vmlsl_lane_s16(o3, r9, c_o32_0, 0);
            o3 = vmlsl_lane_s16(o3, r11, c_o32_1, 2);
            o3 = vmlsl_lane_s16(o3, r13, c_o32_3, 1);
            o3 = vmlal_lane_s16(o3, r15, c_o32_2, 3);
            o3 = vmlal_lane_s16(o3, r17, c_o32_1, 0);
            o3 = vmlal_lane_s16(o3, r19, c_o32_0, 2);
            o3 = vmlal_lane_s16(o3, r21, c_o32_2, 1);
            o3 = vmlsl_lane_s16(o3, r23, c_o32_3, 3);
            o3 = vmlsl_lane_s16(o3, r25, c_o32_2, 0);
            o3 = vmlsl_lane_s16(o3, r27, c_o32_0, 0);
            o3 = vmlsl_lane_s16(o3, r29, c_o32_1, 1);
            o3 = vmlsl_lane_s16(o3, r31, c_o32_3, 0);
            y0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
            y3[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));

            /* o4[0-3] */
            int32x4_t e4 = vaddq_s32(ee[4], eo[4]);
            int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
            o4 = vmlal_lane_s16(o4, r3, c_o32_3, 1);
            o4 = vmlsl_lane_s16(o4, r5, c_o32_2, 1);
            o4 = vmlsl_lane_s16(o4, r7, c_o32_0, 0);
            o4 = vmlsl_lane_s16(o4, r9, c_o32_2, 0);
            o4 = vmlal_lane_s16(o4, r11, c_o32_3, 2);
            o4 = vmlal_lane_s16(o4, r13, c_o32_1, 1);
            o4 = vmlal_lane_s16(o4, r15, c_o32_0, 3);
            o4 = vmlal_lane_s16(o4, r17, c_o32_3, 0);
            o4 = vmlsl_lane_s16(o4, r19, c_o32_2, 2);
            o4 = vmlsl_lane_s16(o4, r21, c_o32_0, 0);
            o4 = vmlsl_lane_s16(o4, r23, c_o32_1, 3);
            o4 = vmlal_lane_s16(o4, r25, c_o32_3, 3);
            o4 = vmlal_lane_s16(o4, r27, c_o32_1, 2);
            o4 = vmlal_lane_s16(o4, r29, c_o32_0, 2);
            o4 = vmlal_lane_s16(o4, r31, c_o32_2, 3);
            y0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
            y3[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));

            /* o5[0-3] */
            int32x4_t e5 = vaddq_s32(ee[5], eo[5]);
            int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
            o5 = vmlsl_lane_s16(o5, r3, c_o32_3, 3);
            o5 = vmlsl_lane_s16(o5, r5, c_o32_1, 0);
            o5 = vmlsl_lane_s16(o5, r7, c_o32_1, 2);
            o5 = vmlal_lane_s16(o5, r9, c_o32_3, 2);
            o5 = vmlal_lane_s16(o5, r11, c_o32_0, 3);
            o5 = vmlal_lane_s16(o5, r13, c_o32_1, 3);
            o5 = vmlsl_lane_s16(o5, r15, c_o32_3, 1);
            o5 = vmlsl_lane_s16(o5, r17, c_o32_0, 2);
            o5 = vmlsl_lane_s16(o5, r19, c_o32_2, 0);
            o5 = vmlal_lane_s16(o5, r21, c_o32_3, 0);
            o5 = vmlal_lane_s16(o5, r23, c_o32_0, 0);
            o5 = vmlal_lane_s16(o5, r25, c_o32_2, 1);
            o5 = vmlsl_lane_s16(o5, r27, c_o32_2, 3);
            o5 = vmlsl_lane_s16(o5, r29, c_o32_0, 0);
            o5 = vmlsl_lane_s16(o5, r31, c_o32_2, 2);
            y0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
            y3[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));

            /* o6[0-3] */
            int32x4_t e6 = vaddq_s32(ee[6], eo[6]);
            int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
            o6 = vmlsl_lane_s16(o6, r3, c_o32_3, 0);
            o6 = vmlsl_lane_s16(o6, r5, c_o32_0, 0);
            o6 = vmlsl_lane_s16(o6, r7, c_o32_3, 1);
            o6 = vmlal_lane_s16(o6, r9, c_o32_1, 1);
            o6 = vmlal_lane_s16(o6, r11, c_o32_1, 3);
            o6 = vmlsl_lane_s16(o6, r13, c_o32_2, 3);
            o6 = vmlsl_lane_s16(o6, r15, c_o32_0, 0);
            o6 = vmlsl_lane_s16(o6, r17, c_o32_3, 2);
            o6 = vmlal_lane_s16(o6, r19, c_o32_1, 0);
            o6 = vmlal_lane_s16(o6, r21, c_o32_2, 0);
            o6 = vmlsl_lane_s16(o6, r23, c_o32_2, 2);
            o6 = vmlsl_lane_s16(o6, r25, c_o32_0, 2);
            o6 = vmlsl_lane_s16(o6, r27, c_o32_3, 3);
            o6 = vmlal_lane_s16(o6, r29, c_o32_0, 3);
            o6 = vmlal_lane_s16(o6, r31, c_o32_2, 1);
            y0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
            y3[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));

            /* o7[0-3] */
            int32x4_t e7 = vaddq_s32(ee[7], eo[7]);
            int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
            o7 = vmlsl_lane_s16(o7, r3, c_o32_2, 1);
            o7 = vmlsl_lane_s16(o7, r5, c_o32_1, 1);
            o7 = vmlal_lane_s16(o7, r7, c_o32_2, 3);
            o7 = vmlal_lane_s16(o7, r9, c_o32_0, 3);
            o7 = vmlsl_lane_s16(o7, r11, c_o32_3, 1);
            o7 = vmlsl_lane_s16(o7, r13, c_o32_0, 0);
            o7 = vmlal_lane_s16(o7, r15, c_o32_3, 3);
            o7 = vmlal_lane_s16(o7, r17, c_o32_0, 0);
            o7 = vmlal_lane_s16(o7, r19, c_o32_3, 2);
            o7 = vmlsl_lane_s16(o7, r21, c_o32_0, 2);
            o7 = vmlsl_lane_s16(o7, r23, c_o32_3, 0);
            o7 = vmlal_lane_s16(o7, r25, c_o32_1, 0);
            o7 = vmlal_lane_s16(o7, r27, c_o32_2, 2);
            o7 = vmlsl_lane_s16(o7, r29, c_o32_1, 2);
            o7 = vmlsl_lane_s16(o7, r31, c_o32_2, 0);
            y0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
            y3[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));

            /* o8[0-3] */
            int32x4_t e8 = vsubq_s32(ee[7], eo[7]);
            int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
            o8 = vmlsl_lane_s16(o8, r3, c_o32_1, 2);
            o8 = vmlsl_lane_s16(o8, r5, c_o32_2, 2);
            o8 = vmlal_lane_s16(o8, r7, c_o32_1, 0);
            o8 = vmlal_lane_s16(o8, r9, c_o32_3, 0);
            o8 = vmlsl_lane_s16(o8, r11, c_o32_0, 2);
            o8 = vmlsl_lane_s16(o8, r13, c_o32_3, 2);
            o8 = vmlal_lane_s16(o8, r15, c_o32_0, 0);
            o8 = vmlsl_lane_s16(o8, r17, c_o32_3, 3);
            o8 = vmlsl_lane_s16(o8, r19, c_o32_0, 0);
            o8 = vmlal_lane_s16(o8, r21, c_o32_3, 1);
            o8 = vmlal_lane_s16(o8, r23, c_o32_0, 3);
            o8 = vmlsl_lane_s16(o8, r25, c_o32_2, 3);
            o8 = vmlsl_lane_s16(o8, r27, c_o32_1, 1);
            o8 = vmlal_lane_s16(o8, r29, c_o32_2, 1);
            o8 = vmlal_lane_s16(o8, r31, c_o32_1, 3);
            y1[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e8, o8), neg_shift));
            y2[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e8, o8), neg_shift));

            /* o9[0-3] */
            int32x4_t e9 = vsubq_s32(ee[6], eo[6]);
            int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
            o9 = vmlsl_lane_s16(o9, r3, c_o32_0, 3);
            o9 = vmlsl_lane_s16(o9, r5, c_o32_3, 3);
            o9 = vmlal_lane_s16(o9, r7, c_o32_0, 2);
            o9 = vmlsl_lane_s16(o9, r9, c_o32_2, 2);
            o9 = vmlsl_lane_s16(o9, r11, c_o32_2, 0);
            o9 = vmlal_lane_s16(o9, r13, c_o32_1, 0);
            o9 = vmlal_lane_s16(o9, r15, c_o32_3, 2);
            o9 = vmlsl_lane_s16(o9, r17, c_o32_0, 0);
            o9 = vmlal_lane_s16(o9, r19, c_o32_2, 3);
            o9 = vmlal_lane_s16(o9, r21, c_o32_1, 3);
            o9 = vmlsl_lane_s16(o9, r23, c_o32_1, 1);
            o9 = vmlsl_lane_s16(o9, r25, c_o32_3, 1);
            o9 = vmlal_lane_s16(o9, r27, c_o32_0, 0);
            o9 = vmlsl_lane_s16(o9, r29, c_o32_3, 0);
            o9 = vmlsl_lane_s16(o9, r31, c_o32_1, 2);
            y1[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e9, o9), neg_shift));
            y2[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e9, o9), neg_shift));

            /* o10[0-3] */
            int32x4_t e10 = vsubq_s32(ee[5], eo[5]);
            int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
            o10 = vmlsl_lane_s16(o10, r3, c_o32_0, 1);
            o10 = vmlal_lane_s16(o10, r5, c_o32_2, 3);
            o10 = vmlal_lane_s16(o10, r7, c_o32_2, 1);
            o10 = vmlsl_lane_s16(o10, r9, c_o32_0, 0);
            o10 = vmlal_lane_s16(o10, r11, c_o32_3, 0);
            o10 = vmlal_lane_s16(o10, r13, c_o32_2, 0);
            o10 = vmlsl_lane_s16(o10, r15, c_o32_0, 2);
            o10 = vmlal_lane_s16(o10, r17, c_o32_3, 1);
            o10 = vmlal_lane_s16(o10, r19, c_o32_1, 3);
            o10 = vmlsl_lane_s16(o10, r21, c_o32_0, 3);
            o10 = vmlal_lane_s16(o10, r23, c_o32_3, 2);
            o10 = vmlal_lane_s16(o10, r25, c_o32_1, 2);
            o10 = vmlsl_lane_s16(o10, r27, c_o32_1, 0);
            o10 = vmlal_lane_s16(o10, r29, c_o32_3, 3);
            o10 = vmlal_lane_s16(o10, r31, c_o32_1, 1);
            y1[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e10, o10), neg_shift));
            y2[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e10, o10), neg_shift));

            /* o11[0-3] */
            int32x4_t e11 = vsubq_s32(ee[4], eo[4]);
            int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
            o11 = vmlsl_lane_s16(o11, r3, c_o32_0, 2);
            o11 = vmlal_lane_s16(o11, r5, c_o32_1, 2);
            o11 = vmlsl_lane_s16(o11, r7, c_o32_3, 3);
            o11 = vmlsl_lane_s16(o11, r9, c_o32_1, 3);
            o11 = vmlal_lane_s16(o11, r11, c_o32_0, 0);
            o11 = vmlsl_lane_s16(o11, r13, c_o32_2, 2);
            o11 = vmlsl_lane_s16(o11, r15, c_o32_3, 0);
            o11 = vmlal_lane_s16(o11, r17, c_o32_0, 3);
            o11 = vmlsl_lane_s16(o11, r19, c_o32_1, 1);
            o11 = vmlal_lane_s16(o11, r21, c_o32_3, 2);
            o11 = vmlal_lane_s16(o11, r23, c_o32_2, 0);
            o11 = vmlsl_lane_s16(o11, r25, c_o32_0, 0);
            o11 = vmlal_lane_s16(o11, r27, c_o32_2, 1);
            o11 = vmlal_lane_s16(o11, r29, c_o32_3, 1);
            o11 = vmlsl_lane_s16(o11, r31, c_o32_1, 0);
            y1[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e11, o11), neg_shift));
            y2[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e11, o11), neg_shift));

            /* o12[0-3] */
            int32x4_t e12 = vsubq_s32(ee[3], eo[3]);
            int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
            o12 = vmlsl_lane_s16(o12, r3, c_o32_1, 1);
            o12 = vmlal_lane_s16(o12, r5, c_o32_0, 0);
            o12 = vmlsl_lane_s16(o12, r7, c_o32_2, 0);
            o12 = vmlal_lane_s16(o12, r9, c_o32_3, 3);
            o12 = vmlal_lane_s16(o12, r11, c_o32_2, 1);
            o12 = vmlsl_lane_s16(o12, r13, c_o32_0, 2);
            o12 = vmlal_lane_s16(o12, r15, c_o32_1, 0);
            o12 = vmlsl_lane_s16(o12, r17, c_o32_2, 3);
            o12 = vmlsl_lane_s16(o12, r19, c_o32_3, 1);
            o12 = vmlal_lane_s16(o12, r21, c_o32_1, 2);
            o12 = vmlsl_lane_s16(o12, r23, c_o32_0, 0);
            o12 = vmlal_lane_s16(o12, r25, c_o32_1, 3);
            o12 = vmlsl_lane_s16(o12, r27, c_o32_3, 2);
            o12 = vmlsl_lane_s16(o12, r29, c_o32_2, 2);
            o12 = vmlal_lane_s16(o12, r31, c_o32_0, 3);
            y1[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e12, o12), neg_shift));
            y2[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e12, o12), neg_shift));

            /* o13[0-3] */
            int32x4_t e13 = vsubq_s32(ee[2], eo[2]);
            int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
            o13 = vmlsl_lane_s16(o13, r3, c_o32_2, 0);
            o13 = vmlal_lane_s16(o13, r5, c_o32_0, 3);
            o13 = vmlsl_lane_s16(o13, r7, c_o32_0, 0);
            o13 = vmlal_lane_s16(o13, r9, c_o32_1, 2);
            o13 = vmlsl_lane_s16(o13, r11, c_o32_2, 3);
            o13 = vmlsl_lane_s16(o13, r13, c_o32_3, 3);
            o13 = vmlal_lane_s16(o13, r15, c_o32_2, 2);
            o13 = vmlsl_lane_s16(o13, r17, c_o32_1, 1);
            o13 = vmlal_lane_s16(o13, r19, c_o32_0, 0);
            o13 = vmlsl_lane_s16(o13, r21, c_o32_1, 0);
            o13 = vmlal_lane_s16(o13, r23, c_o32_2, 1);
            o13 = vmlsl_lane_s16(o13, r25, c_o32_3, 2);
            o13 = vmlsl_lane_s16(o13, r27, c_o32_3, 0);
            o13 = vmlal_lane_s16(o13, r29, c_o32_1, 3);
            o13 = vmlsl_lane_s16(o13, r31, c_o32_0, 2);
            y1[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e13, o13), neg_shift));
            y2[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e13, o13), neg_shift));

            /* o14[0-3] */
            int32x4_t e14 = vsubq_s32(ee[1], eo[1]);
            int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
            o14 = vmlsl_lane_s16(o14, r3, c_o32_2, 3);
            o14 = vmlal_lane_s16(o14, r5, c_o32_2, 0);
            o14 = vmlsl_lane_s16(o14, r7, c_o32_1, 1);
            o14 = vmlal_lane_s16(o14, r9, c_o32_0, 2);
            o14 = vmlsl_lane_s16(o14, r11, c_o32_0, 0);
            o14 = vmlal_lane_s16(o14, r13, c_o32_0, 3);
            o14 = vmlsl_lane_s16(o14, r15, c_o32_1, 2);
            o14 = vmlal_lane_s16(o14, r17, c_o32_2, 1);
            o14 = vmlsl_lane_s16(o14, r19, c_o32_3, 0);
            o14 = vmlal_lane_s16(o14, r21, c_o32_3, 3);
            o14 = vmlal_lane_s16(o14, r23, c_o32_3, 1);
            o14 = vmlsl_lane_s16(o14, r25, c_o32_2, 2);
            o14 = vmlal_lane_s16(o14, r27, c_o32_1, 3);
            o14 = vmlsl_lane_s16(o14, r29, c_o32_1, 0);
            o14 = vmlal_lane_s16(o14, r31, c_o32_0, 0);
            y1[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e14, o14), neg_shift));
            y2[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e14, o14), neg_shift));

            /* o15[0-3] */
            int32x4_t e15 = vsubq_s32(ee[0], eo[0]);
            int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
            o15 = vmlsl_lane_s16(o15, r3, c_o32_3, 2);
            o15 = vmlal_lane_s16(o15, r5, c_o32_3, 1);
            o15 = vmlsl_lane_s16(o15, r7, c_o32_3, 0);
            o15 = vmlal_lane_s16(o15, r9, c_o32_2, 3);
            o15 = vmlsl_lane_s16(o15, r11, c_o32_2, 2);
            o15 = vmlal_lane_s16(o15, r13, c_o32_2, 1);
            o15 = vmlsl_lane_s16(o15, r15, c_o32_2, 0);
            o15 = vmlal_lane_s16(o15, r17, c_o32_1, 3);
            o15 = vmlsl_lane_s16(o15, r19, c_o32_1, 2);
            o15 = vmlal_lane_s16(o15, r21, c_o32_1, 1);
            o15 = vmlsl_lane_s16(o15, r23, c_o32_1, 0);
            o15 = vmlal_lane_s16(o15, r25, c_o32_0, 3);
            o15 = vmlsl_lane_s16(o15, r27, c_o32_0, 2);
            o15 = vmlal_lane_s16(o15, r29, c_o32_0, 0);
            o15 = vmlsl_lane_s16(o15, r31, c_o32_0, 0);
            y1[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e15, o15), neg_shift));
            y2[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e15, o15), neg_shift));
        }

        /* Transpose */
        {
            {
                int16x4x2_t t0 = vtrn_s16(y0[0], y0[1]);
                int16x4x2_t t1 = vtrn_s16(y0[2], y0[3]);
                int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                int16x4x2_t t2 = vtrn_s16(y0[4], y0[5]);
                int16x4x2_t t3 = vtrn_s16(y0[6], y0[7]);
                int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));
                uint16x8x2_t pred_val0 = vld2q_u16(pu2_pred + (col + 0) * pred_strd + 0);
                uint16x8x2_t dst_val0  = vld2q_u16(pu2_dst  + (col + 0) * dst_strd  + 0);
                int16x8_t acc0 = vminq_s16(vmaxq_s16(vqaddq_s16(r0, vreinterpretq_s16_u16(pred_val0.val[0])), clip_min), clip_max);
                dst_val0.val[0] = vreinterpretq_u16_s16(acc0);
                vst2q_u16(pu2_dst + (col + 0) * dst_strd + 0, dst_val0);

                uint16x8x2_t pred_val1 = vld2q_u16(pu2_pred + (col + 1) * pred_strd + 0);
                uint16x8x2_t dst_val1  = vld2q_u16(pu2_dst  + (col + 1) * dst_strd  + 0);
                int16x8_t acc1 = vminq_s16(vmaxq_s16(vqaddq_s16(r1, vreinterpretq_s16_u16(pred_val1.val[0])), clip_min), clip_max);
                dst_val1.val[0] = vreinterpretq_u16_s16(acc1);
                vst2q_u16(pu2_dst + (col + 1) * dst_strd + 0, dst_val1);

                uint16x8x2_t pred_val2 = vld2q_u16(pu2_pred + (col + 2) * pred_strd + 0);
                uint16x8x2_t dst_val2  = vld2q_u16(pu2_dst  + (col + 2) * dst_strd  + 0);
                int16x8_t acc2 = vminq_s16(vmaxq_s16(vqaddq_s16(r2, vreinterpretq_s16_u16(pred_val2.val[0])), clip_min), clip_max);
                dst_val2.val[0] = vreinterpretq_u16_s16(acc2);
                vst2q_u16(pu2_dst + (col + 2) * dst_strd + 0, dst_val2);

                uint16x8x2_t pred_val3 = vld2q_u16(pu2_pred + (col + 3) * pred_strd + 0);
                uint16x8x2_t dst_val3  = vld2q_u16(pu2_dst  + (col + 3) * dst_strd  + 0);
                int16x8_t acc3 = vminq_s16(vmaxq_s16(vqaddq_s16(r3, vreinterpretq_s16_u16(pred_val3.val[0])), clip_min), clip_max);
                dst_val3.val[0] = vreinterpretq_u16_s16(acc3);
                vst2q_u16(pu2_dst + (col + 3) * dst_strd + 0, dst_val3);
            }
            {
                int16x4x2_t t0 = vtrn_s16(y1[0], y1[1]);
                int16x4x2_t t1 = vtrn_s16(y1[2], y1[3]);
                int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                int16x4x2_t t2 = vtrn_s16(y1[4], y1[5]);
                int16x4x2_t t3 = vtrn_s16(y1[6], y1[7]);
                int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));
                uint16x8x2_t pred_val0 = vld2q_u16(pu2_pred + (col + 0) * pred_strd + 16);
                uint16x8x2_t dst_val0  = vld2q_u16(pu2_dst  + (col + 0) * dst_strd  + 16);
                int16x8_t acc0 = vminq_s16(vmaxq_s16(vqaddq_s16(r0, vreinterpretq_s16_u16(pred_val0.val[0])), clip_min), clip_max);
                dst_val0.val[0] = vreinterpretq_u16_s16(acc0);
                vst2q_u16(pu2_dst + (col + 0) * dst_strd + 16, dst_val0);

                uint16x8x2_t pred_val1 = vld2q_u16(pu2_pred + (col + 1) * pred_strd + 16);
                uint16x8x2_t dst_val1  = vld2q_u16(pu2_dst  + (col + 1) * dst_strd  + 16);
                int16x8_t acc1 = vminq_s16(vmaxq_s16(vqaddq_s16(r1, vreinterpretq_s16_u16(pred_val1.val[0])), clip_min), clip_max);
                dst_val1.val[0] = vreinterpretq_u16_s16(acc1);
                vst2q_u16(pu2_dst + (col + 1) * dst_strd + 16, dst_val1);

                uint16x8x2_t pred_val2 = vld2q_u16(pu2_pred + (col + 2) * pred_strd + 16);
                uint16x8x2_t dst_val2  = vld2q_u16(pu2_dst  + (col + 2) * dst_strd  + 16);
                int16x8_t acc2 = vminq_s16(vmaxq_s16(vqaddq_s16(r2, vreinterpretq_s16_u16(pred_val2.val[0])), clip_min), clip_max);
                dst_val2.val[0] = vreinterpretq_u16_s16(acc2);
                vst2q_u16(pu2_dst + (col + 2) * dst_strd + 16, dst_val2);

                uint16x8x2_t pred_val3 = vld2q_u16(pu2_pred + (col + 3) * pred_strd + 16);
                uint16x8x2_t dst_val3  = vld2q_u16(pu2_dst  + (col + 3) * dst_strd  + 16);
                int16x8_t acc3 = vminq_s16(vmaxq_s16(vqaddq_s16(r3, vreinterpretq_s16_u16(pred_val3.val[0])), clip_min), clip_max);
                dst_val3.val[0] = vreinterpretq_u16_s16(acc3);
                vst2q_u16(pu2_dst + (col + 3) * dst_strd + 16, dst_val3);
            }
            {
                int16x4x2_t t0 = vtrn_s16(y2[0], y2[1]);
                int16x4x2_t t1 = vtrn_s16(y2[2], y2[3]);
                int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                int16x4x2_t t2 = vtrn_s16(y2[4], y2[5]);
                int16x4x2_t t3 = vtrn_s16(y2[6], y2[7]);
                int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));
                uint16x8x2_t pred_val0 = vld2q_u16(pu2_pred + (col + 0) * pred_strd + 32);
                uint16x8x2_t dst_val0  = vld2q_u16(pu2_dst  + (col + 0) * dst_strd  + 32);
                int16x8_t acc0 = vminq_s16(vmaxq_s16(vqaddq_s16(r0, vreinterpretq_s16_u16(pred_val0.val[0])), clip_min), clip_max);
                dst_val0.val[0] = vreinterpretq_u16_s16(acc0);
                vst2q_u16(pu2_dst + (col + 0) * dst_strd + 32, dst_val0);

                uint16x8x2_t pred_val1 = vld2q_u16(pu2_pred + (col + 1) * pred_strd + 32);
                uint16x8x2_t dst_val1  = vld2q_u16(pu2_dst  + (col + 1) * dst_strd  + 32);
                int16x8_t acc1 = vminq_s16(vmaxq_s16(vqaddq_s16(r1, vreinterpretq_s16_u16(pred_val1.val[0])), clip_min), clip_max);
                dst_val1.val[0] = vreinterpretq_u16_s16(acc1);
                vst2q_u16(pu2_dst + (col + 1) * dst_strd + 32, dst_val1);

                uint16x8x2_t pred_val2 = vld2q_u16(pu2_pred + (col + 2) * pred_strd + 32);
                uint16x8x2_t dst_val2  = vld2q_u16(pu2_dst  + (col + 2) * dst_strd  + 32);
                int16x8_t acc2 = vminq_s16(vmaxq_s16(vqaddq_s16(r2, vreinterpretq_s16_u16(pred_val2.val[0])), clip_min), clip_max);
                dst_val2.val[0] = vreinterpretq_u16_s16(acc2);
                vst2q_u16(pu2_dst + (col + 2) * dst_strd + 32, dst_val2);

                uint16x8x2_t pred_val3 = vld2q_u16(pu2_pred + (col + 3) * pred_strd + 32);
                uint16x8x2_t dst_val3  = vld2q_u16(pu2_dst  + (col + 3) * dst_strd  + 32);
                int16x8_t acc3 = vminq_s16(vmaxq_s16(vqaddq_s16(r3, vreinterpretq_s16_u16(pred_val3.val[0])), clip_min), clip_max);
                dst_val3.val[0] = vreinterpretq_u16_s16(acc3);
                vst2q_u16(pu2_dst + (col + 3) * dst_strd + 32, dst_val3);
            }
            {
                int16x4x2_t t0 = vtrn_s16(y3[0], y3[1]);
                int16x4x2_t t1 = vtrn_s16(y3[2], y3[3]);
                int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                int16x4x2_t t2 = vtrn_s16(y3[4], y3[5]);
                int16x4x2_t t3 = vtrn_s16(y3[6], y3[7]);
                int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));
                uint16x8x2_t pred_val0 = vld2q_u16(pu2_pred + (col + 0) * pred_strd + 48);
                uint16x8x2_t dst_val0  = vld2q_u16(pu2_dst  + (col + 0) * dst_strd  + 48);
                int16x8_t acc0 = vminq_s16(vmaxq_s16(vqaddq_s16(r0, vreinterpretq_s16_u16(pred_val0.val[0])), clip_min), clip_max);
                dst_val0.val[0] = vreinterpretq_u16_s16(acc0);
                vst2q_u16(pu2_dst + (col + 0) * dst_strd + 48, dst_val0);

                uint16x8x2_t pred_val1 = vld2q_u16(pu2_pred + (col + 1) * pred_strd + 48);
                uint16x8x2_t dst_val1  = vld2q_u16(pu2_dst  + (col + 1) * dst_strd  + 48);
                int16x8_t acc1 = vminq_s16(vmaxq_s16(vqaddq_s16(r1, vreinterpretq_s16_u16(pred_val1.val[0])), clip_min), clip_max);
                dst_val1.val[0] = vreinterpretq_u16_s16(acc1);
                vst2q_u16(pu2_dst + (col + 1) * dst_strd + 48, dst_val1);

                uint16x8x2_t pred_val2 = vld2q_u16(pu2_pred + (col + 2) * pred_strd + 48);
                uint16x8x2_t dst_val2  = vld2q_u16(pu2_dst  + (col + 2) * dst_strd  + 48);
                int16x8_t acc2 = vminq_s16(vmaxq_s16(vqaddq_s16(r2, vreinterpretq_s16_u16(pred_val2.val[0])), clip_min), clip_max);
                dst_val2.val[0] = vreinterpretq_u16_s16(acc2);
                vst2q_u16(pu2_dst + (col + 2) * dst_strd + 48, dst_val2);

                uint16x8x2_t pred_val3 = vld2q_u16(pu2_pred + (col + 3) * pred_strd + 48);
                uint16x8x2_t dst_val3  = vld2q_u16(pu2_dst  + (col + 3) * dst_strd  + 48);
                int16x8_t acc3 = vminq_s16(vmaxq_s16(vqaddq_s16(r3, vreinterpretq_s16_u16(pred_val3.val[0])), clip_min), clip_max);
                dst_val3.val[0] = vreinterpretq_u16_s16(acc3);
                vst2q_u16(pu2_dst + (col + 3) * dst_strd + 48, dst_val3);
            }
        }
    }
}
