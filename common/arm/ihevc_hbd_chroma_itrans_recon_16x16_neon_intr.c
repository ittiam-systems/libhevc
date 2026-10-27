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
*  ihevc_hbd_chroma_itrans_recon_16x16_neon_intr.c
*
* @brief
*  Contains function definitions for chroma 16x16 inverse transform and
*  reconstruction (HBD NEON).
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_chroma_itrans_recon_16x16_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/
#include "ihevc_typedefs.h"
#include <arm_neon.h>
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_itrans_recon.h"

void ihevc_hbd_chroma_itrans_recon_16x16_neonintr(WORD16 *pi2_src,
                                                  WORD16 *pi2_tmp,
                                                  UWORD16 *pu2_pred,
                                                  UWORD16 *pu2_dst,
                                                  WORD32 src_strd,
                                                  WORD32 pred_strd,
                                                  WORD32 dst_strd,
                                                  WORD32 zero_cols,
                                                  WORD32 zero_rows,
                                                  UWORD8 bit_depth)
{
    /* Fast Zero-block bypass: both zero_cols and zero_rows checked */
    if (((UWORD32)(zero_cols & 0xFFFF) == 0xFFFFU) || ((UWORD32)(zero_rows & 0xFFFF) == 0xFFFFU))
    {
        for (WORD32 i = 0; i < 16; i++)
        {
            uint16x8x2_t pred0 = vld2q_u16(pu2_pred + 0);
            uint16x8x2_t dst0  = vld2q_u16(pu2_dst + 0);
            dst0.val[0] = pred0.val[0];
            vst2q_u16(pu2_dst + 0, dst0);

            uint16x8x2_t pred1 = vld2q_u16(pu2_pred + 16);
            uint16x8x2_t dst1  = vld2q_u16(pu2_dst + 16);
            dst1.val[0] = pred1.val[0];
            vst2q_u16(pu2_dst + 16, dst1);

            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
        return;
    }

    int16x4_t ce0 = vld1_s16(&g_ai2_ihevc_trans_16[4][0]);
    int16x4_t ce1 = vld1_s16(&g_ai2_ihevc_trans_16[2][0]);
    int16x4_t co0 = vld1_s16(&g_ai2_ihevc_trans_16[1][0]);
    int16x4_t co1 = vld1_s16(&g_ai2_ihevc_trans_16[1][4]);

    /* Last 8 cols of 16x16 block are skipped based on the below flag */
    WORD32 zero_last8_cols_stg1 = ((zero_cols & 0xFF00) == 0xFF00) ? 1 : 0;
    WORD32 zero_last8_rows_stg1 = ((zero_rows & 0xFF00) == 0xFF00) ? 1 : 0;
    WORD32 zero_last12_rows_stg1 = ((zero_rows & 0xFFF0) == 0xFFF0) ? 1 : 0;

    WORD32 zero_last12_rows_stg2 = ((zero_cols & 0xFFF0) == 0xFFF0) ? 1 : 0;
    WORD32 zero_last8_rows_stg2 = zero_last8_cols_stg1;

    WORD32 loop;
    if (zero_last8_cols_stg1)
    {
        loop = 1;
    }
    else
    {
        loop = 2;
    }

    /* Stage 1 */
    /* i = 0 => lower 8 samples */
    /* i = 1 => higher 8 samples */
    for (WORD32 i = 0; i < loop; i++)
    {
        int32x4_t e[2][8];
        int16x4_t s0[2][8], s1[2][8];
        int16x8_t m_temp_reg_70, m_temp_reg_71, m_temp_reg_72, m_temp_reg_73;
        int16x8_t m_temp_reg_74, m_temp_reg_75, m_temp_reg_76, m_temp_reg_77;

        {
            WORD32 sample_half_index = i << 3;
            WORD16* pi2_tmp_src = pi2_src + sample_half_index;

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

            /* If last 12 rows are zero */
            if (zero_last12_rows_stg1)
            {
                int16x4_t r0_lo = vget_low_s16(m_temp_reg_70);
                int16x4_t r0_hi = vget_high_s16(m_temp_reg_70);
                int16x4_t r2_lo = vget_low_s16(m_temp_reg_71);
                int16x4_t r2_hi = vget_high_s16(m_temp_reg_71);

                /* eee */
                int32x4_t ee_lo = vshll_n_s16(r0_lo, 6);
                int32x4_t ee_hi = vshll_n_s16(r0_hi, 6);

                /* eo */

                /* eo0[0-3] */
                {
                    int32x4_t eo0 = vmull_lane_s16(r2_lo, ce1, 0);
                    e[0][0] = vaddq_s32(ee_lo, eo0);
                    e[0][7] = vsubq_s32(ee_lo, eo0);
                }

                /* eo0[4-7] */
                {
                    int32x4_t eo0 = vmull_lane_s16(r2_hi, ce1, 0);
                    e[1][0] = vaddq_s32(ee_hi, eo0);
                    e[1][7] = vsubq_s32(ee_hi, eo0);
                }

                /* eo1[0-3] */
                {
                    int32x4_t eo1 = vmull_lane_s16(r2_lo, ce1, 1);
                    e[0][1] = vaddq_s32(ee_lo, eo1);
                    e[0][6] = vsubq_s32(ee_lo, eo1);
                }

                /* eo1[4-7] */
                {
                    int32x4_t eo1 = vmull_lane_s16(r2_hi, ce1, 1);
                    e[1][1] = vaddq_s32(ee_hi, eo1);
                    e[1][6] = vsubq_s32(ee_hi, eo1);
                }

                /* eo2[0-3] */
                {
                    int32x4_t eo2 = vmull_lane_s16(r2_lo, ce1, 2);
                    e[0][2] = vaddq_s32(ee_lo, eo2);
                    e[0][5] = vsubq_s32(ee_lo, eo2);
                }

                /* eo2[4-7] */
                {
                    int32x4_t eo2 = vmull_lane_s16(r2_hi, ce1, 2);
                    e[1][2] = vaddq_s32(ee_hi, eo2);
                    e[1][5] = vsubq_s32(ee_hi, eo2);
                }

                /* eo3[0-3] */
                {
                    int32x4_t eo3 = vmull_lane_s16(r2_lo, ce1, 3);
                    e[0][3] = vaddq_s32(ee_lo, eo3);
                    e[0][4] = vsubq_s32(ee_lo, eo3);
                }

                /* eo3[4-7] */
                {
                    int32x4_t eo3 = vmull_lane_s16(r2_hi, ce1, 3);
                    e[1][3] = vaddq_s32(ee_hi, eo3);
                    e[1][4] = vsubq_s32(ee_hi, eo3);
                }
            }
            /* If last 8 rows are zero */
            else if (zero_last8_rows_stg1)
            {
                int16x4_t r0_lo = vget_low_s16(m_temp_reg_70);
                int16x4_t r0_hi = vget_high_s16(m_temp_reg_70);
                int16x4_t r2_lo = vget_low_s16(m_temp_reg_71);
                int16x4_t r2_hi = vget_high_s16(m_temp_reg_71);
                int16x4_t r4_lo = vget_low_s16(m_temp_reg_72);
                int16x4_t r4_hi = vget_high_s16(m_temp_reg_72);
                int16x4_t r6_lo = vget_low_s16(m_temp_reg_73);
                int16x4_t r6_hi = vget_high_s16(m_temp_reg_73);

                /* eeo */
                int32x4_t eeo0_lo = vmull_lane_s16(r4_lo, ce0, 0);
                int32x4_t eeo1_lo = vmull_lane_s16(r4_lo, ce0, 1);
                int32x4_t eeo0_hi = vmull_lane_s16(r4_hi, ce0, 0);
                int32x4_t eeo1_hi = vmull_lane_s16(r4_hi, ce0, 1);

                /* eee */
                int32x4_t eee_lo = vshll_n_s16(r0_lo, 6);
                int32x4_t ee0_lo = vaddq_s32(eee_lo, eeo0_lo);
                int32x4_t ee1_lo = vaddq_s32(eee_lo, eeo1_lo);
                int32x4_t ee2_lo = vsubq_s32(eee_lo, eeo1_lo);
                int32x4_t ee3_lo = vsubq_s32(eee_lo, eeo0_lo);

                int32x4_t eee_hi = vshll_n_s16(r0_hi, 6);
                int32x4_t ee0_hi = vaddq_s32(eee_hi, eeo0_hi);
                int32x4_t ee1_hi = vaddq_s32(eee_hi, eeo1_hi);
                int32x4_t ee2_hi = vsubq_s32(eee_hi, eeo1_hi);
                int32x4_t ee3_hi = vsubq_s32(eee_hi, eeo0_hi);

                /* eo */

                /* eo0[0-3] */
                {
                    int32x4_t eo0 = vmlal_lane_s16(vmull_lane_s16(r2_lo, ce1, 0), r6_lo, ce1, 1);
                    e[0][0] = vaddq_s32(ee0_lo, eo0);
                    e[0][7] = vsubq_s32(ee0_lo, eo0);
                }

                /* eo0[4-7] */
                {
                    int32x4_t eo0 = vmlal_lane_s16(vmull_lane_s16(r2_hi, ce1, 0), r6_hi, ce1, 1);
                    e[1][0] = vaddq_s32(ee0_hi, eo0);
                    e[1][7] = vsubq_s32(ee0_hi, eo0);
                }

                /* eo1[0-3] */
                {
                    int32x4_t eo1 = vmlsl_lane_s16(vmull_lane_s16(r2_lo, ce1, 1), r6_lo, ce1, 3);
                    e[0][1] = vaddq_s32(ee1_lo, eo1);
                    e[0][6] = vsubq_s32(ee1_lo, eo1);
                }

                /* eo1[4-7] */
                {
                    int32x4_t eo1 = vmlsl_lane_s16(vmull_lane_s16(r2_hi, ce1, 1), r6_hi, ce1, 3);
                    e[1][1] = vaddq_s32(ee1_hi, eo1);
                    e[1][6] = vsubq_s32(ee1_hi, eo1);
                }

                /* eo2[0-3] */
                {
                    int32x4_t eo2 = vmlsl_lane_s16(vmull_lane_s16(r2_lo, ce1, 2), r6_lo, ce1, 0);
                    e[0][2] = vaddq_s32(ee2_lo, eo2);
                    e[0][5] = vsubq_s32(ee2_lo, eo2);
                }

                /* eo2[4-7] */
                {
                    int32x4_t eo2 = vmlsl_lane_s16(vmull_lane_s16(r2_hi, ce1, 2), r6_hi, ce1, 0);
                    e[1][2] = vaddq_s32(ee2_hi, eo2);
                    e[1][5] = vsubq_s32(ee2_hi, eo2);
                }

                /* eo3[0-3] */
                {
                    int32x4_t eo3 = vmlsl_lane_s16(vmull_lane_s16(r2_lo, ce1, 3), r6_lo, ce1, 2);
                    e[0][3] = vaddq_s32(ee3_lo, eo3);
                    e[0][4] = vsubq_s32(ee3_lo, eo3);
                }

                /* eo3[4-7] */
                {
                    int32x4_t eo3 = vmlsl_lane_s16(vmull_lane_s16(r2_hi, ce1, 3), r6_hi, ce1, 2);
                    e[1][3] = vaddq_s32(ee3_hi, eo3);
                    e[1][4] = vsubq_s32(ee3_hi, eo3);
                }
            }
            /* If all the rows are non-zero */
            else
            {
                int16x4_t r0_lo = vget_low_s16(m_temp_reg_70);
                int16x4_t r0_hi = vget_high_s16(m_temp_reg_70);
                int16x4_t r2_lo = vget_low_s16(m_temp_reg_71);
                int16x4_t r2_hi = vget_high_s16(m_temp_reg_71);
                int16x4_t r4_lo = vget_low_s16(m_temp_reg_72);
                int16x4_t r4_hi = vget_high_s16(m_temp_reg_72);
                int16x4_t r6_lo = vget_low_s16(m_temp_reg_73);
                int16x4_t r6_hi = vget_high_s16(m_temp_reg_73);
                int16x4_t r8_lo = vget_low_s16(m_temp_reg_74);
                int16x4_t r8_hi = vget_high_s16(m_temp_reg_74);
                int16x4_t r10_lo = vget_low_s16(m_temp_reg_75);
                int16x4_t r10_hi = vget_high_s16(m_temp_reg_75);
                int16x4_t r12_lo = vget_low_s16(m_temp_reg_76);
                int16x4_t r12_hi = vget_high_s16(m_temp_reg_76);
                int16x4_t r14_lo = vget_low_s16(m_temp_reg_77);
                int16x4_t r14_hi = vget_high_s16(m_temp_reg_77);

                /* eeo */
                int32x4_t eeo0_lo = vmlal_lane_s16(vmull_lane_s16(r4_lo, ce0, 0), r12_lo, ce0, 1);
                int32x4_t eeo1_lo = vmlsl_lane_s16(vmull_lane_s16(r4_lo, ce0, 1), r12_lo, ce0, 0);
                int32x4_t eeo0_hi = vmlal_lane_s16(vmull_lane_s16(r4_hi, ce0, 0), r12_hi, ce0, 1);
                int32x4_t eeo1_hi = vmlsl_lane_s16(vmull_lane_s16(r4_hi, ce0, 1), r12_hi, ce0, 0);

                /* eee */
                int32x4_t eee0_lo = vshlq_n_s32(vaddl_s16(r0_lo, r8_lo), 6);
                int32x4_t eee1_lo = vshlq_n_s32(vsubl_s16(r0_lo, r8_lo), 6);
                int32x4_t ee0_lo = vaddq_s32(eee0_lo, eeo0_lo);
                int32x4_t ee1_lo = vaddq_s32(eee1_lo, eeo1_lo);
                int32x4_t ee2_lo = vsubq_s32(eee1_lo, eeo1_lo);
                int32x4_t ee3_lo = vsubq_s32(eee0_lo, eeo0_lo);

                int32x4_t eee0_hi = vshlq_n_s32(vaddl_s16(r0_hi, r8_hi), 6);
                int32x4_t eee1_hi = vshlq_n_s32(vsubl_s16(r0_hi, r8_hi), 6);
                int32x4_t ee0_hi = vaddq_s32(eee0_hi, eeo0_hi);
                int32x4_t ee1_hi = vaddq_s32(eee1_hi, eeo1_hi);
                int32x4_t ee2_hi = vsubq_s32(eee1_hi, eeo1_hi);
                int32x4_t ee3_hi = vsubq_s32(eee0_hi, eeo0_hi);

                /* eo */

                /* eo0[0-3] */
                {
                    int32x4_t eo0 = vmull_lane_s16(r2_lo, ce1, 0);
                    eo0 = vmlal_lane_s16(eo0, r6_lo, ce1, 1);
                    eo0 = vmlal_lane_s16(eo0, r10_lo, ce1, 2);
                    eo0 = vmlal_lane_s16(eo0, r14_lo, ce1, 3);
                    e[0][0] = vaddq_s32(ee0_lo, eo0);
                    e[0][7] = vsubq_s32(ee0_lo, eo0);
                }

                /* eo0[4-7] */
                {
                    int32x4_t eo0 = vmull_lane_s16(r2_hi, ce1, 0);
                    eo0 = vmlal_lane_s16(eo0, r6_hi, ce1, 1);
                    eo0 = vmlal_lane_s16(eo0, r10_hi, ce1, 2);
                    eo0 = vmlal_lane_s16(eo0, r14_hi, ce1, 3);
                    e[1][0] = vaddq_s32(ee0_hi, eo0);
                    e[1][7] = vsubq_s32(ee0_hi, eo0);
                }

                /* eo1[0-3] */
                {
                    int32x4_t eo1 = vmull_lane_s16(r2_lo, ce1, 1);
                    eo1 = vmlsl_lane_s16(eo1, r6_lo, ce1, 3);
                    eo1 = vmlsl_lane_s16(eo1, r10_lo, ce1, 0);
                    eo1 = vmlsl_lane_s16(eo1, r14_lo, ce1, 2);
                    e[0][1] = vaddq_s32(ee1_lo, eo1);
                    e[0][6] = vsubq_s32(ee1_lo, eo1);
                }

                /* eo1[4-7] */
                {
                    int32x4_t eo1 = vmull_lane_s16(r2_hi, ce1, 1);
                    eo1 = vmlsl_lane_s16(eo1, r6_hi, ce1, 3);
                    eo1 = vmlsl_lane_s16(eo1, r10_hi, ce1, 0);
                    eo1 = vmlsl_lane_s16(eo1, r14_hi, ce1, 2);
                    e[1][1] = vaddq_s32(ee1_hi, eo1);
                    e[1][6] = vsubq_s32(ee1_hi, eo1);
                }

                /* eo2[0-3] */
                {
                    int32x4_t eo2 = vmull_lane_s16(r2_lo, ce1, 2);
                    eo2 = vmlsl_lane_s16(eo2, r6_lo, ce1, 0);
                    eo2 = vmlal_lane_s16(eo2, r10_lo, ce1, 3);
                    eo2 = vmlal_lane_s16(eo2, r14_lo, ce1, 1);
                    e[0][2] = vaddq_s32(ee2_lo, eo2);
                    e[0][5] = vsubq_s32(ee2_lo, eo2);
                }

                /* eo2[4-7] */
                {
                    int32x4_t eo2 = vmull_lane_s16(r2_hi, ce1, 2);
                    eo2 = vmlsl_lane_s16(eo2, r6_hi, ce1, 0);
                    eo2 = vmlal_lane_s16(eo2, r10_hi, ce1, 3);
                    eo2 = vmlal_lane_s16(eo2, r14_hi, ce1, 1);
                    e[1][2] = vaddq_s32(ee2_hi, eo2);
                    e[1][5] = vsubq_s32(ee2_hi, eo2);
                }

                /* eo3[0-3] */
                {
                    int32x4_t eo3 = vmull_lane_s16(r2_lo, ce1, 3);
                    eo3 = vmlsl_lane_s16(eo3, r6_lo, ce1, 2);
                    eo3 = vmlal_lane_s16(eo3, r10_lo, ce1, 1);
                    eo3 = vmlsl_lane_s16(eo3, r14_lo, ce1, 0);
                    e[0][3] = vaddq_s32(ee3_lo, eo3);
                    e[0][4] = vsubq_s32(ee3_lo, eo3);
                }

                /* eo3[4-7] */
                {
                    int32x4_t eo3 = vmull_lane_s16(r2_hi, ce1, 3);
                    eo3 = vmlsl_lane_s16(eo3, r6_hi, ce1, 2);
                    eo3 = vmlal_lane_s16(eo3, r10_hi, ce1, 1);
                    eo3 = vmlsl_lane_s16(eo3, r14_hi, ce1, 0);
                    e[1][3] = vaddq_s32(ee3_hi, eo3);
                    e[1][4] = vsubq_s32(ee3_hi, eo3);
                }
            }
        }

        {
            WORD32 sample_half_index = i << 3;
            WORD16* pi2_tmp_src = pi2_src + sample_half_index + src_strd;

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
        }

        /* o & stage 1 out */
        {
            WORD32 j;

            if (zero_last12_rows_stg1)
            {
                for (j = 0; j < 2; j++)
                {
                    int16x4_t r1, r3;
                    if (j) // H8B= higher 8 bytes L8B lower 8 bytes
                    {
                        r1 = vget_high_s16(m_temp_reg_70);
                        r3 = vget_high_s16(m_temp_reg_71);
                    }
                    else
                    {
                        r1 = vget_low_s16(m_temp_reg_70);
                        r3 = vget_low_s16(m_temp_reg_71);
                    }

                    /* o0[0-3] */
                    {
                        int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 0), r3, co0, 1);
                        s0[j][0] = vqrshrn_n_s32(vaddq_s32(e[j][0], o0), 7);
                        s1[j][7] = vqrshrn_n_s32(vsubq_s32(e[j][0], o0), 7);
                    }

                    /* o1[0-3] */
                    {
                        int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 1), r3, co1, 0);
                        s0[j][1] = vqrshrn_n_s32(vaddq_s32(e[j][1], o1), 7);
                        s1[j][6] = vqrshrn_n_s32(vsubq_s32(e[j][1], o1), 7);
                    }

                    /* o2[0-3] */
                    {
                        int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 2), r3, co1, 3);
                        s0[j][2] = vqrshrn_n_s32(vaddq_s32(e[j][2], o2), 7);
                        s1[j][5] = vqrshrn_n_s32(vsubq_s32(e[j][2], o2), 7);
                    }

                    /* o3[0-3] */
                    {
                        int32x4_t o3 = vmlsl_lane_s16(vmull_lane_s16(r1, co0, 3), r3, co1, 1);
                        s0[j][3] = vqrshrn_n_s32(vaddq_s32(e[j][3], o3), 7);
                        s1[j][4] = vqrshrn_n_s32(vsubq_s32(e[j][3], o3), 7);
                    }

                    /* o4[0-3] */
                    {
                        int32x4_t o4 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 0), r3, co0, 2);
                        s0[j][4] = vqrshrn_n_s32(vaddq_s32(e[j][4], o4), 7);
                        s1[j][3] = vqrshrn_n_s32(vsubq_s32(e[j][4], o4), 7);
                    }

                    /* o5[0-3] */
                    {
                        int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 1), r3, co0, 0);
                        s0[j][5] = vqrshrn_n_s32(vaddq_s32(e[j][5], o5), 7);
                        s1[j][2] = vqrshrn_n_s32(vsubq_s32(e[j][5], o5), 7);
                    }

                    /* o6[0-3] */
                    {
                        int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 2), r3, co0, 3);
                        s0[j][6] = vqrshrn_n_s32(vaddq_s32(e[j][6], o6), 7);
                        s1[j][1] = vqrshrn_n_s32(vsubq_s32(e[j][6], o6), 7);
                    }

                    /* o7[0-3] */
                    {
                        int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 3), r3, co1, 2);
                        s0[j][7] = vqrshrn_n_s32(vaddq_s32(e[j][7], o7), 7);
                        s1[j][0] = vqrshrn_n_s32(vsubq_s32(e[j][7], o7), 7);
                    }
                }
            }
            else if (zero_last8_rows_stg1)
            {
                for (j = 0; j < 2; j++)
                {
                    int16x4_t r1, r3, r5, r7;
                    if (j)
                    {
                        r1 = vget_high_s16(m_temp_reg_70);
                        r3 = vget_high_s16(m_temp_reg_71);
                        r5 = vget_high_s16(m_temp_reg_72);
                        r7 = vget_high_s16(m_temp_reg_73);
                    }
                    else
                    {
                        r1 = vget_low_s16(m_temp_reg_70);
                        r3 = vget_low_s16(m_temp_reg_71);
                        r5 = vget_low_s16(m_temp_reg_72);
                        r7 = vget_low_s16(m_temp_reg_73);
                    }

                    /* o0[0-3] */
                    {
                        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
                        o0 = vmlal_lane_s16(o0, r3, co0, 1);
                        o0 = vmlal_lane_s16(o0, r5, co0, 2);
                        o0 = vmlal_lane_s16(o0, r7, co0, 3);
                        s0[j][0] = vqrshrn_n_s32(vaddq_s32(e[j][0], o0), 7);
                        s1[j][7] = vqrshrn_n_s32(vsubq_s32(e[j][0], o0), 7);
                    }

                    /* o1[0-3] */
                    {
                        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
                        o1 = vmlal_lane_s16(o1, r3, co1, 0);
                        o1 = vmlal_lane_s16(o1, r5, co1, 3);
                        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
                        s0[j][1] = vqrshrn_n_s32(vaddq_s32(e[j][1], o1), 7);
                        s1[j][6] = vqrshrn_n_s32(vsubq_s32(e[j][1], o1), 7);
                    }

                    /* o2[0-3] */
                    {
                        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
                        o2 = vmlal_lane_s16(o2, r3, co1, 3);
                        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
                        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
                        s0[j][2] = vqrshrn_n_s32(vaddq_s32(e[j][2], o2), 7);
                        s1[j][5] = vqrshrn_n_s32(vsubq_s32(e[j][2], o2), 7);
                    }

                    /* o3[0-3] */
                    {
                        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
                        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
                        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
                        o3 = vmlal_lane_s16(o3, r7, co1, 3);
                        s0[j][3] = vqrshrn_n_s32(vaddq_s32(e[j][3], o3), 7);
                        s1[j][4] = vqrshrn_n_s32(vsubq_s32(e[j][3], o3), 7);
                    }

                    /* o4[0-3] */
                    {
                        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
                        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
                        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
                        o4 = vmlal_lane_s16(o4, r7, co0, 0);
                        s0[j][4] = vqrshrn_n_s32(vaddq_s32(e[j][4], o4), 7);
                        s1[j][3] = vqrshrn_n_s32(vsubq_s32(e[j][4], o4), 7);
                    }

                    /* o5[0-3] */
                    {
                        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
                        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
                        o5 = vmlal_lane_s16(o5, r5, co1, 0);
                        o5 = vmlal_lane_s16(o5, r7, co1, 2);
                        s0[j][5] = vqrshrn_n_s32(vaddq_s32(e[j][5], o5), 7);
                        s1[j][2] = vqrshrn_n_s32(vsubq_s32(e[j][5], o5), 7);
                    }

                    /* o6[0-3] */
                    {
                        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
                        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
                        o6 = vmlal_lane_s16(o6, r5, co0, 0);
                        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
                        s0[j][6] = vqrshrn_n_s32(vaddq_s32(e[j][6], o6), 7);
                        s1[j][1] = vqrshrn_n_s32(vsubq_s32(e[j][6], o6), 7);
                    }

                    /* o7[0-3] */
                    {
                        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
                        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
                        o7 = vmlal_lane_s16(o7, r5, co1, 1);
                        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
                        s0[j][7] = vqrshrn_n_s32(vaddq_s32(e[j][7], o7), 7);
                        s1[j][0] = vqrshrn_n_s32(vsubq_s32(e[j][7], o7), 7);
                    }
                }
            }
            /* If all the rows are non-zero */
            else
            {
                for (j = 0; j < 2; j++)
                {
                    int16x4_t r1, r3, r5, r7, r9, r11, r13, r15;
                    if (j)
                    {
                        r1 = vget_high_s16(m_temp_reg_70);
                        r3 = vget_high_s16(m_temp_reg_71);
                        r5 = vget_high_s16(m_temp_reg_72);
                        r7 = vget_high_s16(m_temp_reg_73);
                        r9 = vget_high_s16(m_temp_reg_74);
                        r11 = vget_high_s16(m_temp_reg_75);
                        r13 = vget_high_s16(m_temp_reg_76);
                        r15 = vget_high_s16(m_temp_reg_77);
                    }
                    else
                    {
                        r1 = vget_low_s16(m_temp_reg_70);
                        r3 = vget_low_s16(m_temp_reg_71);
                        r5 = vget_low_s16(m_temp_reg_72);
                        r7 = vget_low_s16(m_temp_reg_73);
                        r9 = vget_low_s16(m_temp_reg_74);
                        r11 = vget_low_s16(m_temp_reg_75);
                        r13 = vget_low_s16(m_temp_reg_76);
                        r15 = vget_low_s16(m_temp_reg_77);
                    }

                    /* o0[0-3] */
                    {
                        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
                        o0 = vmlal_lane_s16(o0, r3, co0, 1);
                        o0 = vmlal_lane_s16(o0, r5, co0, 2);
                        o0 = vmlal_lane_s16(o0, r7, co0, 3);
                        o0 = vmlal_lane_s16(o0, r9, co1, 0);
                        o0 = vmlal_lane_s16(o0, r11, co1, 1);
                        o0 = vmlal_lane_s16(o0, r13, co1, 2);
                        o0 = vmlal_lane_s16(o0, r15, co1, 3);
                        s0[j][0] = vqrshrn_n_s32(vaddq_s32(e[j][0], o0), 7);
                        s1[j][7] = vqrshrn_n_s32(vsubq_s32(e[j][0], o0), 7);
                    }

                    /* o1[0-3] */
                    {
                        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
                        o1 = vmlal_lane_s16(o1, r3, co1, 0);
                        o1 = vmlal_lane_s16(o1, r5, co1, 3);
                        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
                        o1 = vmlsl_lane_s16(o1, r9, co0, 2);
                        o1 = vmlsl_lane_s16(o1, r11, co0, 0);
                        o1 = vmlsl_lane_s16(o1, r13, co0, 3);
                        o1 = vmlsl_lane_s16(o1, r15, co1, 2);
                        s0[j][1] = vqrshrn_n_s32(vaddq_s32(e[j][1], o1), 7);
                        s1[j][6] = vqrshrn_n_s32(vsubq_s32(e[j][1], o1), 7);
                    }

                    /* o2[0-3] */
                    {
                        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
                        o2 = vmlal_lane_s16(o2, r3, co1, 3);
                        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
                        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
                        o2 = vmlsl_lane_s16(o2, r9, co1, 2);
                        o2 = vmlal_lane_s16(o2, r11, co1, 0);
                        o2 = vmlal_lane_s16(o2, r13, co0, 0);
                        o2 = vmlal_lane_s16(o2, r15, co1, 1);
                        s0[j][2] = vqrshrn_n_s32(vaddq_s32(e[j][2], o2), 7);
                        s1[j][5] = vqrshrn_n_s32(vsubq_s32(e[j][2], o2), 7);
                    }

                    /* o3[0-3] */
                    {
                        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
                        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
                        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
                        o3 = vmlal_lane_s16(o3, r7, co1, 3);
                        o3 = vmlal_lane_s16(o3, r9, co0, 0);
                        o3 = vmlal_lane_s16(o3, r11, co1, 2);
                        o3 = vmlsl_lane_s16(o3, r13, co0, 2);
                        o3 = vmlsl_lane_s16(o3, r15, co1, 0);
                        s0[j][3] = vqrshrn_n_s32(vaddq_s32(e[j][3], o3), 7);
                        s1[j][4] = vqrshrn_n_s32(vsubq_s32(e[j][3], o3), 7);
                    }

                    /* o4[0-3] */
                    {
                        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
                        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
                        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
                        o4 = vmlal_lane_s16(o4, r7, co0, 0);
                        o4 = vmlsl_lane_s16(o4, r9, co1, 3);
                        o4 = vmlsl_lane_s16(o4, r11, co0, 1);
                        o4 = vmlal_lane_s16(o4, r13, co1, 1);
                        o4 = vmlal_lane_s16(o4, r15, co0, 3);
                        s0[j][4] = vqrshrn_n_s32(vaddq_s32(e[j][4], o4), 7);
                        s1[j][3] = vqrshrn_n_s32(vsubq_s32(e[j][4], o4), 7);
                    }

                    /* o5[0-3] */
                    {
                        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
                        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
                        o5 = vmlal_lane_s16(o5, r5, co1, 0);
                        o5 = vmlal_lane_s16(o5, r7, co1, 2);
                        o5 = vmlsl_lane_s16(o5, r9, co0, 1);
                        o5 = vmlal_lane_s16(o5, r11, co0, 3);
                        o5 = vmlal_lane_s16(o5, r13, co1, 3);
                        o5 = vmlsl_lane_s16(o5, r15, co0, 2);
                        s0[j][5] = vqrshrn_n_s32(vaddq_s32(e[j][5], o5), 7);
                        s1[j][2] = vqrshrn_n_s32(vsubq_s32(e[j][5], o5), 7);
                    }

                    /* o6[0-3] */
                    {
                        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
                        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
                        o6 = vmlal_lane_s16(o6, r5, co0, 0);
                        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
                        o6 = vmlal_lane_s16(o6, r9, co1, 1);
                        o6 = vmlal_lane_s16(o6, r11, co1, 3);
                        o6 = vmlsl_lane_s16(o6, r13, co1, 0);
                        o6 = vmlal_lane_s16(o6, r15, co0, 1);
                        s0[j][6] = vqrshrn_n_s32(vaddq_s32(e[j][6], o6), 7);
                        s1[j][1] = vqrshrn_n_s32(vsubq_s32(e[j][6], o6), 7);
                    }

                    /* o7[0-3] */
                    {
                        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
                        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
                        o7 = vmlal_lane_s16(o7, r5, co1, 1);
                        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
                        o7 = vmlal_lane_s16(o7, r9, co0, 3);
                        o7 = vmlsl_lane_s16(o7, r11, co0, 2);
                        o7 = vmlal_lane_s16(o7, r13, co0, 1);
                        o7 = vmlsl_lane_s16(o7, r15, co0, 0);
                        s0[j][7] = vqrshrn_n_s32(vaddq_s32(e[j][7], o7), 7);
                        s1[j][0] = vqrshrn_n_s32(vsubq_s32(e[j][7], o7), 7);
                    }
                }
            }
        }

        /* Transpose */
        {
            for (WORD32 j = 0; j < 2; j++)
            {
                WORD32 cb = (i << 3) + (j << 2);

                /* Transpose s0 (8 vectors of 4 elements) to 4 vectors of 8 elements and store to pi2_tmp at col 0 */
                int16x4x2_t t0 = vtrn_s16(s0[j][0], s0[j][1]);
                int16x4x2_t t1 = vtrn_s16(s0[j][2], s0[j][3]);
                int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                int16x4x2_t t2 = vtrn_s16(s0[j][4], s0[j][5]);
                int16x4x2_t t3 = vtrn_s16(s0[j][6], s0[j][7]);
                int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));
                vst1q_s16(pi2_tmp + (cb + 0) * 16 + 0, r0);
                vst1q_s16(pi2_tmp + (cb + 1) * 16 + 0, r1);
                vst1q_s16(pi2_tmp + (cb + 2) * 16 + 0, r2);
                vst1q_s16(pi2_tmp + (cb + 3) * 16 + 0, r3);

                /* Transpose s1 (8 vectors of 4 elements) to 4 vectors of 8 elements and store to pi2_tmp at col 8 */
                int16x4x2_t v0 = vtrn_s16(s1[j][0], s1[j][1]);
                int16x4x2_t v1 = vtrn_s16(s1[j][2], s1[j][3]);
                int32x2x2_t w0 = vtrn_s32(vreinterpret_s32_s16(v0.val[0]), vreinterpret_s32_s16(v1.val[0]));
                int32x2x2_t w1 = vtrn_s32(vreinterpret_s32_s16(v0.val[1]), vreinterpret_s32_s16(v1.val[1]));
                int16x4x2_t v2 = vtrn_s16(s1[j][4], s1[j][5]);
                int16x4x2_t v3 = vtrn_s16(s1[j][6], s1[j][7]);
                int32x2x2_t w2 = vtrn_s32(vreinterpret_s32_s16(v2.val[0]), vreinterpret_s32_s16(v3.val[0]));
                int32x2x2_t w3 = vtrn_s32(vreinterpret_s32_s16(v2.val[1]), vreinterpret_s32_s16(v3.val[1]));
                int16x8_t r4 = vcombine_s16(vreinterpret_s16_s32(w0.val[0]), vreinterpret_s16_s32(w2.val[0]));
                int16x8_t r5 = vcombine_s16(vreinterpret_s16_s32(w1.val[0]), vreinterpret_s16_s32(w3.val[0]));
                int16x8_t r6 = vcombine_s16(vreinterpret_s16_s32(w0.val[1]), vreinterpret_s16_s32(w2.val[1]));
                int16x8_t r7 = vcombine_s16(vreinterpret_s16_s32(w1.val[1]), vreinterpret_s16_s32(w3.val[1]));
                vst1q_s16(pi2_tmp + (cb + 0) * 16 + 8, r4);
                vst1q_s16(pi2_tmp + (cb + 1) * 16 + 8, r5);
                vst1q_s16(pi2_tmp + (cb + 2) * 16 + 8, r6);
                vst1q_s16(pi2_tmp + (cb + 3) * 16 + 8, r7);
            }
        }
    }

    /* Stage 2 (Fused with Transpose & Recon - 2-Pass Pipeline) */
    int32x4_t neg_shift = vdupq_n_s32(-(20 - bit_depth));
    int16x8_t clip_min = vdupq_n_s16(0);
    int16x8_t clip_max = vdupq_n_s16((1 << bit_depth) - 1);

    /* i = 0 => lower 8 samples */
    /* i = 1 => higher 8 samples */
    for (WORD32 i = 0; i < 2; i++)
    {
        WORD32 sample_half_index = i << 3;
        int32x4_t e[2][8];
        int16x4_t s0[2][8], s1[2][8];

        /* If last 12 rows are zero */
        if (zero_last12_rows_stg2)
        {
            int16x8_t r0_vec = vld1q_s16(pi2_tmp + 0 * 16 + sample_half_index);
            int16x8_t r1_vec = vld1q_s16(pi2_tmp + 1 * 16 + sample_half_index);
            int16x8_t r2_vec = vld1q_s16(pi2_tmp + 2 * 16 + sample_half_index);
            int16x8_t r3_vec = vld1q_s16(pi2_tmp + 3 * 16 + sample_half_index);

            int16x4_t r0_lo = vget_low_s16(r0_vec);
            int16x4_t r0_hi = vget_high_s16(r0_vec);
            int16x4_t r1_lo = vget_low_s16(r1_vec);
            int16x4_t r1_hi = vget_high_s16(r1_vec);
            int16x4_t r2_lo = vget_low_s16(r2_vec);
            int16x4_t r2_hi = vget_high_s16(r2_vec);
            int16x4_t r3_lo = vget_low_s16(r3_vec);
            int16x4_t r3_hi = vget_high_s16(r3_vec);

            /* eee */
            int32x4_t ee_lo = vshll_n_s16(r0_lo, 6);
            int32x4_t ee_hi = vshll_n_s16(r0_hi, 6);

            /* eo */

            /* eo0[0-3] */
            {
                int32x4_t eo0 = vmull_lane_s16(r2_lo, ce1, 0);
                e[0][0] = vaddq_s32(ee_lo, eo0);
                e[0][7] = vsubq_s32(ee_lo, eo0);
            }

            /* eo0[4-7] */
            {
                int32x4_t eo0 = vmull_lane_s16(r2_hi, ce1, 0);
                e[1][0] = vaddq_s32(ee_hi, eo0);
                e[1][7] = vsubq_s32(ee_hi, eo0);
            }

            /* eo1[0-3] */
            {
                int32x4_t eo1 = vmull_lane_s16(r2_lo, ce1, 1);
                e[0][1] = vaddq_s32(ee_lo, eo1);
                e[0][6] = vsubq_s32(ee_lo, eo1);
            }

            /* eo1[4-7] */
            {
                int32x4_t eo1 = vmull_lane_s16(r2_hi, ce1, 1);
                e[1][1] = vaddq_s32(ee_hi, eo1);
                e[1][6] = vsubq_s32(ee_hi, eo1);
            }

            /* eo2[0-3] */
            {
                int32x4_t eo2 = vmull_lane_s16(r2_lo, ce1, 2);
                e[0][2] = vaddq_s32(ee_lo, eo2);
                e[0][5] = vsubq_s32(ee_lo, eo2);
            }

            /* eo2[4-7] */
            {
                int32x4_t eo2 = vmull_lane_s16(r2_hi, ce1, 2);
                e[1][2] = vaddq_s32(ee_hi, eo2);
                e[1][5] = vsubq_s32(ee_hi, eo2);
            }

            /* eo3[0-3] */
            {
                int32x4_t eo3 = vmull_lane_s16(r2_lo, ce1, 3);
                e[0][3] = vaddq_s32(ee_lo, eo3);
                e[0][4] = vsubq_s32(ee_lo, eo3);
            }

            /* eo3[4-7] */
            {
                int32x4_t eo3 = vmull_lane_s16(r2_hi, ce1, 3);
                e[1][3] = vaddq_s32(ee_hi, eo3);
                e[1][4] = vsubq_s32(ee_hi, eo3);
            }

            /* o & stage 2 pre-transposed out */
            {
                WORD32 j;
                for (j = 0; j < 2; j++)
                {
                    int16x4_t r1 = j ? r1_hi : r1_lo;
                    int16x4_t r3 = j ? r3_hi : r3_lo;

                    /* o0[0-3] */
                    {
                        int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 0), r3, co0, 1);
                        s0[j][0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][0], o0), neg_shift));
                        s1[j][7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][0], o0), neg_shift));
                    }

                    /* o1[0-3] */
                    {
                        int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 1), r3, co1, 0);
                        s0[j][1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][1], o1), neg_shift));
                        s1[j][6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][1], o1), neg_shift));
                    }

                    /* o2[0-3] */
                    {
                        int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 2), r3, co1, 3);
                        s0[j][2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][2], o2), neg_shift));
                        s1[j][5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][2], o2), neg_shift));
                    }

                    /* o3[0-3] */
                    {
                        int32x4_t o3 = vmlsl_lane_s16(vmull_lane_s16(r1, co0, 3), r3, co1, 1);
                        s0[j][3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][3], o3), neg_shift));
                        s1[j][4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][3], o3), neg_shift));
                    }

                    /* o4[0-3] */
                    {
                        int32x4_t o4 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 0), r3, co0, 2);
                        s0[j][4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][4], o4), neg_shift));
                        s1[j][3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][4], o4), neg_shift));
                    }

                    /* o5[0-3] */
                    {
                        int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 1), r3, co0, 0);
                        s0[j][5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][5], o5), neg_shift));
                        s1[j][2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][5], o5), neg_shift));
                    }

                    /* o6[0-3] */
                    {
                        int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 2), r3, co0, 3);
                        s0[j][6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][6], o6), neg_shift));
                        s1[j][1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][6], o6), neg_shift));
                    }

                    /* o7[0-3] */
                    {
                        int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 3), r3, co1, 2);
                        s0[j][7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][7], o7), neg_shift));
                        s1[j][0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][7], o7), neg_shift));
                    }
                }

                for (WORD32 k = 0; k < 2; k++)
                {
                    WORD32 row_out = sample_half_index + (k << 2);

                    /* Transpose s0[k] (8 vectors of 4 elements) to 4 vectors of 8 elements: cols 0..7 */
                    int16x4x2_t t0 = vtrn_s16(s0[k][0], s0[k][1]);
                    int16x4x2_t t1 = vtrn_s16(s0[k][2], s0[k][3]);
                    int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                    int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                    int16x4x2_t t2 = vtrn_s16(s0[k][4], s0[k][5]);
                    int16x4x2_t t3 = vtrn_s16(s0[k][6], s0[k][7]);
                    int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                    int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                    int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                    int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                    int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                    int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));

                    /* Transpose s1[k] (8 vectors of 4 elements) to 4 vectors of 8 elements: cols 8..15 */
                    int16x4x2_t v0 = vtrn_s16(s1[k][0], s1[k][1]);
                    int16x4x2_t v1 = vtrn_s16(s1[k][2], s1[k][3]);
                    int32x2x2_t w0 = vtrn_s32(vreinterpret_s32_s16(v0.val[0]), vreinterpret_s32_s16(v1.val[0]));
                    int32x2x2_t w1 = vtrn_s32(vreinterpret_s32_s16(v0.val[1]), vreinterpret_s32_s16(v1.val[1]));
                    int16x4x2_t v2 = vtrn_s16(s1[k][4], s1[k][5]);
                    int16x4x2_t v3 = vtrn_s16(s1[k][6], s1[k][7]);
                    int32x2x2_t w2 = vtrn_s32(vreinterpret_s32_s16(v2.val[0]), vreinterpret_s32_s16(v3.val[0]));
                    int32x2x2_t w3 = vtrn_s32(vreinterpret_s32_s16(v2.val[1]), vreinterpret_s32_s16(v3.val[1]));
                    int16x8_t r4 = vcombine_s16(vreinterpret_s16_s32(w0.val[0]), vreinterpret_s16_s32(w2.val[0]));
                    int16x8_t r5 = vcombine_s16(vreinterpret_s16_s32(w1.val[0]), vreinterpret_s16_s32(w3.val[0]));
                    int16x8_t r6 = vcombine_s16(vreinterpret_s16_s32(w0.val[1]), vreinterpret_s16_s32(w2.val[1]));
                    int16x8_t r7 = vcombine_s16(vreinterpret_s16_s32(w1.val[1]), vreinterpret_s16_s32(w3.val[1]));

                    /* Row 0 */
                    uint16x8x2_t pred0 = vld2q_u16(pu2_pred + (row_out + 0) * pred_strd + 0);
                    uint16x8x2_t dst0  = vld2q_u16(pu2_dst  + (row_out + 0) * dst_strd  + 0);
                    int16x8_t acc0 = vminq_s16(vmaxq_s16(vqaddq_s16(r0, vreinterpretq_s16_u16(pred0.val[0])), clip_min), clip_max);
                    dst0.val[0] = vreinterpretq_u16_s16(acc0);
                    vst2q_u16(pu2_dst + (row_out + 0) * dst_strd + 0, dst0);

                    uint16x8x2_t pred4 = vld2q_u16(pu2_pred + (row_out + 0) * pred_strd + 16);
                    uint16x8x2_t dst4  = vld2q_u16(pu2_dst  + (row_out + 0) * dst_strd  + 16);
                    int16x8_t acc4 = vminq_s16(vmaxq_s16(vqaddq_s16(r4, vreinterpretq_s16_u16(pred4.val[0])), clip_min), clip_max);
                    dst4.val[0] = vreinterpretq_u16_s16(acc4);
                    vst2q_u16(pu2_dst + (row_out + 0) * dst_strd + 16, dst4);

                    /* Row 1 */
                    uint16x8x2_t pred1 = vld2q_u16(pu2_pred + (row_out + 1) * pred_strd + 0);
                    uint16x8x2_t dst1  = vld2q_u16(pu2_dst  + (row_out + 1) * dst_strd  + 0);
                    int16x8_t acc1 = vminq_s16(vmaxq_s16(vqaddq_s16(r1, vreinterpretq_s16_u16(pred1.val[0])), clip_min), clip_max);
                    dst1.val[0] = vreinterpretq_u16_s16(acc1);
                    vst2q_u16(pu2_dst + (row_out + 1) * dst_strd + 0, dst1);

                    uint16x8x2_t pred5 = vld2q_u16(pu2_pred + (row_out + 1) * pred_strd + 16);
                    uint16x8x2_t dst5  = vld2q_u16(pu2_dst  + (row_out + 1) * dst_strd  + 16);
                    int16x8_t acc5 = vminq_s16(vmaxq_s16(vqaddq_s16(r5, vreinterpretq_s16_u16(pred5.val[0])), clip_min), clip_max);
                    dst5.val[0] = vreinterpretq_u16_s16(acc5);
                    vst2q_u16(pu2_dst + (row_out + 1) * dst_strd + 16, dst5);

                    /* Row 2 */
                    uint16x8x2_t pred2 = vld2q_u16(pu2_pred + (row_out + 2) * pred_strd + 0);
                    uint16x8x2_t dst2  = vld2q_u16(pu2_dst  + (row_out + 2) * dst_strd  + 0);
                    int16x8_t acc2 = vminq_s16(vmaxq_s16(vqaddq_s16(r2, vreinterpretq_s16_u16(pred2.val[0])), clip_min), clip_max);
                    dst2.val[0] = vreinterpretq_u16_s16(acc2);
                    vst2q_u16(pu2_dst + (row_out + 2) * dst_strd + 0, dst2);

                    uint16x8x2_t pred6 = vld2q_u16(pu2_pred + (row_out + 2) * pred_strd + 16);
                    uint16x8x2_t dst6  = vld2q_u16(pu2_dst  + (row_out + 2) * dst_strd  + 16);
                    int16x8_t acc6 = vminq_s16(vmaxq_s16(vqaddq_s16(r6, vreinterpretq_s16_u16(pred6.val[0])), clip_min), clip_max);
                    dst6.val[0] = vreinterpretq_u16_s16(acc6);
                    vst2q_u16(pu2_dst + (row_out + 2) * dst_strd + 16, dst6);

                    /* Row 3 */
                    uint16x8x2_t pred3 = vld2q_u16(pu2_pred + (row_out + 3) * pred_strd + 0);
                    uint16x8x2_t dst3  = vld2q_u16(pu2_dst  + (row_out + 3) * dst_strd  + 0);
                    int16x8_t acc3 = vminq_s16(vmaxq_s16(vqaddq_s16(r3, vreinterpretq_s16_u16(pred3.val[0])), clip_min), clip_max);
                    dst3.val[0] = vreinterpretq_u16_s16(acc3);
                    vst2q_u16(pu2_dst + (row_out + 3) * dst_strd + 0, dst3);

                    uint16x8x2_t pred7 = vld2q_u16(pu2_pred + (row_out + 3) * pred_strd + 16);
                    uint16x8x2_t dst7  = vld2q_u16(pu2_dst  + (row_out + 3) * dst_strd  + 16);
                    int16x8_t acc7 = vminq_s16(vmaxq_s16(vqaddq_s16(r7, vreinterpretq_s16_u16(pred7.val[0])), clip_min), clip_max);
                    dst7.val[0] = vreinterpretq_u16_s16(acc7);
                    vst2q_u16(pu2_dst + (row_out + 3) * dst_strd + 16, dst7);
                }
            }
        }
        /* If last 8 rows are zero */
        else if (zero_last8_rows_stg2)
        {
            int16x8_t r0_vec = vld1q_s16(pi2_tmp + 0 * 16 + sample_half_index);
            int16x8_t r1_vec = vld1q_s16(pi2_tmp + 1 * 16 + sample_half_index);
            int16x8_t r2_vec = vld1q_s16(pi2_tmp + 2 * 16 + sample_half_index);
            int16x8_t r3_vec = vld1q_s16(pi2_tmp + 3 * 16 + sample_half_index);
            int16x8_t r4_vec = vld1q_s16(pi2_tmp + 4 * 16 + sample_half_index);
            int16x8_t r5_vec = vld1q_s16(pi2_tmp + 5 * 16 + sample_half_index);
            int16x8_t r6_vec = vld1q_s16(pi2_tmp + 6 * 16 + sample_half_index);
            int16x8_t r7_vec = vld1q_s16(pi2_tmp + 7 * 16 + sample_half_index);

            int16x4_t r0_lo = vget_low_s16(r0_vec);
            int16x4_t r0_hi = vget_high_s16(r0_vec);
            int16x4_t r2_lo = vget_low_s16(r2_vec);
            int16x4_t r2_hi = vget_high_s16(r2_vec);
            int16x4_t r4_lo = vget_low_s16(r4_vec);
            int16x4_t r4_hi = vget_high_s16(r4_vec);
            int16x4_t r6_lo = vget_low_s16(r6_vec);
            int16x4_t r6_hi = vget_high_s16(r6_vec);

            int16x4_t r1_lo = vget_low_s16(r1_vec);
            int16x4_t r1_hi = vget_high_s16(r1_vec);
            int16x4_t r3_lo = vget_low_s16(r3_vec);
            int16x4_t r3_hi = vget_high_s16(r3_vec);
            int16x4_t r5_lo = vget_low_s16(r5_vec);
            int16x4_t r5_hi = vget_high_s16(r5_vec);
            int16x4_t r7_lo = vget_low_s16(r7_vec);
            int16x4_t r7_hi = vget_high_s16(r7_vec);

            /* eeo */
            int32x4_t eeo0_lo = vmull_lane_s16(r4_lo, ce0, 0);
            int32x4_t eeo1_lo = vmull_lane_s16(r4_lo, ce0, 1);
            int32x4_t eeo0_hi = vmull_lane_s16(r4_hi, ce0, 0);
            int32x4_t eeo1_hi = vmull_lane_s16(r4_hi, ce0, 1);

            /* eee */
            int32x4_t eee_lo = vshll_n_s16(r0_lo, 6);
            int32x4_t ee0_lo = vaddq_s32(eee_lo, eeo0_lo);
            int32x4_t ee1_lo = vaddq_s32(eee_lo, eeo1_lo);
            int32x4_t ee2_lo = vsubq_s32(eee_lo, eeo1_lo);
            int32x4_t ee3_lo = vsubq_s32(eee_lo, eeo0_lo);

            int32x4_t eee_hi = vshll_n_s16(r0_hi, 6);
            int32x4_t ee0_hi = vaddq_s32(eee_hi, eeo0_hi);
            int32x4_t ee1_hi = vaddq_s32(eee_hi, eeo1_hi);
            int32x4_t ee2_hi = vsubq_s32(eee_hi, eeo1_hi);
            int32x4_t ee3_hi = vsubq_s32(eee_hi, eeo0_hi);

            /* eo */

            /* eo0[0-3] */
            {
                int32x4_t eo0 = vmlal_lane_s16(vmull_lane_s16(r2_lo, ce1, 0), r6_lo, ce1, 1);
                e[0][0] = vaddq_s32(ee0_lo, eo0);
                e[0][7] = vsubq_s32(ee0_lo, eo0);
            }

            /* eo0[4-7] */
            {
                int32x4_t eo0 = vmlal_lane_s16(vmull_lane_s16(r2_hi, ce1, 0), r6_hi, ce1, 1);
                e[1][0] = vaddq_s32(ee0_hi, eo0);
                e[1][7] = vsubq_s32(ee0_hi, eo0);
            }

            /* eo1[0-3] */
            {
                int32x4_t eo1 = vmlsl_lane_s16(vmull_lane_s16(r2_lo, ce1, 1), r6_lo, ce1, 3);
                e[0][1] = vaddq_s32(ee1_lo, eo1);
                e[0][6] = vsubq_s32(ee1_lo, eo1);
            }

            /* eo1[4-7] */
            {
                int32x4_t eo1 = vmlsl_lane_s16(vmull_lane_s16(r2_hi, ce1, 1), r6_hi, ce1, 3);
                e[1][1] = vaddq_s32(ee1_hi, eo1);
                e[1][6] = vsubq_s32(ee1_hi, eo1);
            }

            /* eo2[0-3] */
            {
                int32x4_t eo2 = vmlsl_lane_s16(vmull_lane_s16(r2_lo, ce1, 2), r6_lo, ce1, 0);
                e[0][2] = vaddq_s32(ee2_lo, eo2);
                e[0][5] = vsubq_s32(ee2_lo, eo2);
            }

            /* eo2[4-7] */
            {
                int32x4_t eo2 = vmlsl_lane_s16(vmull_lane_s16(r2_hi, ce1, 2), r6_hi, ce1, 0);
                e[1][2] = vaddq_s32(ee2_hi, eo2);
                e[1][5] = vsubq_s32(ee2_hi, eo2);
            }

            /* eo3[0-3] */
            {
                int32x4_t eo3 = vmlsl_lane_s16(vmull_lane_s16(r2_lo, ce1, 3), r6_lo, ce1, 2);
                e[0][3] = vaddq_s32(ee3_lo, eo3);
                e[0][4] = vsubq_s32(ee3_lo, eo3);
            }

            /* eo3[4-7] */
            {
                int32x4_t eo3 = vmlsl_lane_s16(vmull_lane_s16(r2_hi, ce1, 3), r6_hi, ce1, 2);
                e[1][3] = vaddq_s32(ee3_hi, eo3);
                e[1][4] = vsubq_s32(ee3_hi, eo3);
            }

            /* o & stage 2 pre-transposed out */
            {
                WORD32 j;
                for (j = 0; j < 2; j++)
                {
                    int16x4_t r1 = j ? r1_hi : r1_lo;
                    int16x4_t r3 = j ? r3_hi : r3_lo;
                    int16x4_t r5 = j ? r5_hi : r5_lo;
                    int16x4_t r7 = j ? r7_hi : r7_lo;

                    /* o0[0-3] */
                    {
                        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
                        o0 = vmlal_lane_s16(o0, r3, co0, 1);
                        o0 = vmlal_lane_s16(o0, r5, co0, 2);
                        o0 = vmlal_lane_s16(o0, r7, co0, 3);
                        s0[j][0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][0], o0), neg_shift));
                        s1[j][7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][0], o0), neg_shift));
                    }

                    /* o1[0-3] */
                    {
                        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
                        o1 = vmlal_lane_s16(o1, r3, co1, 0);
                        o1 = vmlal_lane_s16(o1, r5, co1, 3);
                        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
                        s0[j][1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][1], o1), neg_shift));
                        s1[j][6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][1], o1), neg_shift));
                    }

                    /* o2[0-3] */
                    {
                        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
                        o2 = vmlal_lane_s16(o2, r3, co1, 3);
                        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
                        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
                        s0[j][2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][2], o2), neg_shift));
                        s1[j][5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][2], o2), neg_shift));
                    }

                    /* o3[0-3] */
                    {
                        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
                        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
                        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
                        o3 = vmlal_lane_s16(o3, r7, co1, 3);
                        s0[j][3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][3], o3), neg_shift));
                        s1[j][4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][3], o3), neg_shift));
                    }

                    /* o4[0-3] */
                    {
                        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
                        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
                        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
                        o4 = vmlal_lane_s16(o4, r7, co0, 0);
                        s0[j][4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][4], o4), neg_shift));
                        s1[j][3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][4], o4), neg_shift));
                    }

                    /* o5[0-3] */
                    {
                        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
                        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
                        o5 = vmlal_lane_s16(o5, r5, co1, 0);
                        o5 = vmlal_lane_s16(o5, r7, co1, 2);
                        s0[j][5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][5], o5), neg_shift));
                        s1[j][2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][5], o5), neg_shift));
                    }

                    /* o6[0-3] */
                    {
                        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
                        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
                        o6 = vmlal_lane_s16(o6, r5, co0, 0);
                        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
                        s0[j][6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][6], o6), neg_shift));
                        s1[j][1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][6], o6), neg_shift));
                    }

                    /* o7[0-3] */
                    {
                        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
                        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
                        o7 = vmlal_lane_s16(o7, r5, co1, 1);
                        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
                        s0[j][7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][7], o7), neg_shift));
                        s1[j][0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][7], o7), neg_shift));
                    }
                }

                for (WORD32 k = 0; k < 2; k++)
                {
                    WORD32 row_out = sample_half_index + (k << 2);

                    /* Transpose s0[k] (8 vectors of 4 elements) to 4 vectors of 8 elements: cols 0..7 */
                    int16x4x2_t t0 = vtrn_s16(s0[k][0], s0[k][1]);
                    int16x4x2_t t1 = vtrn_s16(s0[k][2], s0[k][3]);
                    int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                    int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                    int16x4x2_t t2 = vtrn_s16(s0[k][4], s0[k][5]);
                    int16x4x2_t t3 = vtrn_s16(s0[k][6], s0[k][7]);
                    int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                    int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                    int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                    int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                    int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                    int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));

                    /* Transpose s1[k] (8 vectors of 4 elements) to 4 vectors of 8 elements: cols 8..15 */
                    int16x4x2_t v0 = vtrn_s16(s1[k][0], s1[k][1]);
                    int16x4x2_t v1 = vtrn_s16(s1[k][2], s1[k][3]);
                    int32x2x2_t w0 = vtrn_s32(vreinterpret_s32_s16(v0.val[0]), vreinterpret_s32_s16(v1.val[0]));
                    int32x2x2_t w1 = vtrn_s32(vreinterpret_s32_s16(v0.val[1]), vreinterpret_s32_s16(v1.val[1]));
                    int16x4x2_t v2 = vtrn_s16(s1[k][4], s1[k][5]);
                    int16x4x2_t v3 = vtrn_s16(s1[k][6], s1[k][7]);
                    int32x2x2_t w2 = vtrn_s32(vreinterpret_s32_s16(v2.val[0]), vreinterpret_s32_s16(v3.val[0]));
                    int32x2x2_t w3 = vtrn_s32(vreinterpret_s32_s16(v2.val[1]), vreinterpret_s32_s16(v3.val[1]));
                    int16x8_t r4 = vcombine_s16(vreinterpret_s16_s32(w0.val[0]), vreinterpret_s16_s32(w2.val[0]));
                    int16x8_t r5 = vcombine_s16(vreinterpret_s16_s32(w1.val[0]), vreinterpret_s16_s32(w3.val[0]));
                    int16x8_t r6 = vcombine_s16(vreinterpret_s16_s32(w0.val[1]), vreinterpret_s16_s32(w2.val[1]));
                    int16x8_t r7 = vcombine_s16(vreinterpret_s16_s32(w1.val[1]), vreinterpret_s16_s32(w3.val[1]));

                    /* Row 0 */
                    uint16x8x2_t pred0 = vld2q_u16(pu2_pred + (row_out + 0) * pred_strd + 0);
                    uint16x8x2_t dst0  = vld2q_u16(pu2_dst  + (row_out + 0) * dst_strd  + 0);
                    int16x8_t acc0 = vminq_s16(vmaxq_s16(vqaddq_s16(r0, vreinterpretq_s16_u16(pred0.val[0])), clip_min), clip_max);
                    dst0.val[0] = vreinterpretq_u16_s16(acc0);
                    vst2q_u16(pu2_dst + (row_out + 0) * dst_strd + 0, dst0);

                    uint16x8x2_t pred4 = vld2q_u16(pu2_pred + (row_out + 0) * pred_strd + 16);
                    uint16x8x2_t dst4  = vld2q_u16(pu2_dst  + (row_out + 0) * dst_strd  + 16);
                    int16x8_t acc4 = vminq_s16(vmaxq_s16(vqaddq_s16(r4, vreinterpretq_s16_u16(pred4.val[0])), clip_min), clip_max);
                    dst4.val[0] = vreinterpretq_u16_s16(acc4);
                    vst2q_u16(pu2_dst + (row_out + 0) * dst_strd + 16, dst4);

                    /* Row 1 */
                    uint16x8x2_t pred1 = vld2q_u16(pu2_pred + (row_out + 1) * pred_strd + 0);
                    uint16x8x2_t dst1  = vld2q_u16(pu2_dst  + (row_out + 1) * dst_strd  + 0);
                    int16x8_t acc1 = vminq_s16(vmaxq_s16(vqaddq_s16(r1, vreinterpretq_s16_u16(pred1.val[0])), clip_min), clip_max);
                    dst1.val[0] = vreinterpretq_u16_s16(acc1);
                    vst2q_u16(pu2_dst + (row_out + 1) * dst_strd + 0, dst1);

                    uint16x8x2_t pred5 = vld2q_u16(pu2_pred + (row_out + 1) * pred_strd + 16);
                    uint16x8x2_t dst5  = vld2q_u16(pu2_dst  + (row_out + 1) * dst_strd  + 16);
                    int16x8_t acc5 = vminq_s16(vmaxq_s16(vqaddq_s16(r5, vreinterpretq_s16_u16(pred5.val[0])), clip_min), clip_max);
                    dst5.val[0] = vreinterpretq_u16_s16(acc5);
                    vst2q_u16(pu2_dst + (row_out + 1) * dst_strd + 16, dst5);

                    /* Row 2 */
                    uint16x8x2_t pred2 = vld2q_u16(pu2_pred + (row_out + 2) * pred_strd + 0);
                    uint16x8x2_t dst2  = vld2q_u16(pu2_dst  + (row_out + 2) * dst_strd  + 0);
                    int16x8_t acc2 = vminq_s16(vmaxq_s16(vqaddq_s16(r2, vreinterpretq_s16_u16(pred2.val[0])), clip_min), clip_max);
                    dst2.val[0] = vreinterpretq_u16_s16(acc2);
                    vst2q_u16(pu2_dst + (row_out + 2) * dst_strd + 0, dst2);

                    uint16x8x2_t pred6 = vld2q_u16(pu2_pred + (row_out + 2) * pred_strd + 16);
                    uint16x8x2_t dst6  = vld2q_u16(pu2_dst  + (row_out + 2) * dst_strd  + 16);
                    int16x8_t acc6 = vminq_s16(vmaxq_s16(vqaddq_s16(r6, vreinterpretq_s16_u16(pred6.val[0])), clip_min), clip_max);
                    dst6.val[0] = vreinterpretq_u16_s16(acc6);
                    vst2q_u16(pu2_dst + (row_out + 2) * dst_strd + 16, dst6);

                    /* Row 3 */
                    uint16x8x2_t pred3 = vld2q_u16(pu2_pred + (row_out + 3) * pred_strd + 0);
                    uint16x8x2_t dst3  = vld2q_u16(pu2_dst  + (row_out + 3) * dst_strd  + 0);
                    int16x8_t acc3 = vminq_s16(vmaxq_s16(vqaddq_s16(r3, vreinterpretq_s16_u16(pred3.val[0])), clip_min), clip_max);
                    dst3.val[0] = vreinterpretq_u16_s16(acc3);
                    vst2q_u16(pu2_dst + (row_out + 3) * dst_strd + 0, dst3);

                    uint16x8x2_t pred7 = vld2q_u16(pu2_pred + (row_out + 3) * pred_strd + 16);
                    uint16x8x2_t dst7  = vld2q_u16(pu2_dst  + (row_out + 3) * dst_strd  + 16);
                    int16x8_t acc7 = vminq_s16(vmaxq_s16(vqaddq_s16(r7, vreinterpretq_s16_u16(pred7.val[0])), clip_min), clip_max);
                    dst7.val[0] = vreinterpretq_u16_s16(acc7);
                    vst2q_u16(pu2_dst + (row_out + 3) * dst_strd + 16, dst7);
                }
            }
        }
        /* If all the rows are non-zero */
        else
        {
            int16x8_t r0_vec = vld1q_s16(pi2_tmp + 0 * 16 + sample_half_index);
            int16x8_t r1_vec = vld1q_s16(pi2_tmp + 1 * 16 + sample_half_index);
            int16x8_t r2_vec = vld1q_s16(pi2_tmp + 2 * 16 + sample_half_index);
            int16x8_t r3_vec = vld1q_s16(pi2_tmp + 3 * 16 + sample_half_index);
            int16x8_t r4_vec = vld1q_s16(pi2_tmp + 4 * 16 + sample_half_index);
            int16x8_t r5_vec = vld1q_s16(pi2_tmp + 5 * 16 + sample_half_index);
            int16x8_t r6_vec = vld1q_s16(pi2_tmp + 6 * 16 + sample_half_index);
            int16x8_t r7_vec = vld1q_s16(pi2_tmp + 7 * 16 + sample_half_index);
            int16x8_t r8_vec = vld1q_s16(pi2_tmp + 8 * 16 + sample_half_index);
            int16x8_t r9_vec = vld1q_s16(pi2_tmp + 9 * 16 + sample_half_index);
            int16x8_t r10_vec = vld1q_s16(pi2_tmp + 10 * 16 + sample_half_index);
            int16x8_t r11_vec = vld1q_s16(pi2_tmp + 11 * 16 + sample_half_index);
            int16x8_t r12_vec = vld1q_s16(pi2_tmp + 12 * 16 + sample_half_index);
            int16x8_t r13_vec = vld1q_s16(pi2_tmp + 13 * 16 + sample_half_index);
            int16x8_t r14_vec = vld1q_s16(pi2_tmp + 14 * 16 + sample_half_index);
            int16x8_t r15_vec = vld1q_s16(pi2_tmp + 15 * 16 + sample_half_index);

            int16x4_t r0_lo = vget_low_s16(r0_vec);
            int16x4_t r0_hi = vget_high_s16(r0_vec);
            int16x4_t r2_lo = vget_low_s16(r2_vec);
            int16x4_t r2_hi = vget_high_s16(r2_vec);
            int16x4_t r4_lo = vget_low_s16(r4_vec);
            int16x4_t r4_hi = vget_high_s16(r4_vec);
            int16x4_t r6_lo = vget_low_s16(r6_vec);
            int16x4_t r6_hi = vget_high_s16(r6_vec);
            int16x4_t r8_lo = vget_low_s16(r8_vec);
            int16x4_t r8_hi = vget_high_s16(r8_vec);
            int16x4_t r10_lo = vget_low_s16(r10_vec);
            int16x4_t r10_hi = vget_high_s16(r10_vec);
            int16x4_t r12_lo = vget_low_s16(r12_vec);
            int16x4_t r12_hi = vget_high_s16(r12_vec);
            int16x4_t r14_lo = vget_low_s16(r14_vec);
            int16x4_t r14_hi = vget_high_s16(r14_vec);

            int16x4_t r1_lo = vget_low_s16(r1_vec);
            int16x4_t r1_hi = vget_high_s16(r1_vec);
            int16x4_t r3_lo = vget_low_s16(r3_vec);
            int16x4_t r3_hi = vget_high_s16(r3_vec);
            int16x4_t r5_lo = vget_low_s16(r5_vec);
            int16x4_t r5_hi = vget_high_s16(r5_vec);
            int16x4_t r7_lo = vget_low_s16(r7_vec);
            int16x4_t r7_hi = vget_high_s16(r7_vec);
            int16x4_t r9_lo = vget_low_s16(r9_vec);
            int16x4_t r9_hi = vget_high_s16(r9_vec);
            int16x4_t r11_lo = vget_low_s16(r11_vec);
            int16x4_t r11_hi = vget_high_s16(r11_vec);
            int16x4_t r13_lo = vget_low_s16(r13_vec);
            int16x4_t r13_hi = vget_high_s16(r13_vec);
            int16x4_t r15_lo = vget_low_s16(r15_vec);
            int16x4_t r15_hi = vget_high_s16(r15_vec);

            /* eee */
            int32x4_t eee0_lo = vshlq_n_s32(vaddl_s16(r0_lo, r8_lo), 6);
            int32x4_t eee1_lo = vshlq_n_s32(vsubl_s16(r0_lo, r8_lo), 6);
            int32x4_t eee0_hi = vshlq_n_s32(vaddl_s16(r0_hi, r8_hi), 6);
            int32x4_t eee1_hi = vshlq_n_s32(vsubl_s16(r0_hi, r8_hi), 6);

            /* eeo */
            int32x4_t eeo0_lo = vmlal_lane_s16(vmull_lane_s16(r4_lo, ce0, 0), r12_lo, ce0, 1);
            int32x4_t eeo1_lo = vmlsl_lane_s16(vmull_lane_s16(r4_lo, ce0, 1), r12_lo, ce0, 0);
            int32x4_t eeo0_hi = vmlal_lane_s16(vmull_lane_s16(r4_hi, ce0, 0), r12_hi, ce0, 1);
            int32x4_t eeo1_hi = vmlsl_lane_s16(vmull_lane_s16(r4_hi, ce0, 1), r12_hi, ce0, 0);

            int32x4_t ee0_lo = vaddq_s32(eee0_lo, eeo0_lo);
            int32x4_t ee1_lo = vaddq_s32(eee1_lo, eeo1_lo);
            int32x4_t ee2_lo = vsubq_s32(eee1_lo, eeo1_lo);
            int32x4_t ee3_lo = vsubq_s32(eee0_lo, eeo0_lo);

            int32x4_t ee0_hi = vaddq_s32(eee0_hi, eeo0_hi);
            int32x4_t ee1_hi = vaddq_s32(eee1_hi, eeo1_hi);
            int32x4_t ee2_hi = vsubq_s32(eee1_hi, eeo1_hi);
            int32x4_t ee3_hi = vsubq_s32(eee0_hi, eeo0_hi);

            /* eo */

            /* eo0[0-3] */
            {
                int32x4_t eo0 = vmull_lane_s16(r2_lo, ce1, 0);
                eo0 = vmlal_lane_s16(eo0, r6_lo, ce1, 1);
                eo0 = vmlal_lane_s16(eo0, r10_lo, ce1, 2);
                eo0 = vmlal_lane_s16(eo0, r14_lo, ce1, 3);
                e[0][0] = vaddq_s32(ee0_lo, eo0);
                e[0][7] = vsubq_s32(ee0_lo, eo0);
            }

            /* eo0[4-7] */
            {
                int32x4_t eo0 = vmull_lane_s16(r2_hi, ce1, 0);
                eo0 = vmlal_lane_s16(eo0, r6_hi, ce1, 1);
                eo0 = vmlal_lane_s16(eo0, r10_hi, ce1, 2);
                eo0 = vmlal_lane_s16(eo0, r14_hi, ce1, 3);
                e[1][0] = vaddq_s32(ee0_hi, eo0);
                e[1][7] = vsubq_s32(ee0_hi, eo0);
            }

            /* eo1[0-3] */
            {
                int32x4_t eo1 = vmull_lane_s16(r2_lo, ce1, 1);
                eo1 = vmlsl_lane_s16(eo1, r6_lo, ce1, 3);
                eo1 = vmlsl_lane_s16(eo1, r10_lo, ce1, 0);
                eo1 = vmlsl_lane_s16(eo1, r14_lo, ce1, 2);
                e[0][1] = vaddq_s32(ee1_lo, eo1);
                e[0][6] = vsubq_s32(ee1_lo, eo1);
            }

            /* eo1[4-7] */
            {
                int32x4_t eo1 = vmull_lane_s16(r2_hi, ce1, 1);
                eo1 = vmlsl_lane_s16(eo1, r6_hi, ce1, 3);
                eo1 = vmlsl_lane_s16(eo1, r10_hi, ce1, 0);
                eo1 = vmlsl_lane_s16(eo1, r14_hi, ce1, 2);
                e[1][1] = vaddq_s32(ee1_hi, eo1);
                e[1][6] = vsubq_s32(ee1_hi, eo1);
            }

            /* eo2[0-3] */
            {
                int32x4_t eo2 = vmull_lane_s16(r2_lo, ce1, 2);
                eo2 = vmlsl_lane_s16(eo2, r6_lo, ce1, 0);
                eo2 = vmlal_lane_s16(eo2, r10_lo, ce1, 3);
                eo2 = vmlal_lane_s16(eo2, r14_lo, ce1, 1);
                e[0][2] = vaddq_s32(ee2_lo, eo2);
                e[0][5] = vsubq_s32(ee2_lo, eo2);
            }

            /* eo2[4-7] */
            {
                int32x4_t eo2 = vmull_lane_s16(r2_hi, ce1, 2);
                eo2 = vmlsl_lane_s16(eo2, r6_hi, ce1, 0);
                eo2 = vmlal_lane_s16(eo2, r10_hi, ce1, 3);
                eo2 = vmlal_lane_s16(eo2, r14_hi, ce1, 1);
                e[1][2] = vaddq_s32(ee2_hi, eo2);
                e[1][5] = vsubq_s32(ee2_hi, eo2);
            }

            /* eo3[0-3] */
            {
                int32x4_t eo3 = vmull_lane_s16(r2_lo, ce1, 3);
                eo3 = vmlsl_lane_s16(eo3, r6_lo, ce1, 2);
                eo3 = vmlal_lane_s16(eo3, r10_lo, ce1, 1);
                eo3 = vmlsl_lane_s16(eo3, r14_lo, ce1, 0);
                e[0][3] = vaddq_s32(ee3_lo, eo3);
                e[0][4] = vsubq_s32(ee3_lo, eo3);
            }

            /* eo3[4-7] */
            {
                int32x4_t eo3 = vmull_lane_s16(r2_hi, ce1, 3);
                eo3 = vmlsl_lane_s16(eo3, r6_hi, ce1, 2);
                eo3 = vmlal_lane_s16(eo3, r10_hi, ce1, 1);
                eo3 = vmlsl_lane_s16(eo3, r14_hi, ce1, 0);
                e[1][3] = vaddq_s32(ee3_hi, eo3);
                e[1][4] = vsubq_s32(ee3_hi, eo3);
            }

            /* o & stage 2 pre-transposed out */
            {
                WORD32 j;
                for (j = 0; j < 2; j++)
                {
                    int16x4_t r1 = j ? r1_hi : r1_lo;
                    int16x4_t r3 = j ? r3_hi : r3_lo;
                    int16x4_t r5 = j ? r5_hi : r5_lo;
                    int16x4_t r7 = j ? r7_hi : r7_lo;
                    int16x4_t r9 = j ? r9_hi : r9_lo;
                    int16x4_t r11 = j ? r11_hi : r11_lo;
                    int16x4_t r13 = j ? r13_hi : r13_lo;
                    int16x4_t r15 = j ? r15_hi : r15_lo;

                    /* o0[0-3] */
                    {
                        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
                        o0 = vmlal_lane_s16(o0, r3, co0, 1);
                        o0 = vmlal_lane_s16(o0, r5, co0, 2);
                        o0 = vmlal_lane_s16(o0, r7, co0, 3);
                        o0 = vmlal_lane_s16(o0, r9, co1, 0);
                        o0 = vmlal_lane_s16(o0, r11, co1, 1);
                        o0 = vmlal_lane_s16(o0, r13, co1, 2);
                        o0 = vmlal_lane_s16(o0, r15, co1, 3);
                        s0[j][0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][0], o0), neg_shift));
                        s1[j][7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][0], o0), neg_shift));
                    }

                    /* o1[0-3] */
                    {
                        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
                        o1 = vmlal_lane_s16(o1, r3, co1, 0);
                        o1 = vmlal_lane_s16(o1, r5, co1, 3);
                        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
                        o1 = vmlsl_lane_s16(o1, r9, co0, 2);
                        o1 = vmlsl_lane_s16(o1, r11, co0, 0);
                        o1 = vmlsl_lane_s16(o1, r13, co0, 3);
                        o1 = vmlsl_lane_s16(o1, r15, co1, 2);
                        s0[j][1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][1], o1), neg_shift));
                        s1[j][6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][1], o1), neg_shift));
                    }

                    /* o2[0-3] */
                    {
                        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
                        o2 = vmlal_lane_s16(o2, r3, co1, 3);
                        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
                        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
                        o2 = vmlsl_lane_s16(o2, r9, co1, 2);
                        o2 = vmlal_lane_s16(o2, r11, co1, 0);
                        o2 = vmlal_lane_s16(o2, r13, co0, 0);
                        o2 = vmlal_lane_s16(o2, r15, co1, 1);
                        s0[j][2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][2], o2), neg_shift));
                        s1[j][5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][2], o2), neg_shift));
                    }

                    /* o3[0-3] */
                    {
                        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
                        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
                        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
                        o3 = vmlal_lane_s16(o3, r7, co1, 3);
                        o3 = vmlal_lane_s16(o3, r9, co0, 0);
                        o3 = vmlal_lane_s16(o3, r11, co1, 2);
                        o3 = vmlsl_lane_s16(o3, r13, co0, 2);
                        o3 = vmlsl_lane_s16(o3, r15, co1, 0);
                        s0[j][3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][3], o3), neg_shift));
                        s1[j][4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][3], o3), neg_shift));
                    }

                    /* o4[0-3] */
                    {
                        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
                        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
                        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
                        o4 = vmlal_lane_s16(o4, r7, co0, 0);
                        o4 = vmlsl_lane_s16(o4, r9, co1, 3);
                        o4 = vmlsl_lane_s16(o4, r11, co0, 1);
                        o4 = vmlal_lane_s16(o4, r13, co1, 1);
                        o4 = vmlal_lane_s16(o4, r15, co0, 3);
                        s0[j][4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][4], o4), neg_shift));
                        s1[j][3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][4], o4), neg_shift));
                    }

                    /* o5[0-3] */
                    {
                        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
                        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
                        o5 = vmlal_lane_s16(o5, r5, co1, 0);
                        o5 = vmlal_lane_s16(o5, r7, co1, 2);
                        o5 = vmlsl_lane_s16(o5, r9, co0, 1);
                        o5 = vmlal_lane_s16(o5, r11, co0, 3);
                        o5 = vmlal_lane_s16(o5, r13, co1, 3);
                        o5 = vmlsl_lane_s16(o5, r15, co0, 2);
                        s0[j][5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][5], o5), neg_shift));
                        s1[j][2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][5], o5), neg_shift));
                    }

                    /* o6[0-3] */
                    {
                        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
                        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
                        o6 = vmlal_lane_s16(o6, r5, co0, 0);
                        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
                        o6 = vmlal_lane_s16(o6, r9, co1, 1);
                        o6 = vmlal_lane_s16(o6, r11, co1, 3);
                        o6 = vmlsl_lane_s16(o6, r13, co1, 0);
                        o6 = vmlal_lane_s16(o6, r15, co0, 1);
                        s0[j][6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][6], o6), neg_shift));
                        s1[j][1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][6], o6), neg_shift));
                    }

                    /* o7[0-3] */
                    {
                        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
                        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
                        o7 = vmlal_lane_s16(o7, r5, co1, 1);
                        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
                        o7 = vmlal_lane_s16(o7, r9, co0, 3);
                        o7 = vmlsl_lane_s16(o7, r11, co0, 2);
                        o7 = vmlal_lane_s16(o7, r13, co0, 1);
                        o7 = vmlsl_lane_s16(o7, r15, co0, 0);
                        s0[j][7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e[j][7], o7), neg_shift));
                        s1[j][0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e[j][7], o7), neg_shift));
                    }
                }

                for (WORD32 k = 0; k < 2; k++)
                {
                    WORD32 row_out = sample_half_index + (k << 2);

                    /* Transpose s0[k] (8 vectors of 4 elements) to 4 vectors of 8 elements: cols 0..7 */
                    int16x4x2_t t0 = vtrn_s16(s0[k][0], s0[k][1]);
                    int16x4x2_t t1 = vtrn_s16(s0[k][2], s0[k][3]);
                    int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]), vreinterpret_s32_s16(t1.val[0]));
                    int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]), vreinterpret_s32_s16(t1.val[1]));
                    int16x4x2_t t2 = vtrn_s16(s0[k][4], s0[k][5]);
                    int16x4x2_t t3 = vtrn_s16(s0[k][6], s0[k][7]);
                    int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]), vreinterpret_s32_s16(t3.val[0]));
                    int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]), vreinterpret_s32_s16(t3.val[1]));
                    int16x8_t r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
                    int16x8_t r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
                    int16x8_t r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
                    int16x8_t r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));

                    /* Transpose s1[k] (8 vectors of 4 elements) to 4 vectors of 8 elements: cols 8..15 */
                    int16x4x2_t v0 = vtrn_s16(s1[k][0], s1[k][1]);
                    int16x4x2_t v1 = vtrn_s16(s1[k][2], s1[k][3]);
                    int32x2x2_t w0 = vtrn_s32(vreinterpret_s32_s16(v0.val[0]), vreinterpret_s32_s16(v1.val[0]));
                    int32x2x2_t w1 = vtrn_s32(vreinterpret_s32_s16(v0.val[1]), vreinterpret_s32_s16(v1.val[1]));
                    int16x4x2_t v2 = vtrn_s16(s1[k][4], s1[k][5]);
                    int16x4x2_t v3 = vtrn_s16(s1[k][6], s1[k][7]);
                    int32x2x2_t w2 = vtrn_s32(vreinterpret_s32_s16(v2.val[0]), vreinterpret_s32_s16(v3.val[0]));
                    int32x2x2_t w3 = vtrn_s32(vreinterpret_s32_s16(v2.val[1]), vreinterpret_s32_s16(v3.val[1]));
                    int16x8_t r4 = vcombine_s16(vreinterpret_s16_s32(w0.val[0]), vreinterpret_s16_s32(w2.val[0]));
                    int16x8_t r5 = vcombine_s16(vreinterpret_s16_s32(w1.val[0]), vreinterpret_s16_s32(w3.val[0]));
                    int16x8_t r6 = vcombine_s16(vreinterpret_s16_s32(w0.val[1]), vreinterpret_s16_s32(w2.val[1]));
                    int16x8_t r7 = vcombine_s16(vreinterpret_s16_s32(w1.val[1]), vreinterpret_s16_s32(w3.val[1]));

                    /* Row 0 */
                    uint16x8x2_t pred0 = vld2q_u16(pu2_pred + (row_out + 0) * pred_strd + 0);
                    uint16x8x2_t dst0  = vld2q_u16(pu2_dst  + (row_out + 0) * dst_strd  + 0);
                    int16x8_t acc0 = vminq_s16(vmaxq_s16(vqaddq_s16(r0, vreinterpretq_s16_u16(pred0.val[0])), clip_min), clip_max);
                    dst0.val[0] = vreinterpretq_u16_s16(acc0);
                    vst2q_u16(pu2_dst + (row_out + 0) * dst_strd + 0, dst0);

                    uint16x8x2_t pred4 = vld2q_u16(pu2_pred + (row_out + 0) * pred_strd + 16);
                    uint16x8x2_t dst4  = vld2q_u16(pu2_dst  + (row_out + 0) * dst_strd  + 16);
                    int16x8_t acc4 = vminq_s16(vmaxq_s16(vqaddq_s16(r4, vreinterpretq_s16_u16(pred4.val[0])), clip_min), clip_max);
                    dst4.val[0] = vreinterpretq_u16_s16(acc4);
                    vst2q_u16(pu2_dst + (row_out + 0) * dst_strd + 16, dst4);

                    /* Row 1 */
                    uint16x8x2_t pred1 = vld2q_u16(pu2_pred + (row_out + 1) * pred_strd + 0);
                    uint16x8x2_t dst1  = vld2q_u16(pu2_dst  + (row_out + 1) * dst_strd  + 0);
                    int16x8_t acc1 = vminq_s16(vmaxq_s16(vqaddq_s16(r1, vreinterpretq_s16_u16(pred1.val[0])), clip_min), clip_max);
                    dst1.val[0] = vreinterpretq_u16_s16(acc1);
                    vst2q_u16(pu2_dst + (row_out + 1) * dst_strd + 0, dst1);

                    uint16x8x2_t pred5 = vld2q_u16(pu2_pred + (row_out + 1) * pred_strd + 16);
                    uint16x8x2_t dst5  = vld2q_u16(pu2_dst  + (row_out + 1) * dst_strd  + 16);
                    int16x8_t acc5 = vminq_s16(vmaxq_s16(vqaddq_s16(r5, vreinterpretq_s16_u16(pred5.val[0])), clip_min), clip_max);
                    dst5.val[0] = vreinterpretq_u16_s16(acc5);
                    vst2q_u16(pu2_dst + (row_out + 1) * dst_strd + 16, dst5);

                    /* Row 2 */
                    uint16x8x2_t pred2 = vld2q_u16(pu2_pred + (row_out + 2) * pred_strd + 0);
                    uint16x8x2_t dst2  = vld2q_u16(pu2_dst  + (row_out + 2) * dst_strd  + 0);
                    int16x8_t acc2 = vminq_s16(vmaxq_s16(vqaddq_s16(r2, vreinterpretq_s16_u16(pred2.val[0])), clip_min), clip_max);
                    dst2.val[0] = vreinterpretq_u16_s16(acc2);
                    vst2q_u16(pu2_dst + (row_out + 2) * dst_strd + 0, dst2);

                    uint16x8x2_t pred6 = vld2q_u16(pu2_pred + (row_out + 2) * pred_strd + 16);
                    uint16x8x2_t dst6  = vld2q_u16(pu2_dst  + (row_out + 2) * dst_strd  + 16);
                    int16x8_t acc6 = vminq_s16(vmaxq_s16(vqaddq_s16(r6, vreinterpretq_s16_u16(pred6.val[0])), clip_min), clip_max);
                    dst6.val[0] = vreinterpretq_u16_s16(acc6);
                    vst2q_u16(pu2_dst + (row_out + 2) * dst_strd + 16, dst6);

                    /* Row 3 */
                    uint16x8x2_t pred3 = vld2q_u16(pu2_pred + (row_out + 3) * pred_strd + 0);
                    uint16x8x2_t dst3  = vld2q_u16(pu2_dst  + (row_out + 3) * dst_strd  + 0);
                    int16x8_t acc3 = vminq_s16(vmaxq_s16(vqaddq_s16(r3, vreinterpretq_s16_u16(pred3.val[0])), clip_min), clip_max);
                    dst3.val[0] = vreinterpretq_u16_s16(acc3);
                    vst2q_u16(pu2_dst + (row_out + 3) * dst_strd + 0, dst3);

                    uint16x8x2_t pred7 = vld2q_u16(pu2_pred + (row_out + 3) * pred_strd + 16);
                    uint16x8x2_t dst7  = vld2q_u16(pu2_dst  + (row_out + 3) * dst_strd  + 16);
                    int16x8_t acc7 = vminq_s16(vmaxq_s16(vqaddq_s16(r7, vreinterpretq_s16_u16(pred7.val[0])), clip_min), clip_max);
                    dst7.val[0] = vreinterpretq_u16_s16(acc7);
                    vst2q_u16(pu2_dst + (row_out + 3) * dst_strd + 16, dst7);
                }
            }
        }
    }
}
