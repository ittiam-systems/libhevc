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
 *  ihevc_itrans_res.h
 *
 * @brief
 *  Functions declarations for inverse transform
 *
 * @author
 *  Ittiam
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
#ifndef _IHEVC_ITRANS_RES_H_
#define _IHEVC_ITRANS_RES_H_

typedef void ihevc_itrans_res_4x4_ttype1_ft(WORD16 *pi2_src,
                                            WORD16 *pi2_tmp,
                                            WORD16 *pi2_dst,
                                            WORD32 src_strd,
                                            WORD32 dst_strd,
                                            WORD32 zero_cols,
                                            WORD32 zero_rows);

typedef void ihevc_itrans_res_4x4_ft(WORD16 *pi2_src,
                                     WORD16 *pi2_tmp,
                                     WORD16 *pi2_dst,
                                     WORD32 src_strd,
                                     WORD32 dst_strd,
                                     WORD32 zero_cols,
                                     WORD32 zero_rows);

typedef void ihevcd_itrans_res_dc_ft(WORD16 *pi2_dst,
                                     WORD32 dst_strd,
                                     WORD32 log2_trans_size,
                                     WORD16 i2_coeff_value);

typedef void ihevc_itrans_res_8x8_ft(WORD16 *pi2_src,
                                     WORD16 *pi2_tmp,
                                     WORD16 *pi2_dst,
                                     WORD32 src_strd,
                                     WORD32 dst_strd,
                                     WORD32 zero_cols,
                                     WORD32 zero_rows);

typedef void ihevc_itrans_res_16x16_ft(WORD16 *pi2_src,
                                       WORD16 *pi2_tmp,
                                       WORD16 *pi2_dst,
                                       WORD32 src_strd,
                                       WORD32 dst_strd,
                                       WORD32 zero_cols,
                                       WORD32 zero_rows);

typedef void ihevc_itrans_res_32x32_ft(WORD16 *pi2_src,
                                       WORD16 *pi2_tmp,
                                       WORD16 *pi2_dst,
                                       WORD32 src_strd,
                                       WORD32 dst_strd,
                                       WORD32 zero_cols,
                                       WORD32 zero_rows);

/* C function declarations */
ihevc_itrans_res_4x4_ttype1_ft ihevc_itrans_res_4x4_ttype1;
ihevc_itrans_res_4x4_ft ihevc_itrans_res_4x4;
ihevcd_itrans_res_dc_ft ihevcd_itrans_res_dc;
ihevc_itrans_res_8x8_ft ihevc_itrans_res_8x8;
ihevc_itrans_res_16x16_ft ihevc_itrans_res_16x16;
ihevc_itrans_res_32x32_ft ihevc_itrans_res_32x32;

#endif /*_IHEVC_ITRANS_RES_H_*/
