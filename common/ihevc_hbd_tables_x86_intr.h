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
*  ihevc_hbd_tables_x86_intr.h
*
* @brief
*  Declarations for the fucntions defined in  ihevc_intra_pred_filters
*
* @author
*  Mamatha
*
*
* @remarks
*  None
*
*******************************************************************************
*/

#ifndef IHEVC_HBD_TABLES_X86_INTR_H_
#define IHEVC_HBD_TABLES_X86_INTR_H_


//Luma intra pred
extern const UWORD8 IHEVCE_SHUFFLEMASKY1_HBD[16];
extern const UWORD8 IHEVCE_SHUFFLEMASKY2_HBD[16] ;
extern const UWORD8 IHEVCE_SHUFFLEMASKY3_HBD[16] ;
extern const UWORD8 IHEVCE_SHUFFLEMASK4_HBD[16] ;
extern const UWORD8 IHEVCE_SHUFFLEMASK5_HBD[16] ;
//Chroma intra pred
extern const UWORD8 IHEVCE_SHUFFLEMASKY7_HBD[16] ;

extern const UWORD8 IHEVCE_SHUFFLEMASKY8_HBD[16] ;

extern const UWORD8 IHEVCE_SHUFFLEMASKY9_HBD[16] ;

// DEBLOCK TABLES
extern const WORD16 coef_hbd_d[16] ;
extern const WORD16 coef_hbd_de1_1[16] ;
extern const WORD16 coef_hbd_de1_2[16] ;
extern const WORD16 coef_hbd_dep1_1[16] ;
extern const WORD16 coef_hbd_dep1_2[16] ;
extern const WORD32 shuffle_hbd_d[4] ;
extern const WORD32 shuffle0_hbd[2] ;
extern const WORD32 shuffle1_hbd[4] ;
extern const WORD32 shuffle2_hbd[4] ;
extern const WORD32 shuffle3_hbd[4] ;

extern const WORD16 delta0_hbd[8];
extern const WORD16 delta1_hbd[8];
extern const WORD32 shuffle_uv_hbd[4];
extern const WORD32 shuffle_uv_hbd1[4];
extern const WORD32 shuffle_uv_hbd2[4];
//SAO  TABLES
extern  const WORD8 gi1_table_edge_idx_hbd[5] ;
extern  const WORD8 gi1_table_band_idx_hbd[44];
extern  const WORD32 gi4_ihevc_hbd_table_edge_idx[5];

// TRANS TABLES
extern const WORD16 g_ai2_ihevc_trans_16_even_hbd[12][8];
extern const WORD16 g_ai2_ihevc_trans_16_odd_hbd[32][8];

#endif /*IHEVC_TABLES_X86_INTR_H_*/
