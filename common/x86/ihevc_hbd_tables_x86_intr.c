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
*  ihevc_hbd_tables_x86_intr.c
*
* @brief
*  Contains table definitions for high bit depth x86 intrinsics.
*
* @author
*  Ittiam
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
#include "ihevc_platform_macros.h"
#include "ihevc_macros.h"
#include "ihevc_defs.h"
#include "ihevc_mem_fns.h"
#include "ihevc_hbd_tables_x86_intr.h"

// LUMA INTRA PRED
const UWORD8 IHEVCE_SHUFFLEMASKY1_HBD[16] = { 0x06, 0x07, 0x04, 0x05,
    0x02, 0x03, 0x00, 0x01,
    0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08};

const UWORD8 IHEVCE_SHUFFLEMASKY2_HBD[16] = { 0x0e, 0x0f, 0x0c, 0x0d,
    0x0a, 0x0b, 0x08, 0x09,
    0x06, 0x07, 0x04, 0x05,
    0x02, 0x03, 0x00, 0x01};

const UWORD8 IHEVCE_SHUFFLEMASKY3_HBD[16] = { 0x0e, 0x0f, 0x0c, 0x0d,
    0x0a, 0x0b, 0x08, 0x09,
    0x06, 0x07, 0x04, 0x05,
    0x02, 0x03, 0x00, 0x01};

const UWORD8 IHEVCE_SHUFFLEMASK4_HBD[16] = { 0x00, 0x01, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x01};

const UWORD8 IHEVCE_SHUFFLEMASK5_HBD[16] = { 0x00, 0x01, 0x08, 0x09,
    0x0f, 0x0f, 0x0f, 0x0f,
    0x0f, 0x0f, 0x0f, 0x0f,
    0x0f, 0x0f, 0x0f, 0x0f};

// CHROMA INTRA PRED
const UWORD8 IHEVCE_SHUFFLEMASKY7_HBD[16] = { 0x0c, 0x0d, 0x0e, 0x0f,
    0x08, 0x09, 0x0a, 0x0b,
    0x04, 0x05, 0x06, 0x07,
    0x00, 0x01, 0x02, 0x03};

const UWORD8 IHEVCE_SHUFFLEMASKY8_HBD[16] = { 0x0e, 0x0f, 0x0c, 0x0d,
    0x0a, 0x0b, 0x08, 0x09,
    0x06, 0x07, 0x04, 0x05,
    0x02, 0x03, 0x00, 0x01};

const UWORD8 IHEVCE_SHUFFLEMASKY9_HBD[16] = { 0x00, 0x01, 0x04, 0x05,
    0x08, 0x09, 0x0c, 0x0d,
    0x02, 0x03, 0x06, 0x07,
    0x0a, 0x0b, 0x0e, 0x0f};

// DEBLOCK TABLES
const WORD16 coef_hbd_d[8] = {0, 1, -2, 1, 1, -2, 1, 0};
const WORD16 coef_hbd_de1_1[8] = {9, -3, 9, -3, 9, -3, 9, -3};
const WORD16 coef_hbd_de1_2[8] = {3, -9, 3, -9, 3, -9, 3, -9};
const WORD16 coef_hbd_dep1_1[8] = {-2, 1, -2, 1, -2, 1, -2, 1};
const WORD16 coef_hbd_dep1_2[8] = {1, -2, 1, -2, 1, -2, 1, -2};
const WORD32 shuffle_hbd_d[4] = {0x09080706, 0x80808080, 0x0f0e0908, 0x07060100};
const WORD32 shuffle0_hbd[2] = {0x0d0c0302, 0x80808080};
const WORD32 shuffle1_hbd[4] = {0x05040100, 0x0d0c0908, 0x07060302, 0x0f0e0b0a};
const WORD32 shuffle2_hbd[4] = {0x80808080, 0x03020100, 0x07060504, 0x80808080};
const WORD32 shuffle3_hbd[4] = {0x80808080, 0x0b0a0908, 0x0f0e0d0c, 0x80808080};

const WORD16 delta0_hbd[8] = {1, -4, 1, -4, 1, -4, 1, -4};
const WORD16 delta1_hbd[8] = {4, -1, 4, -1, 4, -1, 4, -1};
const WORD32 shuffle_uv_hbd[4] = {0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a};
const WORD32 shuffle_uv_hbd1[4] = {0x80808080, 0x03020100, 0x07060504, 0x80808080};
const WORD32 shuffle_uv_hbd2[4] = {0x80808080, 0x0b0a0908, 0x0f0e0d0c, 0x80808080};

// SAO TABLES
const WORD8 gi1_table_edge_idx_hbd[8] = {1, 2, 0, 3, 4, 0, 0, 0};
const WORD8 gi1_table_band_idx_hbd[44] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 2, 3, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
};
const WORD32 gi4_ihevc_hbd_table_edge_idx[5] = {1, 2, 0, 3, 4};
