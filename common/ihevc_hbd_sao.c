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
*  ihevc_sao.c
*
* @brief
*  Contains leaf level function definitions for sample adaptive offset process
*
* @author
*  Ittiam
*
* @par List of Functions:
*
* @remarks
*  None
*
*******************************************************************************
*/
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_function_selector.h"
#include "ihevc_defs.h"
#include "ihevc_structs.h"
#include "ihevc_sao.h"
#include "ihevc_hbd_tables_x86_intr.h"

#define NUM_BAND_TABLE  32


/**
 * au4_avail is an array of flags - one for each neighboring block specifying if the block is available
 * au4_avail[0] - left
 * au4_avail[1] - right
 * au4_avail[2] - top
 * au4_avail[3] - bottom
 * au4_avail[4] - top-left
 * au4_avail[5] - top-right
 * au4_avail[6] - bottom-left
 * au4_avail[7] - bottom-right
 */
void ihevc_hbd_sao_band_offset_luma(UWORD16 *pu2_src,
                           WORD32 i4_src_strd,
                           UWORD16 *pu2_src_left,
                           UWORD16 *pu2_src_top,
                           UWORD16 *pu2_src_top_left,
                           WORD32 i4_sao_band_pos,
                           WORD8 *pi1_sao_offset,
                           WORD32 i4_wd,
                           WORD32 i4_ht,
                           UWORD32 u4_bit_depth)
{
    WORD32 band_shift;
    WORD32 band_table[NUM_BAND_TABLE];
    WORD32 i;
    WORD32 row, col;

    /* Updating left and top and top-left */
    for(row = 0; row < i4_ht; row++)
    {
        pu2_src_left[row] = pu2_src[row * i4_src_strd + (i4_wd - 1)];
    }
    pu2_src_top_left[0] = pu2_src_top[i4_wd - 1];
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    band_shift = u4_bit_depth - 5;
    for(i = 0; i < NUM_BAND_TABLE; i++)
    {
        band_table[i] = 0;
    }
    for(i = 0; i < 4; i++)
    {
        band_table[(i + i4_sao_band_pos) & 31] = i + 1;
    }

    for(row = 0; row < i4_ht; row++)
    {
        for(col = 0; col < i4_wd; col++)
        {
            WORD32 band_idx;

            band_idx = band_table[pu2_src[col] >> band_shift];
            pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[band_idx], 0, (1 << (band_shift + 5)) - 1);
        }
        pu2_src += i4_src_strd;
    }
}

void ihevc_hbd_sao_band_offset_chroma(UWORD16 *pu2_src,
                           WORD32 i4_src_strd,
                           UWORD16 *pu2_src_left,
                           UWORD16 *pu2_src_top,
                           UWORD16 *pu2_src_top_left,
                           WORD32 i4_sao_band_pos_u,
                           WORD32 i4_sao_band_pos_v,
                           WORD8 *pi1_sao_offset_u,
                           WORD8 *pi1_sao_offset_v,
                           WORD32 i4_wd,
                           WORD32 i4_ht,
                           UWORD32 u4_bit_depth)
{
    WORD32 band_shift;
    WORD32 band_table_u[NUM_BAND_TABLE];
    WORD32 band_table_v[NUM_BAND_TABLE];
    WORD32 i;
    WORD32 row, col;

    /* Updating left and top and top-left */
    for(row = 0; row < i4_ht; row++)
    {
        pu2_src_left[2 * row] = pu2_src[row * i4_src_strd + (i4_wd - 2)];
        pu2_src_left[2 * row + 1] = pu2_src[row * i4_src_strd + (i4_wd - 1)];
    }
    pu2_src_top_left[0] = pu2_src_top[i4_wd - 2];
    pu2_src_top_left[1] = pu2_src_top[i4_wd - 1];
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    band_shift = u4_bit_depth - 5;
    for(i = 0; i < NUM_BAND_TABLE; i++)
    {
        band_table_u[i] = 0;
        band_table_v[i] = 0;
    }
    for(i = 0; i < 4; i++)
    {
        band_table_u[(i + i4_sao_band_pos_u) & 31] = i + 1;
        band_table_v[(i + i4_sao_band_pos_v) & 31] = i + 1;
    }

    for(row = 0; row < i4_ht; row++)
    {
        for(col = 0; col < i4_wd; col++)
        {
            WORD32 band_idx;
            WORD8 *pi1_sao_offset;

            pi1_sao_offset = (0 == col % 2) ? pi1_sao_offset_u : pi1_sao_offset_v;
            band_idx = (0 == col % 2) ? band_table_u[pu2_src[col] >> band_shift] : band_table_v[pu2_src[col] >> band_shift];
            pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[band_idx], 0, (1 << (band_shift + 5)) - 1);
        }
        pu2_src += i4_src_strd;
    }
}

void ihevc_hbd_sao_edge_offset_class0(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE];
    WORD8 u1_sign_left, u1_sign_right;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update top and top-left arrays */
    *pu2_src_top_left = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        au2_src_left_tmp[row] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* Update masks based on the availability flags */
    if(0 == pu1_avail[0])
    {
        au1_mask[0] = 0;
    }
    if(0 == pu1_avail[1])
    {
        au1_mask[i4_wd - 1] = 0;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            u1_sign_left = SIGN(pu2_src[0] - pu2_src_left[row]);
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;

                u1_sign_right = SIGN(pu2_src[col] - pu2_src[col + 1]);
                edge_idx = 2 + u1_sign_left + u1_sign_right;
                u1_sign_left = -u1_sign_right;

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            pu2_src += i4_src_strd;
        }
    }

    /* Update left array */
    for(row = 0; row < i4_ht; row++)
    {
        pu2_src_left[row] = au2_src_left_tmp[row];
    }
}

void ihevc_hbd_sao_edge_offset_class0_chroma(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset_u,
                              WORD8 *pi1_sao_offset_v,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_left_tmp[2 * MAX_CTB_SIZE];
    WORD8 u1_sign_left_u, u1_sign_right_u;
    WORD8 u1_sign_left_v, u1_sign_right_v;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update left, top and top-left arrays */
    pu2_src_top_left[0] = pu2_src_top[i4_wd - 2];
    pu2_src_top_left[1] = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        au2_src_left_tmp[2 * row] = pu2_src[row * i4_src_strd + i4_wd - 2];
        au2_src_left_tmp[2 * row + 1] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* Update masks based on the availability flags */
    if(0 == pu1_avail[0])
    {
        au1_mask[0] = 0;
    }
    if(0 == pu1_avail[1])
    {
        au1_mask[(i4_wd - 1) >> 1] = 0;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            u1_sign_left_u = SIGN(pu2_src[0] - pu2_src_left[2 * row]);
            u1_sign_left_v = SIGN(pu2_src[1] - pu2_src_left[2 * row + 1]);
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;
                WORD8 *pi1_sao_offset;

                if(0 == col % 2)
                {
                    pi1_sao_offset = pi1_sao_offset_u;
                    u1_sign_right_u = SIGN(pu2_src[col] - pu2_src[col + 2]);
                    edge_idx = 2 + u1_sign_left_u + u1_sign_right_u;
                    u1_sign_left_u = -u1_sign_right_u;
                }
                else
                {
                    pi1_sao_offset = pi1_sao_offset_v;
                    u1_sign_right_v = SIGN(pu2_src[col] - pu2_src[col + 2]);
                    edge_idx = 2 + u1_sign_left_v + u1_sign_right_v;
                    u1_sign_left_v = -u1_sign_right_v;
                }

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col >> 1];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            pu2_src += i4_src_strd;
        }
    }

    for(row = 0; row < 2 * i4_ht; row++)
    {
        pu2_src_left[row] = au2_src_left_tmp[row];
    }
}

void ihevc_hbd_sao_edge_offset_class1(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_top_tmp[MAX_CTB_SIZE];
    WORD8 au1_sign_up[MAX_CTB_SIZE];
    WORD8 u1_sign_down;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update left, top and top-left arrays */
    *pu2_src_top_left = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        pu2_src_left[row] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        au2_src_top_tmp[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_src += i4_src_strd;
        i4_ht--;
        for(col = 0; col < i4_wd; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src[col - i4_src_strd]);
        }
    }
    else
    {
        for(col = 0; col < i4_wd; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src_top[col]);
        }
    }
    if(0 == pu1_avail[3])
    {
        i4_ht--;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;

                u1_sign_down = SIGN(pu2_src[col] - pu2_src[col + i4_src_strd]);
                edge_idx = 2 + au1_sign_up[col] + u1_sign_down;
                au1_sign_up[col] = -u1_sign_down;

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            pu2_src += i4_src_strd;
        }
    }

    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = au2_src_top_tmp[col];
    }
}

void ihevc_hbd_sao_edge_offset_class1_chroma(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset_u,
                              WORD8 *pi1_sao_offset_v,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_top_tmp[MAX_CTB_SIZE];
    WORD8 au1_sign_up[MAX_CTB_SIZE];
    WORD8 u1_sign_down;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update left, top and top-left arrays */
    pu2_src_top_left[0] = pu2_src_top[i4_wd - 2];
    pu2_src_top_left[1] = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        pu2_src_left[2 * row] = pu2_src[row * i4_src_strd + i4_wd - 2];
        pu2_src_left[2 * row + 1] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        au2_src_top_tmp[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_src += i4_src_strd;
        i4_ht--;
        for(col = 0; col < i4_wd; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src[col - i4_src_strd]);
        }
    }
    else
    {
        for(col = 0; col < i4_wd; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src_top[col]);
        }
    }
    if(0 == pu1_avail[3])
    {
        i4_ht--;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;
                WORD8 *pi1_sao_offset;

                pi1_sao_offset = (0 == col % 2) ? pi1_sao_offset_u : pi1_sao_offset_v;

                u1_sign_down = SIGN(pu2_src[col] - pu2_src[col + i4_src_strd]);
                edge_idx = 2 + au1_sign_up[col] + u1_sign_down;
                au1_sign_up[col] = -u1_sign_down;

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col >> 1];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            pu2_src += i4_src_strd;
        }
    }

    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = au2_src_top_tmp[col];
    }
}

void ihevc_hbd_sao_edge_offset_class2(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE], au2_src_top_tmp[MAX_CTB_SIZE];
    UWORD16 u2_src_top_left_tmp;
    WORD8 au1_sign_up[MAX_CTB_SIZE + 1], au1_sign_up_tmp[MAX_CTB_SIZE + 1];
    WORD8 u1_sign_down;
    WORD8 *pu1_sign_up;
    WORD8 *pu1_sign_up_tmp;
    UWORD16 *pu2_src_left_cpy;

    UWORD16 u2_pos_0_0_tmp;
    UWORD16 u2_pos_wd_ht_tmp;

    pu1_sign_up = au1_sign_up;
    pu1_sign_up_tmp = au1_sign_up_tmp;
    pu2_src_left_cpy = pu2_src_left;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update left, top and top-left arrays */
    u2_src_top_left_tmp = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        au2_src_left_tmp[row] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        au2_src_top_tmp[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* If top-left is available, process separately */
    if(0 != pu1_avail[4])
    {
        WORD32 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[0] - pu2_src_top_left[0]) +
                        SIGN(pu2_src[0] - pu2_src[1 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_0_tmp = CLIP3(pu2_src[0] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_0_0_tmp = pu2_src[0];
        }
    }
    else
    {
        u2_pos_0_0_tmp = pu2_src[0];
    }

    /* If bottom-right is available, process separately */
    if(0 != pu1_avail[7])
    {
        WORD32 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd] - pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd- 1 - i4_src_strd]) +
                        SIGN(pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd] - pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd + 1 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_ht_tmp = CLIP3(pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_ht_tmp = pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd];
        }
    }
    else
    {
        u2_pos_wd_ht_tmp = pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd];
    }

    /* If Left is not available */
    if(0 == pu1_avail[0])
    {
        au1_mask[0] = 0;
    }

    /* If Top is not available */
    if(0 == pu1_avail[2])
    {
        pu2_src += i4_src_strd;
        i4_ht--;
        pu2_src_left_cpy += 1;
        for(col = 1;col < i4_wd; col++)
        {
            pu1_sign_up[col] = SIGN(pu2_src[col] - pu2_src[col - 1 - i4_src_strd]);
        }
    }
    else
    {
        for(col = 1;col < i4_wd; col++)
        {
            pu1_sign_up[col] = SIGN(pu2_src[col] - pu2_src_top[col - 1]);
        }
    }

    /* If Right is not available */
    if(0 == pu1_avail[1])
    {
        au1_mask[i4_wd - 1] = 0;
    }

    /* If Bottom is not available */
    if(0 == pu1_avail[3])
    {
        i4_ht--;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            pu1_sign_up[0] = SIGN(pu2_src[0] - pu2_src_left_cpy[row - 1]);
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;

                u1_sign_down = SIGN(pu2_src[col] - pu2_src[col + 1 + i4_src_strd]);
                edge_idx = 2 + pu1_sign_up[col] + u1_sign_down;
                pu1_sign_up_tmp[col + 1] = -u1_sign_down;

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            /* Swapping pu1_sign_up_tmp and pu1_sign_up */
            {
                WORD8 *pu1_swap_tmp = pu1_sign_up;
                pu1_sign_up = pu1_sign_up_tmp;
                pu1_sign_up_tmp = pu1_swap_tmp;
            }

            pu2_src += i4_src_strd;
        }

        pu2_src[- (pu1_avail[2] ? i4_ht : i4_ht + 1) * i4_src_strd] = u2_pos_0_0_tmp;
        pu2_src[(pu1_avail[3] ? i4_wd - 1 - i4_src_strd : i4_wd - 1)] = u2_pos_wd_ht_tmp;
    }

    if(0 == pu1_avail[2])
        i4_ht++;
    if(0 == pu1_avail[3])
        i4_ht++;
    *pu2_src_top_left = u2_src_top_left_tmp;
    for(row = 0; row < i4_ht; row++)
    {
        pu2_src_left[row] = au2_src_left_tmp[row];
    }
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = au2_src_top_tmp[col];
    }
}

void ihevc_hbd_sao_edge_offset_class2_chroma(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset_u,
                              WORD8 *pi1_sao_offset_v,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_left_tmp[2 * MAX_CTB_SIZE], au2_src_top_tmp[MAX_CTB_SIZE];
    UWORD16 au2_src_top_left_tmp[2];
    WORD8 au1_sign_up[MAX_CTB_SIZE + 2], au1_sign_up_tmp[MAX_CTB_SIZE + 2];
    WORD8 u1_sign_down;
    WORD8 *pu1_sign_up;
    WORD8 *pu1_sign_up_tmp;
    UWORD16 *pu2_src_left_cpy;

    UWORD16 u2_pos_0_0_tmp_u;
    UWORD16 u2_pos_0_0_tmp_v;
    UWORD16 u2_pos_wd_ht_tmp_u;
    UWORD16 u2_pos_wd_ht_tmp_v;

    pu1_sign_up = au1_sign_up;
    pu1_sign_up_tmp = au1_sign_up_tmp;
    pu2_src_left_cpy = pu2_src_left;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update left, top and top-left arrays */
    au2_src_top_left_tmp[0] = pu2_src_top[i4_wd - 2];
    au2_src_top_left_tmp[1] = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        au2_src_left_tmp[2 * row] = pu2_src[row * i4_src_strd + i4_wd - 2];
        au2_src_left_tmp[2 * row + 1] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        au2_src_top_tmp[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* If top-left is available, process separately */
    if(0 != pu1_avail[4])
    {
        WORD32 edge_idx;

        /* U */
        edge_idx = 2 + SIGN(pu2_src[0] - pu2_src_top_left[0]) +
                        SIGN(pu2_src[0] - pu2_src[2 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_0_tmp_u = CLIP3(pu2_src[0] + pi1_sao_offset_u[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_0_0_tmp_u = pu2_src[0];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[1] - pu2_src_top_left[1]) +
                        SIGN(pu2_src[1] - pu2_src[1 + 2 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_0_tmp_v = CLIP3(pu2_src[1] + pi1_sao_offset_v[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_0_0_tmp_v = pu2_src[1];
        }
    }
    else
    {
        u2_pos_0_0_tmp_u = pu2_src[0];
        u2_pos_0_0_tmp_v = pu2_src[1];
    }

    /* If bottom-right is available, process separately */
    if(0 != pu1_avail[7])
    {
        WORD32 edge_idx;

        /* U */
        edge_idx = 2 + SIGN(pu2_src[i4_wd - 2 + (i4_ht - 1) * i4_src_strd] - pu2_src[i4_wd - 2 + (i4_ht - 1) * i4_src_strd - 2 - i4_src_strd]) +
                        SIGN(pu2_src[i4_wd - 2 + (i4_ht - 1) * i4_src_strd] - pu2_src[i4_wd - 2 + (i4_ht - 1) * i4_src_strd + 2 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_ht_tmp_u = CLIP3(pu2_src[i4_wd - 2 + (i4_ht - 1) * i4_src_strd] + pi1_sao_offset_u[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_ht_tmp_u = pu2_src[i4_wd - 2 + (i4_ht - 1) * i4_src_strd];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd] - pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd - 2 - i4_src_strd]) +
                        SIGN(pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd] - pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd + 2 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_ht_tmp_v = CLIP3(pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd] + pi1_sao_offset_v[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_ht_tmp_v = pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd];
        }
    }
    else
    {
        u2_pos_wd_ht_tmp_u = pu2_src[i4_wd - 2 + (i4_ht - 1) * i4_src_strd];
        u2_pos_wd_ht_tmp_v = pu2_src[i4_wd - 1 + (i4_ht - 1) * i4_src_strd];
    }

    /* If Left is not available */
    if(0 == pu1_avail[0])
    {
        au1_mask[0] = 0;
    }

    /* If Top is not available */
    if(0 == pu1_avail[2])
    {
        pu2_src += i4_src_strd;
        pu2_src_left_cpy += 2;
        i4_ht--;
        for(col = 2; col < i4_wd; col++)
        {
            pu1_sign_up[col] = SIGN(pu2_src[col] - pu2_src[col - 2 - i4_src_strd]);
        }
    }
    else
    {
        for(col = 2; col < i4_wd; col++)
        {
            pu1_sign_up[col] = SIGN(pu2_src[col] - pu2_src_top[col - 2]);
        }
    }

    /* If Right is not available */
    if(0 == pu1_avail[1])
    {
        au1_mask[(i4_wd - 1) >> 1] = 0;
    }

    /* If Bottom is not available */
    if(0 == pu1_avail[3])
    {
        i4_ht--;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            pu1_sign_up[0] = SIGN(pu2_src[0] - pu2_src_left_cpy[2 * (row - 1)]);
            pu1_sign_up[1] = SIGN(pu2_src[1] - pu2_src_left_cpy[2 * (row - 1) + 1]);
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;
                WORD8 * pi1_sao_offset;

                pi1_sao_offset = (0 == col % 2) ? pi1_sao_offset_u : pi1_sao_offset_v;

                u1_sign_down = SIGN(pu2_src[col] - pu2_src[col + 2 + i4_src_strd]);
                edge_idx = 2 + pu1_sign_up[col] + u1_sign_down;
                pu1_sign_up_tmp[col + 2] = -u1_sign_down;

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col >> 1];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            /* Swapping pu1_sign_up_tmp and pu1_sign_up */
            {
                WORD8 *pu1_swap_tmp = pu1_sign_up;
                pu1_sign_up = pu1_sign_up_tmp;
                pu1_sign_up_tmp = pu1_swap_tmp;
            }

            pu2_src += i4_src_strd;
        }

        pu2_src[- (pu1_avail[2] ? i4_ht : i4_ht + 1) * i4_src_strd] = u2_pos_0_0_tmp_u;
        pu2_src[- (pu1_avail[2] ? i4_ht : i4_ht + 1) * i4_src_strd + 1] = u2_pos_0_0_tmp_v;
        pu2_src[(pu1_avail[3] ? i4_wd - 2 - i4_src_strd : i4_wd - 2)] = u2_pos_wd_ht_tmp_u;
        pu2_src[(pu1_avail[3] ? i4_wd - 1 - i4_src_strd : i4_wd - 1)] = u2_pos_wd_ht_tmp_v;
    }

    if(0 == pu1_avail[2])
        i4_ht++;
    if(0 == pu1_avail[3])
        i4_ht++;
    pu2_src_top_left[0] = au2_src_top_left_tmp[0];
    pu2_src_top_left[1] = au2_src_top_left_tmp[1];
    for(row = 0; row < 2 * i4_ht; row++)
    {
        pu2_src_left[row] = au2_src_left_tmp[row];
    }
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = au2_src_top_tmp[col];
    }
}

void ihevc_hbd_sao_edge_offset_class3(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_top_tmp[MAX_CTB_SIZE];
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE];
    UWORD16 u2_src_top_left_tmp;
    WORD8 au1_sign_up[MAX_CTB_SIZE];
    UWORD16 *pu2_src_left_cpy;
    WORD8 u1_sign_down;

    UWORD16 u2_pos_0_ht_tmp;
    UWORD16 u2_pos_wd_0_tmp;

    pu2_src_left_cpy = pu2_src_left;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update left, top and top-left arrays */
    u2_src_top_left_tmp = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        au2_src_left_tmp[row] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        au2_src_top_tmp[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* If top-right is available, process separately */
    if(0 != pu1_avail[5])
    {
        WORD32 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[i4_wd - 1] - pu2_src_top_right[0]) +
                        SIGN(pu2_src[i4_wd - 1] - pu2_src[i4_wd - 1 - 1 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_0_tmp = CLIP3(pu2_src[i4_wd - 1] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_0_tmp = pu2_src[i4_wd - 1];
        }
    }
    else
    {
        u2_pos_wd_0_tmp = pu2_src[i4_wd - 1];
    }

    /* If bottom-left is available, process separately */
    if(0 != pu1_avail[6])
    {
        WORD32 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[(i4_ht - 1) * i4_src_strd] - pu2_src[(i4_ht - 1) * i4_src_strd + 1 - i4_src_strd]) +
                        SIGN(pu2_src[(i4_ht - 1) * i4_src_strd] - pu2_src_bot_left[0]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_ht_tmp = CLIP3(pu2_src[(i4_ht - 1) * i4_src_strd] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_0_ht_tmp = pu2_src[(i4_ht - 1) * i4_src_strd];
        }
    }
    else
    {
        u2_pos_0_ht_tmp = pu2_src[(i4_ht - 1) * i4_src_strd];
    }

    /* If Left is not available */
    if(0 == pu1_avail[0])
    {
        au1_mask[0] = 0;
    }

    /* If Top is not available */
    if(0 == pu1_avail[2])
    {
        pu2_src += i4_src_strd;
        i4_ht--;
        pu2_src_left_cpy += 1;
        for(col = 0; col < i4_wd - 1; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src[col + 1 - i4_src_strd]);
        }
    }
    else
    {
        for(col = 0; col < i4_wd - 1; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src_top[col + 1]);
        }
    }

    /* If Right is not available */
    if(0 == pu1_avail[1])
    {
        au1_mask[i4_wd - 1] = 0;
    }

    /* If Bottom is not available */
    if(0 == pu1_avail[3])
    {
        i4_ht--;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            au1_sign_up[i4_wd - 1] = SIGN(pu2_src[i4_wd - 1] - pu2_src[i4_wd - 1 + 1 - i4_src_strd]);
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;

                u1_sign_down = SIGN(pu2_src[col] - ((col == 0) ? pu2_src_left_cpy[row + 1] :
                                                            pu2_src[col - 1 + i4_src_strd]));
                edge_idx = 2 + au1_sign_up[col] + u1_sign_down;
                if(col > 0)
                    au1_sign_up[col - 1] = -u1_sign_down;

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            pu2_src += i4_src_strd;
        }

        pu2_src[- (pu1_avail[2] ? i4_ht : i4_ht + 1) * i4_src_strd + i4_wd - 1] = u2_pos_wd_0_tmp;
        pu2_src[(pu1_avail[3] ?  (-i4_src_strd) : 0)] = u2_pos_0_ht_tmp;
    }

    if(0 == pu1_avail[2])
        i4_ht++;
    if(0 == pu1_avail[3])
        i4_ht++;
    *pu2_src_top_left = u2_src_top_left_tmp;
    for(row = 0; row < i4_ht; row++)
    {
        pu2_src_left[row] = au2_src_left_tmp[row];
    }
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = au2_src_top_tmp[col];
    }
}


void ihevc_hbd_sao_edge_offset_class3_chroma(UWORD16 *pu2_src,
                              WORD32 i4_src_strd,
                              UWORD16 *pu2_src_left,
                              UWORD16 *pu2_src_top,
                              UWORD16 *pu2_src_top_left,
                              UWORD16 *pu2_src_top_right,
                              UWORD16 *pu2_src_bot_left,
                              UWORD8 *pu1_avail,
                              WORD8 *pi1_sao_offset_u,
                              WORD8 *pi1_sao_offset_v,
                              WORD32 i4_wd,
                              WORD32 i4_ht,
                              UWORD32 u4_bit_depth)
{
    WORD32 row, col;
    UWORD8 au1_mask[MAX_CTB_SIZE];
    UWORD16 au2_src_left_tmp[2 * MAX_CTB_SIZE], au2_src_top_tmp[MAX_CTB_SIZE];
    UWORD16 au2_src_top_left_tmp[2];
    WORD8 au1_sign_up[MAX_CTB_SIZE];
    UWORD16 *pu2_src_left_cpy;
    WORD8 u1_sign_down;

    UWORD16 u2_pos_wd_0_tmp_u;
    UWORD16 u2_pos_wd_0_tmp_v;
    UWORD16 u2_pos_0_ht_tmp_u;
    UWORD16 u2_pos_0_ht_tmp_v;

    pu2_src_left_cpy = pu2_src_left;

    /* Initialize the mask values */
    memset(au1_mask, 0xFF, MAX_CTB_SIZE);

    /* Update left, top and top-left arrays */
    au2_src_top_left_tmp[0] = pu2_src_top[i4_wd - 2];
    au2_src_top_left_tmp[1] = pu2_src_top[i4_wd - 1];
    for(row = 0; row < i4_ht; row++)
    {
        au2_src_left_tmp[2 * row] = pu2_src[row * i4_src_strd + i4_wd - 2];
        au2_src_left_tmp[2 * row + 1] = pu2_src[row * i4_src_strd + i4_wd - 1];
    }
    for(col = 0; col < i4_wd; col++)
    {
        au2_src_top_tmp[col] = pu2_src[(i4_ht - 1) * i4_src_strd + col];
    }

    /* If top-right is available, process separately */
    if(0 != pu1_avail[5])
    {
        WORD32 edge_idx;

        /* U */
        edge_idx = 2 + SIGN(pu2_src[i4_wd - 2] - pu2_src_top_right[0]) +
                        SIGN(pu2_src[i4_wd - 2] - pu2_src[i4_wd - 2 - 2 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_0_tmp_u = CLIP3(pu2_src[i4_wd - 2] + pi1_sao_offset_u[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_0_tmp_u = pu2_src[i4_wd - 2];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[i4_wd - 1] - pu2_src_top_right[1]) +
                        SIGN(pu2_src[i4_wd - 1] - pu2_src[i4_wd - 1 - 2 + i4_src_strd]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_0_tmp_v = CLIP3(pu2_src[i4_wd - 1] + pi1_sao_offset_v[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_0_tmp_v = pu2_src[i4_wd - 1];
        }
    }
    else
    {
        u2_pos_wd_0_tmp_u = pu2_src[i4_wd - 2];
        u2_pos_wd_0_tmp_v = pu2_src[i4_wd - 1];
    }

    /* If bottom-left is available, process separately */
    if(0 != pu1_avail[6])
    {
        WORD32 edge_idx;

        /* U */
        edge_idx = 2 + SIGN(pu2_src[(i4_ht - 1) * i4_src_strd] - pu2_src[(i4_ht - 1) * i4_src_strd + 2 - i4_src_strd]) +
                        SIGN(pu2_src[(i4_ht - 1) * i4_src_strd] - pu2_src_bot_left[0]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_ht_tmp_u = CLIP3(pu2_src[(i4_ht - 1) * i4_src_strd] + pi1_sao_offset_u[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_0_ht_tmp_u = pu2_src[(i4_ht - 1) * i4_src_strd];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[(i4_ht - 1) * i4_src_strd + 1] - pu2_src[(i4_ht - 1) * i4_src_strd + 1 + 2 - i4_src_strd]) +
                        SIGN(pu2_src[(i4_ht - 1) * i4_src_strd + 1] - pu2_src_bot_left[1]);

        edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_ht_tmp_v = CLIP3(pu2_src[(i4_ht - 1) * i4_src_strd + 1] + pi1_sao_offset_v[edge_idx], 0, (1 << u4_bit_depth) - 1);
        }
        else
        {
            u2_pos_0_ht_tmp_v = pu2_src[(i4_ht - 1) * i4_src_strd + 1];
        }
    }
    else
    {
        u2_pos_0_ht_tmp_u = pu2_src[(i4_ht - 1) * i4_src_strd];
        u2_pos_0_ht_tmp_v = pu2_src[(i4_ht - 1) * i4_src_strd + 1];
    }

    /* If Left is not available */
    if(0 == pu1_avail[0])
    {
        au1_mask[0] = 0;
    }

    /* If Top is not available */
    if(0 == pu1_avail[2])
    {
        pu2_src += i4_src_strd;
        i4_ht--;
        pu2_src_left_cpy += 2;
        for(col = 0; col < i4_wd - 2; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src[col + 2 - i4_src_strd]);
        }
    }
    else
    {
        for(col = 0; col < i4_wd - 2; col++)
        {
            au1_sign_up[col] = SIGN(pu2_src[col] - pu2_src_top[col + 2]);
        }
    }

    /* If Right is not available */
    if(0 == pu1_avail[1])
    {
        au1_mask[(i4_wd - 1) >> 1] = 0;
    }

    /* If Bottom is not available */
    if(0 == pu1_avail[3])
    {
        i4_ht--;
    }

    /* Processing is done on the intermediate buffer and the output is written to the source buffer */
    {
        for(row = 0; row < i4_ht; row++)
        {
            au1_sign_up[i4_wd - 2] = SIGN(pu2_src[i4_wd - 2] - pu2_src[i4_wd - 2 + 2 - i4_src_strd]);
            au1_sign_up[i4_wd - 1] = SIGN(pu2_src[i4_wd - 1] - pu2_src[i4_wd - 1 + 2 - i4_src_strd]);
            for(col = 0; col < i4_wd; col++)
            {
                WORD32 edge_idx;
                WORD8 * pi1_sao_offset;

                pi1_sao_offset = (0 == col % 2) ? pi1_sao_offset_u : pi1_sao_offset_v;

                u1_sign_down = SIGN(pu2_src[col] - ((col < 2) ? pu2_src_left_cpy[2 * (row + 1) + col] :
                                                                    pu2_src[col - 2 + i4_src_strd]));
                edge_idx = 2 + au1_sign_up[col] + u1_sign_down;
                if(col > 1)
                    au1_sign_up[col - 2] = -u1_sign_down;

                edge_idx = gi4_ihevc_hbd_table_edge_idx[edge_idx] & au1_mask[col >> 1];

                if(0 != edge_idx)
                {
                    pu2_src[col] = CLIP3(pu2_src[col] + pi1_sao_offset[edge_idx], 0, (1 << u4_bit_depth) - 1);
                }
            }

            pu2_src += i4_src_strd;
        }

        pu2_src[- (pu1_avail[2] ? i4_ht : i4_ht + 1) * i4_src_strd + i4_wd - 2] = u2_pos_wd_0_tmp_u;
        pu2_src[- (pu1_avail[2] ? i4_ht : i4_ht + 1) * i4_src_strd + i4_wd - 1] = u2_pos_wd_0_tmp_v;
        pu2_src[(pu1_avail[3] ?  (-i4_src_strd) : 0)] = u2_pos_0_ht_tmp_u;
        pu2_src[(pu1_avail[3] ?  (-i4_src_strd) : 0) + 1] = u2_pos_0_ht_tmp_v;
    }

    if(0 == pu1_avail[2])
        i4_ht++;
    if(0 == pu1_avail[3])
        i4_ht++;
    pu2_src_top_left[0] = au2_src_top_left_tmp[0];
    pu2_src_top_left[1] = au2_src_top_left_tmp[1];
    for(row = 0; row < 2 * i4_ht; row++)
    {
        pu2_src_left[row] = au2_src_left_tmp[row];
    }
    for(col = 0; col < i4_wd; col++)
    {
        pu2_src_top[col] = au2_src_top_tmp[col];
    }
}
