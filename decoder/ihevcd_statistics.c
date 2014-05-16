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
*  ihevcd_statistics.c
*
* @brief
*  Contains macros for generating stats about hevc decoder
*
* @author
*  Naveen SR
*
* @par List of Functions:
*
* @remarks
*  None
*
*******************************************************************************
*/
/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <stdio.h>
#include <string.h>

#include "ihevc_typedefs.h"
#include "iv.h"
#include "ivd.h"
#include "ihevcd_cxa.h"
#include "ithread.h"

#include "ihevc_defs.h"
#include "ihevc_structs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_cabac_tables.h"
#include "ihevcd_defs.h"

#include "ihevcd_function_selector.h"
#include "ihevcd_structs.h"
#include "ihevcd_iquant_itrans_recon_ctb.h"
#include "ihevcd_statistics.h"

#if STATISTICS_ENABLE
statistics_t gs_ihevcd_stat;

void ihevcd_init_trans_stat(stat_trans_t *ps_stat_trans)
{
    ps_stat_trans->num_4x4_dst = 0;
    ps_stat_trans->num_4x4 = 0;
    ps_stat_trans->num_8x8 = 0;
    ps_stat_trans->num_16x16 = 0;
    ps_stat_trans->num_32x32 = 0;
    ps_stat_trans->num_64x64 = 0;
};

void ihevcd_sblk_pos_init()
{
    gs_ihevcd_stat.last_sblk_pos_x = 0;
    gs_ihevcd_stat.last_sblk_pos_y = 0;
    gs_ihevcd_stat.num_coded_sblk = 0;
    gs_ihevcd_stat.num_coded_coeffs = 0;
}
void ihevcd_init_sblk_histogram(stat_sblk_histogram_t *ps_last_sblk_pos_histogram_t)
{
    memset(ps_last_sblk_pos_histogram_t->trans_4x4_dst, 0, 1 * sizeof(UWORD32));
    memset(ps_last_sblk_pos_histogram_t->trans_4x4, 0, 1 * sizeof(UWORD32));
    memset(ps_last_sblk_pos_histogram_t->trans_8x8, 0, 4 * sizeof(UWORD32));
    memset(ps_last_sblk_pos_histogram_t->trans_16x16, 0, 16 * sizeof(UWORD32));
    memset(ps_last_sblk_pos_histogram_t->trans_32x32, 0, 64 * sizeof(UWORD32));
}
void ihevcd_init_coeff_histogram(stat_coeff_histogram_t *ps_coeff_histogram)
{
    memset(ps_coeff_histogram->trans_4x4_dst, 0, 16 * sizeof(UWORD32));
    memset(ps_coeff_histogram->trans_4x4, 0, 16 * sizeof(UWORD32));
    memset(ps_coeff_histogram->trans_8x8, 0, 64 * sizeof(UWORD32));
    memset(ps_coeff_histogram->trans_16x16, 0, 256 * sizeof(UWORD32));
    memset(ps_coeff_histogram->trans_32x32, 0, 1024 * sizeof(UWORD32));
}
void ihevcd_init_statistics()
{

    memset(&gs_ihevcd_stat, 0, sizeof(statistics_t));
    /* Number of transform block init */
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_all_trans_block[0]);
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_all_trans_block[1]);
    /* Number of coded transform block init */
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_coded_trans_block[0]);
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_coded_trans_block[1]);
    /* Number of coded DC transform block init */
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_coded_dc_block[0]);
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_coded_dc_block[1]);
    /* Number of coded one coeff transform block init */
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_coded_one_coeff_block[0]);
    ihevcd_init_trans_stat(&gs_ihevcd_stat.stat_num_coded_one_coeff_block[1]);
    /* Last sblk histogram init */
    ihevcd_init_sblk_histogram(&gs_ihevcd_stat.stat_last_sblk_pos_histogram);
    /* Num Coded sblk histogram init */
    ihevcd_init_sblk_histogram(&gs_ihevcd_stat.stat_num_coded_sblk_histogram);
    /* Num Coded coeffs histogram init */
    ihevcd_init_coeff_histogram(&gs_ihevcd_stat.stat_num_coded_coeff_histogram);
    /* Last sblk position init */
    ihevcd_sblk_pos_init();

}

void ihevcd_print_stat_trans(stat_trans_t *ps_stat_trans)
{
    WORD32 total_pixels_y, total_pixels_uv;
    double y_ratio, y_ratio_total, uv_ratio, uv_ratio_total;
    stat_trans_t *ps_stat_trans_all;
    total_pixels_y = ps_stat_trans[0].num_4x4_dst * 4 * 4 +
                    ps_stat_trans[0].num_4x4 * 4 * 4 +
                    ps_stat_trans[0].num_8x8 * 8 * 8 +
                    ps_stat_trans[0].num_16x16 * 16 * 16 +
                    ps_stat_trans[0].num_32x32 * 32 * 32 +
                    ps_stat_trans[0].num_64x64 * 64 * 64;

    total_pixels_uv = ps_stat_trans[1].num_4x4_dst * 4 * 4 +
                    ps_stat_trans[1].num_4x4  * 4 * 4 +
                    ps_stat_trans[1].num_8x8  * 8 * 8 +
                    ps_stat_trans[1].num_16x16 * 16 * 16 +
                    ps_stat_trans[1].num_32x32 * 32 * 32 +
                    ps_stat_trans[1].num_64x64 * 64 * 64;

    ps_stat_trans_all = &gs_ihevcd_stat.stat_num_all_trans_block[0];

    printf("\n_                   Y               Y            Y                U+V             U+V             U+V");
    printf("\nTransform_Type      Num_Blocks      Percentage   %%wrt_total      Num_Blocks     Percentage   %%wrt_total ");

    y_ratio = ps_stat_trans[0].num_4x4_dst * 4 * 4 * 100.0 / total_pixels_y;
    y_ratio_total = ps_stat_trans[0].num_4x4_dst * 100.0 / ps_stat_trans_all[0].num_4x4_dst;
    uv_ratio = ps_stat_trans[1].num_4x4_dst * 4 * 4 * 100.0 / total_pixels_uv;
    uv_ratio_total = ps_stat_trans[1].num_4x4_dst * 100.0 / ps_stat_trans_all[1].num_4x4_dst;
    printf("\nDST_4x4             %6d             %6.2f        %6.2f        %6d         %6.2f       %6.2f ", ps_stat_trans[0].num_4x4_dst, y_ratio, y_ratio_total, ps_stat_trans[1].num_4x4_dst, uv_ratio,  uv_ratio_total);

    y_ratio = ps_stat_trans[0].num_4x4 * 4 * 4 * 100.0 / total_pixels_y;
    y_ratio_total = ps_stat_trans[0].num_4x4 * 100.0 / ps_stat_trans_all[0].num_4x4;
    uv_ratio = ps_stat_trans[1].num_4x4 * 4 * 4 * 100.0 / total_pixels_uv;
    uv_ratio_total = ps_stat_trans[1].num_4x4 * 100.0 / ps_stat_trans_all[1].num_4x4;
    printf("\nDCT_4x4             %6d             %6.2f        %6.2f        %6d         %6.2f       %6.2f ", ps_stat_trans[0].num_4x4, y_ratio, y_ratio_total, ps_stat_trans[1].num_4x4, uv_ratio,  uv_ratio_total);


    y_ratio = ps_stat_trans[0].num_8x8 * 8 * 8 * 100.0 / total_pixels_y;
    y_ratio_total = ps_stat_trans[0].num_8x8 * 100.0 / ps_stat_trans_all[0].num_8x8;
    uv_ratio = ps_stat_trans[1].num_8x8 * 8 * 8 * 100.0 / total_pixels_uv;
    uv_ratio_total = ps_stat_trans[1].num_8x8 * 100.0 / ps_stat_trans_all[1].num_8x8;
    printf("\nDCT_8x8             %6d             %6.2f        %6.2f        %6d         %6.2f       %6.2f ", ps_stat_trans[0].num_8x8, y_ratio, y_ratio_total, ps_stat_trans[1].num_8x8, uv_ratio,  uv_ratio_total);

    y_ratio = ps_stat_trans[0].num_16x16 * 16 * 16 * 100.0 / total_pixels_y;
    y_ratio_total = ps_stat_trans[0].num_16x16 * 100.0 / ps_stat_trans_all[0].num_16x16;
    uv_ratio = ps_stat_trans[1].num_16x16 * 16 * 16 * 100.0 / total_pixels_uv;
    uv_ratio_total = ps_stat_trans[1].num_16x16 * 100.0 / ps_stat_trans_all[1].num_16x16;
    printf("\nDCT_16x16           %6d             %6.2f        %6.2f        %6d         %6.2f       %6.2f ", ps_stat_trans[0].num_16x16, y_ratio, y_ratio_total, ps_stat_trans[1].num_16x16, uv_ratio,  uv_ratio_total);


    y_ratio = ps_stat_trans[0].num_32x32 * 32 * 32 * 100.0 / total_pixels_y;
    y_ratio_total = ps_stat_trans[0].num_32x32 * 100.0 / ps_stat_trans_all[0].num_32x32;
    uv_ratio = ps_stat_trans[1].num_32x32 * 32 * 32 * 100.0 / total_pixels_uv;
    uv_ratio_total = ps_stat_trans[1].num_32x32 * 100.0 / ps_stat_trans_all[1].num_32x32;
    printf("\nDCT_32x32           %6d             %6.2f        %6.2f        %6d         %6.2f       %6.2f ", ps_stat_trans[0].num_32x32, y_ratio, y_ratio_total, ps_stat_trans[1].num_32x32, uv_ratio,  uv_ratio_total);


    y_ratio = ps_stat_trans[0].num_64x64 * 64 * 64 * 100.0 / total_pixels_y;
    y_ratio_total = ps_stat_trans[0].num_64x64 * 100.0 / ps_stat_trans_all[0].num_64x64;
    uv_ratio = ps_stat_trans[1].num_64x64 * 64 * 64 * 100.0 / total_pixels_uv;
    uv_ratio_total = ps_stat_trans[1].num_64x64 * 100.0 / ps_stat_trans_all[1].num_64x64;
    printf("\nDCT_64x64           %6d             %6.2f        %6.2f        %6d         %6.2f       %6.2f ", ps_stat_trans[0].num_64x64, y_ratio, y_ratio_total, ps_stat_trans[1].num_64x64, uv_ratio,  uv_ratio_total);

}

void ihevcd_update_stat_num_trans(stat_trans_t *ps_stat_trans, TRANSFORM_TYPE e_trans_type)
{
    switch(e_trans_type)
    {
        case DST_4x4:
            ps_stat_trans->num_4x4_dst++;
            break;
        case DCT_4x4:
            ps_stat_trans->num_4x4++;
            break;
        case DCT_8x8:
            ps_stat_trans->num_8x8++;
            break;
        case DCT_16x16:
            ps_stat_trans->num_16x16++;
            break;
        case DCT_32x32:
            ps_stat_trans->num_32x32++;
            break;
        case SKIP_64x64:
            ps_stat_trans->num_64x64++;
            break;
        default:
            break;
    }
}

void ihevcd_update_num_all_trans_blocks(TRANSFORM_TYPE e_trans_type, WORD32 c_idx)
{
    stat_trans_t *ps_stat_trans;

    ps_stat_trans = &gs_ihevcd_stat.stat_num_all_trans_block[0];

    if(c_idx != 0)
    {
        ps_stat_trans++;
    }
    ihevcd_update_stat_num_trans(ps_stat_trans, e_trans_type);
}

void ihevcd_update_num_trans_blocks(TRANSFORM_TYPE e_trans_type, WORD32 c_idx, WORD32 update_type)
{
    stat_trans_t *ps_stat_trans;

    if(0 == update_type)
        ps_stat_trans = &gs_ihevcd_stat.stat_num_coded_trans_block[0];
    else if(1 == update_type)
        ps_stat_trans = &gs_ihevcd_stat.stat_num_coded_dc_block[0];
    else
        ps_stat_trans = &gs_ihevcd_stat.stat_num_coded_one_coeff_block[0];

    if(c_idx != 0)
    {
        ps_stat_trans++;
    }
    ihevcd_update_stat_num_trans(ps_stat_trans, e_trans_type);
}

void ihevcd_print_sblk_histogram_per_transform(UWORD32 *pu4_stat, UWORD32 wd, UWORD32 ht, WORD32 is_2d)
{
    UWORD32 i, j, total = 0, val;

    for(i = 0; i < ht; i++)
    {
        for(j = 0; j < wd; j++)
        {
            val = pu4_stat[j + i * ht];
            printf("%d\t\t", val);
            total += val;
        }
        if(1 == is_2d)
            printf("\n");
    }

    {
        printf("\n");
        for(i = 0; i < ht; i++)
        {
            for(j = 0; j < wd; j++)
            {
                val = pu4_stat[j + i * ht];

                printf("%.2f%%\t\t", val * 100.0 / total);
            }
            if(1 == is_2d)
                printf("\n");
        }
    }
}

void ihevcd_print_sblk_histogram(stat_sblk_histogram_t *ps_stat_sblk_pos_histogram, WORD32 is_2d)
{
    printf("\nhistogram_4x4_DST\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_sblk_pos_histogram->trans_4x4_dst, 1, 1, is_2d);
    printf("\nhistogram_4x4\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_sblk_pos_histogram->trans_4x4, 1, 1, is_2d);
    printf("\nhistogram_8x8\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_sblk_pos_histogram->trans_8x8, 2, 2, is_2d);
    printf("\nhistogram_16x16\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_sblk_pos_histogram->trans_16x16, 4, 4, is_2d);
    printf("\nhistogram_32x32\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_sblk_pos_histogram->trans_32x32, 8, 8, is_2d);
}

void ihevcd_print_coeff_histogram(stat_coeff_histogram_t *ps_stat_coeff_histogram, WORD32 is_2d)
{
    printf("\nhistogram_4x4_DST\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_coeff_histogram->trans_4x4_dst, 4, 4, is_2d);
    printf("\nhistogram_4x4\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_coeff_histogram->trans_4x4, 4, 4, is_2d);
    printf("\nhistogram_8x8\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_coeff_histogram->trans_8x8, 8, 8, is_2d);
    printf("\nhistogram_16x16\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_coeff_histogram->trans_16x16, 16, 16, is_2d);
    printf("\nhistogram_32x32\n");
    ihevcd_print_sblk_histogram_per_transform(ps_stat_coeff_histogram->trans_32x32, 32, 32, is_2d);
}
void ihevcd_print_transform_statistics()
{
    stat_trans_t *ps_stat_trans;
    WORD32 total_blocks;

    /* Num coded_transform blocks */
    printf("\nNUM_ALL_TRANSFORM_BLOCKS\n");
    ps_stat_trans = &gs_ihevcd_stat.stat_num_all_trans_block[0];
    {
        /* Updating chroma blocks. As the chroma blocks are not counted if cbf of y,u and v are zero */
        ps_stat_trans[1].num_4x4 = (ps_stat_trans[0].num_4x4_dst + ps_stat_trans[0].num_4x4) / 4 + ps_stat_trans->num_8x8;
        ps_stat_trans[1].num_8x8 = ps_stat_trans->num_16x16;
        ps_stat_trans[1].num_16x16 = ps_stat_trans->num_32x32;
        ps_stat_trans[1].num_32x32 = ps_stat_trans->num_64x64;
    }
    ihevcd_print_stat_trans(ps_stat_trans);

    /* Num coded_transform blocks */
    printf("\nNUM_CODED_TRANSFORM_BLOCKS(excluding_trans_skip_and_trans_quant_bypass)\n");
    ps_stat_trans = &gs_ihevcd_stat.stat_num_coded_trans_block[0];
    ihevcd_print_stat_trans(ps_stat_trans);

    /* Num DC transform blocks */
    printf("\nNUM_DC_TRANSFORM_BLOCKS(excluding_trans_skip_and_trans_quant_bypass)\n");
    ps_stat_trans = &gs_ihevcd_stat.stat_num_coded_dc_block[0];
    ihevcd_print_stat_trans(ps_stat_trans);

    /* Num one coeff transform blocks */
    printf("\nNUM_ONE_COEFF_TRANSFORM_BLOCKS(excluding_trans_skip_and_trans_quant_bypass)\n");
    ps_stat_trans = &gs_ihevcd_stat.stat_num_coded_one_coeff_block[0];
    ihevcd_print_stat_trans(ps_stat_trans);

    /* Last sblk histogram */
    printf("\nLAST_CODED_SBLK_HISTOGRAM\n");
    ihevcd_print_sblk_histogram(&gs_ihevcd_stat.stat_last_sblk_pos_histogram, 1);

    /* Num Coded sblks histogram */
    printf("\nNUM_CODED_SBLK_HISTOGRAM\n");
    ihevcd_print_sblk_histogram(&gs_ihevcd_stat.stat_num_coded_sblk_histogram, 1);

    /* Num Coded coeff histogram */
    printf("\nNUM_CODED_COEFF_HISTOGRAM\n");
    ihevcd_print_coeff_histogram(&gs_ihevcd_stat.stat_num_coded_coeff_histogram, 1);
}

void ihevcd_update_sblk_histogram(stat_sblk_histogram_t *ps_last_sblk_pos_histogram, TRANSFORM_TYPE e_trans_type,  WORD32 last_sblk_x, WORD32 last_sblk_y)
{
    switch(e_trans_type)
    {
        case DST_4x4:
            ps_last_sblk_pos_histogram->trans_4x4_dst[last_sblk_x + last_sblk_y * 0]++;
            break;
        case DCT_4x4:
            ps_last_sblk_pos_histogram->trans_4x4[last_sblk_x + last_sblk_y * 0]++;
            break;
        case DCT_8x8:
            ps_last_sblk_pos_histogram->trans_8x8[last_sblk_x + last_sblk_y * 2]++;
            break;
        case DCT_16x16:
            ps_last_sblk_pos_histogram->trans_16x16[last_sblk_x + last_sblk_y * 4]++;
            break;
        case DCT_32x32:
            ps_last_sblk_pos_histogram->trans_32x32[last_sblk_x + last_sblk_y * 8]++;
            break;
        default:
            break;
    }
}

void ihevcd_update_num_coded_sblk_histogram(stat_sblk_histogram_t *ps_sblk_histogram, TRANSFORM_TYPE e_trans_type,  WORD32 num_coded_blks)
{
    switch(e_trans_type)
    {
        case DST_4x4:
            ps_sblk_histogram->trans_4x4_dst[num_coded_blks - 1]++;
            break;
        case DCT_4x4:
            ps_sblk_histogram->trans_4x4[num_coded_blks - 1]++;
            break;
        case DCT_8x8:
            ps_sblk_histogram->trans_8x8[num_coded_blks - 1]++;
            break;
        case DCT_16x16:
            ps_sblk_histogram->trans_16x16[num_coded_blks - 1]++;
            break;
        case DCT_32x32:
            ps_sblk_histogram->trans_32x32[num_coded_blks - 1]++;
            break;
        default:
            break;
    }
}

void ihevcd_update_num_coded_coeff_histogram(stat_coeff_histogram_t *ps_coeff_histogram, TRANSFORM_TYPE e_trans_type,  WORD32 num_coded_blks)
{
    switch(e_trans_type)
    {
        case DST_4x4:
            ps_coeff_histogram->trans_4x4_dst[num_coded_blks - 1]++;
            break;
        case DCT_4x4:
            ps_coeff_histogram->trans_4x4[num_coded_blks - 1]++;
            break;
        case DCT_8x8:
            ps_coeff_histogram->trans_8x8[num_coded_blks - 1]++;
            break;
        case DCT_16x16:
            ps_coeff_histogram->trans_16x16[num_coded_blks - 1]++;
            break;
        case DCT_32x32:
            ps_coeff_histogram->trans_32x32[num_coded_blks - 1]++;
            break;
        default:
            break;
    }
}

void ihevcd_sblk_pos_update(TRANSFORM_TYPE e_trans_type, WORD32 t_skip_or_tq_bypass, UWORD32 sblk_x, UWORD32 sblk_y)
{
    if(1 == t_skip_or_tq_bypass)
        return;

    gs_ihevcd_stat.num_coded_sblk++;

    /* Updating the last coded sblk pos */
#if 0
    if(gs_ihevcd_stat.last_sblk_pos_y > sblk_y)
        return;
    else if(gs_ihevcd_stat.last_sblk_pos_y == sblk_y)
    {
        if(gs_ihevcd_stat.last_sblk_pos_x >= sblk_x)
            return;
        else
            gs_ihevcd_stat.last_sblk_pos_x = sblk_x;
    }
    else
    {
        gs_ihevcd_stat.last_sblk_pos_y = sblk_y;
        gs_ihevcd_stat.last_sblk_pos_x = sblk_x;
    }
#endif
    if(gs_ihevcd_stat.last_sblk_pos_y < sblk_y)
        gs_ihevcd_stat.last_sblk_pos_y = sblk_y;

    if(gs_ihevcd_stat.last_sblk_pos_x < sblk_x)
        gs_ihevcd_stat.last_sblk_pos_x = sblk_x;
}

void ihevcd_update_coeff_count()
{
    gs_ihevcd_stat.num_coded_coeffs++;
}

void ihevcd_update_sblk_and_coeff_histogram(TRANSFORM_TYPE e_trans_type, WORD32 t_skip_or_tq_bypass)
{
    if(0 == t_skip_or_tq_bypass)
    {
        ihevcd_update_sblk_histogram(&gs_ihevcd_stat.stat_last_sblk_pos_histogram, e_trans_type, gs_ihevcd_stat.last_sblk_pos_x, gs_ihevcd_stat.last_sblk_pos_y);
        ihevcd_update_num_coded_sblk_histogram(&gs_ihevcd_stat.stat_num_coded_sblk_histogram, e_trans_type, gs_ihevcd_stat.num_coded_sblk);
        ihevcd_update_num_coded_coeff_histogram(&gs_ihevcd_stat.stat_num_coded_coeff_histogram, e_trans_type, gs_ihevcd_stat.num_coded_coeffs);
    }
}

void ihevcd_update_pu_skip_size(pu_t *ps_pu)
{
    WORD32 wd, ht;

    wd = (ps_pu->b4_wd);
    ht = (ps_pu->b4_ht);

    gs_ihevcd_stat.stat_pu_skip_size_hist[wd][ht]++;
}
void ihevcd_update_pu_size(pu_t *ps_pu)
{
    WORD32 wd, ht;

    wd = (ps_pu->b4_wd);
    ht = (ps_pu->b4_ht);
    gs_ihevcd_stat.stat_pu_all_size_hist[wd][ht]++;
    if(ps_pu->b1_intra_flag)
    {
        gs_ihevcd_stat.stat_pu_intra_size_hist[wd][ht]++;
    }
    else
    {
        gs_ihevcd_stat.stat_pu_inter_size_hist[wd][ht]++;



        if(ps_pu->b1_merge_flag)
            gs_ihevcd_stat.stat_pu_merge_size_hist[wd][ht]++;

        if(ps_pu->b2_pred_mode == PRED_BI)
            gs_ihevcd_stat.stat_pu_bipred_size_hist[wd][ht]++;

        switch(ps_pu->b2_pred_mode)
        {
            case PRED_L0:
                if((ps_pu->mv.s_l0_mv.i2_mvx == 0) &&
                   (ps_pu->mv.s_l0_mv.i2_mvy == 0))
                {
                    gs_ihevcd_stat.stat_pu_zeromv_size_hist[wd][ht]++;
                }

                if((ABS(ps_pu->mv.s_l0_mv.i2_mvx) < 4) &&
                   (ABS(ps_pu->mv.s_l0_mv.i2_mvy) < 4))
                {
                    gs_ihevcd_stat.stat_pu_zeromvfpel_size_hist[wd][ht]++;
                }

                break;

            case PRED_L1:
                if((ps_pu->mv.s_l1_mv.i2_mvx == 0) &&
                   (ps_pu->mv.s_l1_mv.i2_mvy == 0))
                {
                    gs_ihevcd_stat.stat_pu_zeromv_size_hist[wd][ht]++;
                }

                if((ABS(ps_pu->mv.s_l1_mv.i2_mvx) < 4) &&
                   (ABS(ps_pu->mv.s_l1_mv.i2_mvy) < 4))
                {
                    gs_ihevcd_stat.stat_pu_zeromvfpel_size_hist[wd][ht]++;
                }
                break;


            case PRED_BI:
                if((ps_pu->mv.s_l0_mv.i2_mvx == 0) &&
                   (ps_pu->mv.s_l0_mv.i2_mvy == 0) &&
                   (ps_pu->mv.s_l1_mv.i2_mvx == 0) &&
                   (ps_pu->mv.s_l1_mv.i2_mvy == 0))
                {
                    gs_ihevcd_stat.stat_pu_zeromv_size_hist[wd][ht]++;
                }
                if((ABS(ps_pu->mv.s_l0_mv.i2_mvx) < 4) &&
                   (ABS(ps_pu->mv.s_l0_mv.i2_mvy) < 4) &&
                   (ABS(ps_pu->mv.s_l1_mv.i2_mvx) < 4) &&
                   (ABS(ps_pu->mv.s_l1_mv.i2_mvy) < 4))
                {
                    gs_ihevcd_stat.stat_pu_zeromvfpel_size_hist[wd][ht]++;
                }

                break;

        }
    }
}


void ihevcd_print_pu_size_hist(UWORD32 *pu4_buf)
{
    WORD32 i, j;


    for(i = 0; i < (MAX_CTB_SIZE / MIN_PU_SIZE); i++)
    {
        for(j = 0; j < (MAX_CTB_SIZE / MIN_PU_SIZE); j++)
        {
            printf("%12d ", pu4_buf[j]);
        }
        pu4_buf += (MAX_CTB_SIZE / MIN_PU_SIZE);
        printf("\n");
    }
}

void ihevcd_print_pu_size_hist_normalized(UWORD32 *pu4_buf)
{
    WORD32 i, j;
    WORD32 sum;
    UWORD32 *pu4_buf_orig = pu4_buf;
    sum = 0;

    for(i = 0; i < (MAX_CTB_SIZE / MIN_PU_SIZE); i++)
    {
        for(j = 0; j < (MAX_CTB_SIZE / MIN_PU_SIZE); j++)
        {
            sum += pu4_buf[j] * (i + 1) * (j + 1) * 16;
        }
        pu4_buf += (MAX_CTB_SIZE / MIN_PU_SIZE);
    }

    pu4_buf = pu4_buf_orig;
    for(i = 0; i < (MAX_CTB_SIZE / MIN_PU_SIZE); i++)
    {
        for(j = 0; j < (MAX_CTB_SIZE / MIN_PU_SIZE); j++)
        {
            double num = pu4_buf[j] * (i + 1) * (j + 1) * 16 * 100.0;
            printf("%6.2f ", num  / sum);
        }
        pu4_buf += (MAX_CTB_SIZE / MIN_PU_SIZE);
        printf("\n");
    }
}

void ihevcd_print_pu_size_hist_percentage(UWORD32 *pu4_num, UWORD32 *pu4_denom)
{
    WORD32 i, j;


    for(i = 0; i < (MAX_CTB_SIZE / MIN_PU_SIZE); i++)
    {
        for(j = 0; j < (MAX_CTB_SIZE / MIN_PU_SIZE); j++)
        {
            double val;
            val = 0;
            if(pu4_denom[j])
            {
                val = (pu4_num[j] * 100.0) / pu4_denom[j];
                printf("%6.2f ", val);
            }
            else
            {
                if(0 == pu4_num[j])
                    printf("%6.2f ", 0.0);
                else
                    printf("NaN   ");
            }
        }
        pu4_num += (MAX_CTB_SIZE / MIN_PU_SIZE);
        pu4_denom += (MAX_CTB_SIZE / MIN_PU_SIZE);
        printf("\n");
    }
}

void ihevcd_print_pu_statistics()
{

    printf("\n\nPU Sizes\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes Intra\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_intra_size_hist[0][0]);

    printf("\n\nPU Sizes Inter\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_inter_size_hist[0][0]);

    printf("\n\nPU Sizes Skip\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_skip_size_hist[0][0]);

    printf("\n\nPU Sizes Merge\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_merge_size_hist[0][0]);

    printf("\n\nPU Sizes BiPred\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_bipred_size_hist[0][0]);

    printf("\n\nPU Sizes Zero MV\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_zeromv_size_hist[0][0]);

    printf("\n\nPU Sizes Zero MV including subpel MV less than +/- 1 in fullpel units\n\n");
    ihevcd_print_pu_size_hist(&gs_ihevcd_stat.stat_pu_zeromvfpel_size_hist[0][0]);

    printf("\n\nPU Sizes percentage \n\n");
    ihevcd_print_pu_size_hist_normalized(&gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes Intra ratio w.r.t all PUs\n\n");
    ihevcd_print_pu_size_hist_percentage(&gs_ihevcd_stat.stat_pu_intra_size_hist[0][0], &gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes Inter ratio w.r.t all PUs\n\n");
    ihevcd_print_pu_size_hist_percentage(&gs_ihevcd_stat.stat_pu_inter_size_hist[0][0], &gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes Skip ratio w.r.t all PUs\n\n");
    ihevcd_print_pu_size_hist_percentage(&gs_ihevcd_stat.stat_pu_skip_size_hist[0][0], &gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes Merge ratio w.r.t all PUs\n\n");
    ihevcd_print_pu_size_hist_percentage(&gs_ihevcd_stat.stat_pu_merge_size_hist[0][0], &gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes BiPred ratio w.r.t all PUs\n\n");
    ihevcd_print_pu_size_hist_percentage(&gs_ihevcd_stat.stat_pu_bipred_size_hist[0][0], &gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes Zero MV ratio w.r.t all PUs\n\n");
    ihevcd_print_pu_size_hist_percentage(&gs_ihevcd_stat.stat_pu_zeromv_size_hist[0][0], &gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

    printf("\n\nPU Sizes Zero MV including subpel MV less than +/- 1 in fullpel units ratio w.r.t all PUs\n\n");
    ihevcd_print_pu_size_hist_percentage(&gs_ihevcd_stat.stat_pu_zeromvfpel_size_hist[0][0], &gs_ihevcd_stat.stat_pu_all_size_hist[0][0]);

}

void ihevcd_print_statistics()
{
    ihevcd_print_transform_statistics();
    ihevcd_print_pu_statistics();
}
#endif
