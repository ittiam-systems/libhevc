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
*  ihevcd_debug.c
*
* @brief
*  Functions used for codec debugging
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

/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ihevc_typedefs.h"
#include "iv.h"
#include "ivd.h"
#include "ihevcd_cxa.h"

#include "ihevc_defs.h"
#include "ihevc_debug.h"
#include "ihevc_structs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"

#include "ihevc_common_tables.h"
#include "ihevc_error.h"
#include "ihevc_cabac_tables.h"

#include "ihevcd_trace.h"
#include "ihevcd_defs.h"
#include "ihevcd_function_selector.h"
#include "ihevcd_structs.h"
#include "ihevcd_debug.h"

#include "ihevc_buf_mgr.h"
#include "ihevc_dpb_mgr.h"
#if DEBUG_CODE

void ihevcd_debug_dump_mv_map(codec_t *ps_codec)
{

    process_ctxt_t *ps_proc;
    sps_t *ps_sps = ps_codec->s_parse.ps_sps;
    WORD32 num_minpu_in_ctb, ctb_size, x, y, cur_pu_idx, cur_ctb_idx, pu_idx_start_ctb;
    UWORD8 *pu1_pic_pu_map_ctb;
    pu_t *ps_pu;
    WORD32 i;
    FILE *fp_mv_map, *fp_pu_idx_map, *fp_pu, *fp_mv_print, *fp_mv_print_1;
    char l0_mvx[50], l0_mvy[50], l1_mvx[50], l1_mvy[50];
    UWORD32 *pu4_pu_done, num_pu_done = 0, is_pu_done;

    pu4_pu_done = malloc(MAX_HT * MAX_WD / 4 / 4 * sizeof(UWORD32));

    ctb_size = (1 << ps_sps->i1_log2_ctb_size);

    num_minpu_in_ctb = (ctb_size / MIN_PU_SIZE) * (ctb_size / MIN_PU_SIZE);

    ps_pu = &ps_codec->s_parse.ps_pic_pu[0];
    fp_mv_map = fopen("d:\\dump\\mv_map.txt", "a");
    fp_mv_print = fopen("d:\\dump\\mv_print.txt", "a");
    fp_mv_print_1 = fopen("d:\\dump\\mv_print_1.txt", "a");
    if((NULL == fp_mv_map) || (NULL == fp_mv_print) || (NULL == fp_mv_print_1))
    {
        printf("\n Couldn't open mv dump files");
    }
    else
    {
#if 0
        fp_pu_idx_map = fopen("d:\\dump\\pu_idx_map.txt", "ab");
        fp_pu = fopen("d:\\dump\\pu.txt", "ab");
        {
            WORD32 last_ctb_idx, last_pu_idx;
            last_ctb_idx = ps_sps->i2_pic_ht_in_ctb * ps_sps->i2_pic_wd_in_ctb * num_minpu_in_ctb;
            fwrite(ps_codec->s_parse.pu1_pic_pu_map,last_ctb_idx,sizeof(UWORD32),fp_pu_idx_map );
            fwrite(ps_codec->s_parse.pu4_pic_pu_idx,last_ctb_idx * num_minpu_in_ctb, sizeof(UWORD8),fp_pu_idx_map );

            last_pu_idx = ps_codec->s_parse.pu4_pic_pu_idx[last_ctb_idx];
            fwrite(ps_codec->s_parse.ps_pic_pu,last_pu_idx , sizeof(pu_t),fp_pu );
        }
#endif
        fprintf(fp_mv_map, "\nPOC=%d\n", ps_codec->ps_slice_hdr_base[0].i4_abs_pic_order_cnt);

        {
            WORD32 last_ctb_idx, last_ctb_pu_idx, last_pu_idx;
            last_ctb_idx = ps_sps->i2_pic_ht_in_ctb * ps_sps->i2_pic_wd_in_ctb;
            last_ctb_pu_idx = ps_codec->s_parse.pu4_pic_pu_idx[last_ctb_idx];

            pu1_pic_pu_map_ctb = ps_codec->s_parse.pu1_pic_pu_map
                            + last_ctb_idx * num_minpu_in_ctb;

            last_pu_idx  = last_ctb_pu_idx + pu1_pic_pu_map_ctb[(((ps_sps->i2_pic_wd_in_ctb * ctb_size - 1) & (ctb_size - 1)) >> 2) + ((((ps_sps->i2_pic_ht_in_ctb * ctb_size - 1) & (ctb_size - 1))) >> 2) * (ctb_size >> 2)];

            for(i = 0; i < last_pu_idx; i++)
            {
                ps_pu = &ps_codec->s_parse.ps_pic_pu[i];

                fprintf(fp_mv_print_1, "\n-----------------------");

                fprintf(fp_mv_print_1, "\n pu_x = %d, pu_y = %d",
                        (ps_pu->b4_pos_x << 2), (ps_pu->b4_pos_y << 2));
                fprintf(fp_mv_print_1, "\n pu_wd = %d, pu_ht = %d", ((ps_pu->b4_wd + 1) << 2), ((ps_pu->b4_ht + 1) << 2));
                if(ps_pu->b2_pred_mode == PRED_L0)
                    fprintf(fp_mv_print_1, "\n Pred = 0,Ref_idx = %d, MV l0 = %4d %4d", ps_pu->mv.i1_l0_ref_idx, ps_pu->mv.s_l0_mv.i2_mvx,
                            ps_pu->mv.s_l0_mv.i2_mvy);
                else if(ps_pu->b2_pred_mode == PRED_L1)
                    fprintf(fp_mv_print_1, "\n Pred = 1,Ref_idx = %d,  MV l1 = %4d %4d", ps_pu->mv.i1_l1_ref_idx, ps_pu->mv.s_l1_mv.i2_mvx,
                            ps_pu->mv.s_l1_mv.i2_mvy);
                else
                    fprintf(fp_mv_print_1, "\n Pred = 2,Ref_idx = %d,Ref_idx = %d, MV l0 = %4d %4d, MV l1 = %4d %4d", ps_pu->mv.i1_l0_ref_idx, ps_pu->mv.i1_l1_ref_idx,
                            ps_pu->mv.s_l0_mv.i2_mvx, ps_pu->mv.s_l0_mv.i2_mvy,
                            ps_pu->mv.s_l1_mv.i2_mvx, ps_pu->mv.s_l1_mv.i2_mvy);
            }
        }
        for(y = 0; y < (ps_sps->i2_pic_height_in_luma_samples / MIN_PU_SIZE); y++)
        {
            for(x = 0; x < (ps_sps->i2_pic_width_in_luma_samples / MIN_PU_SIZE); x++)
            {
                cur_ctb_idx = (x * MIN_PU_SIZE / ctb_size) + (y * MIN_PU_SIZE / ctb_size) * ps_sps->i2_pic_wd_in_ctb;
                pu_idx_start_ctb = ps_codec->s_parse.pu4_pic_pu_idx[cur_ctb_idx];

                pu1_pic_pu_map_ctb = ps_codec->s_parse.pu1_pic_pu_map
                                + cur_ctb_idx * num_minpu_in_ctb;

                cur_pu_idx  = pu_idx_start_ctb + pu1_pic_pu_map_ctb[(((x * 4) & (ctb_size - 1)) >> 2) + ((((y * 4) & (ctb_size - 1))) >> 2) * (ctb_size >> 2)];

                ps_pu = &ps_codec->s_parse.ps_pic_pu[cur_pu_idx];

                is_pu_done = 0;
                for(i = 0; i < num_pu_done; i++)
                {
                    if(pu4_pu_done[num_pu_done - i - 1] == cur_pu_idx)
                    {
                        is_pu_done = 1;
                        break;
                    }
                }

                if(is_pu_done)
                {
                    fprintf(fp_mv_map, ",");
                }
                else
                {
                    sprintf(l0_mvx, "%d", ps_pu->mv.s_l0_mv.i2_mvx);
                    sprintf(l0_mvy, "%d", ps_pu->mv.s_l0_mv.i2_mvy);
                    sprintf(l1_mvx, "%d", ps_pu->mv.s_l1_mv.i2_mvx);
                    sprintf(l1_mvy, "%d", ps_pu->mv.s_l1_mv.i2_mvy);
                    fprintf(fp_mv_map, "(%s:%s)(%s:%s),", l0_mvx, l0_mvy, l1_mvx, l1_mvy);

                    fprintf(fp_mv_print, "\n-----------------------");

/*
                    printf("\n CTB X = %d, Y = %d",
                           (x*MIN_PU_SIZE / ctb_size), (y*MIN_PU_SIZE / ctb_size));
*/

                    fprintf(fp_mv_print, "\n pu_x = %d, pu_y = %d",
                            (ps_pu->b4_pos_x << 2), (ps_pu->b4_pos_y << 2));
                    fprintf(fp_mv_print, "\n pu_wd = %d, pu_ht = %d", ((ps_pu->b4_wd + 1) << 2), ((ps_pu->b4_ht + 1) << 2));
                    if(ps_pu->b2_pred_mode == PRED_L0)
                        fprintf(fp_mv_print, "\n Pred = 0,Ref_idx = %d, MV l0 = %4d %4d", ps_pu->mv.i1_l0_ref_idx, ps_pu->mv.s_l0_mv.i2_mvx,
                                ps_pu->mv.s_l0_mv.i2_mvy);
                    else if(ps_pu->b2_pred_mode == PRED_L1)
                        fprintf(fp_mv_print, "\n Pred = 1,Ref_idx = %d,  MV l1 = %4d %4d", ps_pu->mv.i1_l1_ref_idx, ps_pu->mv.s_l1_mv.i2_mvx,
                                ps_pu->mv.s_l1_mv.i2_mvy);
                    else
                        fprintf(fp_mv_print, "\n Pred = 2,Ref_idx = %d,Ref_idx = %d, MV l0 = %4d %4d, MV l1 = %4d %4d", ps_pu->mv.i1_l0_ref_idx, ps_pu->mv.i1_l1_ref_idx,
                                ps_pu->mv.s_l0_mv.i2_mvx, ps_pu->mv.s_l0_mv.i2_mvy,
                                ps_pu->mv.s_l1_mv.i2_mvx, ps_pu->mv.s_l1_mv.i2_mvy);

                    pu4_pu_done[num_pu_done] = cur_pu_idx;
                    num_pu_done++;
                }
            }
            fprintf(fp_mv_map, "\n");
        }
    }
    fclose(fp_mv_map);
    fclose(fp_mv_print);
    fclose(fp_mv_print_1);
//            fclose(fp_pu_idx_map);
//            fclose(fp_pu);
    free(pu4_pu_done);
}

void ihevcd_debug_assert(WORD32 x)
{
    if(!x)
    {
        printf("Assert failed.. Exiting \n");
        exit(-1);
    }
}

void ihevcd_debug_dump_pic_buffers(codec_t *ps_codec)
{
    FILE *fp_pic, *fp_pic_b;
    sps_t *ps_sps = ps_codec->s_parse.ps_sps;
    static WORD32 file_open = 0;
    WORD32 vert_bs_size, horz_bs_size;
    WORD32 qp_size;
    WORD32 qp_const_flag_size;
    WORD32 loop_filter_size;
    WORD32 loop_filter_buffer;
    WORD32 pic_intra_flag_size;

    vert_bs_size = ps_codec->i4_max_wd / 8 + MAX_CTB_SIZE / 8;

    /* Max Number of horizontal edges - extra MAX_CTB_SIZE / 8 to handle the last 4 rows separately(shifted CTB processing) */
    vert_bs_size *= (ps_codec->i4_max_ht + MAX_CTB_SIZE) / MIN_TU_SIZE;

    /* Number of bytes */
    vert_bs_size /= 8;

    /* Two bits per edge */
    vert_bs_size *= 2;

    /* Max Number of horizontal edges */
    horz_bs_size = ps_codec->i4_max_ht / 8 + MAX_CTB_SIZE / 8;

    /* Max Number of vertical edges - extra MAX_CTB_SIZE / 8 to handle the last 4 columns separately(shifted CTB processing) */
    horz_bs_size *= (ps_codec->i4_max_wd + MAX_CTB_SIZE) / MIN_TU_SIZE;

    /* Number of bytes */
    horz_bs_size /= 8;

    /* Two bits per edge */
    horz_bs_size *= 2;

    qp_size = (ps_codec->i4_max_ht * ps_codec->i4_max_wd) / (MIN_CU_SIZE * MIN_CU_SIZE);

    /* Max CTBs in a row */
    qp_const_flag_size = ps_codec->i4_max_wd / MIN_CTB_SIZE;

    /* Max CTBs in a column */
    qp_const_flag_size *= ps_codec->i4_max_ht / MIN_CTB_SIZE;

    /* Number of bytes */
    qp_const_flag_size /= 8;

    loop_filter_size = ((ps_codec->i4_max_wd  + 64) / MIN_CU_SIZE) * ((ps_codec->i4_max_ht + 64) / MIN_CU_SIZE) / 8;

    loop_filter_buffer = (ps_codec->i4_max_wd + 63) >> 6;
    loop_filter_buffer += 1;

    loop_filter_size -= loop_filter_buffer;

    pic_intra_flag_size = (ps_codec->i4_max_wd / MIN_CU_SIZE) * (ps_codec->i4_max_ht / MIN_CU_SIZE) / 8;

    if(0 == file_open)
    {
        fp_pic = fopen("D:\\dump\\pic_dump.txt", "w");
        fp_pic_b = fopen("D:\\dump\\pic_dump_b.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp_pic = fopen("D:\\dump\\pic_dump.txt", "a");
        fp_pic_b = fopen("D:\\dump\\pic_dump_b.txt", "ab");
    }

    {
        WORD32 i, j;

        fwrite(ps_codec->s_parse.s_deblk_ctxt.s_bs_ctxt.pu1_pic_qp, 1, qp_size, fp_pic_b);

        fprintf(fp_pic, " Frame num :%d \n", ps_codec->u4_pic_cnt);

        for(i = 0; i < ps_codec->i4_max_ht / MIN_CU_SIZE; i++)
        {
            for(j = 0; j < ps_codec->i4_max_wd / MIN_CU_SIZE; j++)
            {
                UWORD8 u1_qp;
                WORD32 qp_strd;
                qp_strd = ps_codec->i4_max_wd / MIN_CU_SIZE;
                u1_qp = ps_codec->s_parse.s_deblk_ctxt.s_bs_ctxt.pu1_pic_qp[j + i * qp_strd];
                fprintf(fp_pic, "%d \t", u1_qp);
            }
            fprintf(fp_pic, "\n");
        }
    }
/*
    fwrite(ps_codec->s_parse.s_deblk_ctxt.s_bs_ctxt.pu4_pic_vert_bs, 1, vert_bs_size, fp_pic);
    fwrite(ps_codec->s_parse.s_deblk_ctxt.s_bs_ctxt.pu4_pic_horz_bs, 1, horz_bs_size, fp_pic);
    fwrite(ps_codec->s_parse.s_deblk_ctxt.s_bs_ctxt.pu1_pic_qp_const_in_ctb, 1, qp_const_flag_size, fp_pic);
    fwrite(ps_codec->s_parse.s_deblk_ctxt.pu1_pic_no_loop_filter_flag, 1, loop_filter_size, fp_pic);
    fwrite(ps_codec->s_parse.pu1_pic_intra_flag, 1, pic_intra_flag_size, fp_pic);
*/

    //fwrite(au1_pic_avail_ctb_flags, 1, ps_sps->i2_pic_wd_in_ctb * ps_sps->i2_pic_ht_in_ctb, fp_pic);
    //fwrite(au4_pic_ctb_slice_xy, 4, ps_sps->i2_pic_wd_in_ctb * ps_sps->i2_pic_ht_in_ctb, fp_pic);

    fclose(fp_pic);
    fclose(fp_pic_b);

}


void ihevcd_debug_dump_pic_pu(codec_t *ps_codec)
{
    FILE *fp_pic_pu;
    FILE *fp_pic_pu_idx;
    static WORD32 file_open = 0;
    WORD32 num_pu_in_frame;
    sps_t *ps_sps;

    if(0 == file_open)
    {
        fp_pic_pu = fopen("D:\\dump\\pic_pu.txt", "wb");
        fp_pic_pu_idx = fopen("D:\\dump\\pic_pu_idx.txt", "wb");
        file_open = 1;
    }
    else
    {
        return;
        fp_pic_pu = fopen("D:\\dump\\pic_pu.txt", "ab");
        fp_pic_pu_idx = fopen("D:\\dump\\pic_pu_idx.txt", "ab");
    }
    ps_sps = ps_codec->s_parse.ps_sps;
    num_pu_in_frame = ps_codec->s_parse.pu4_pic_pu_idx[ps_sps->i4_pic_size_in_ctb];

    fwrite(ps_codec->s_parse.ps_pic_pu, sizeof(pu_t), num_pu_in_frame, fp_pic_pu);
    fwrite(ps_codec->s_parse.pu4_pic_pu_idx, sizeof(UWORD32), ps_sps->i4_pic_size_in_ctb + 1, fp_pic_pu_idx);

    fclose(fp_pic_pu);

}


void ihevcd_debug_init_tmp_buf(UWORD8 *pu1_buf_luma, UWORD8 *pu1_buf_chroma)
{
    memset(pu1_buf_luma, 0, 4 * MAX_CTB_SIZE * MAX_CTB_SIZE * sizeof(UWORD8));
    memset(pu1_buf_chroma, 0, 4 * MAX_CTB_SIZE * MAX_CTB_SIZE * sizeof(UWORD8));
}

void ihevcd_debug_process_tmp_buf(UWORD8 *pu1_buf_luma, UWORD8 *pu1_buf_chroma)
{
    WORD32 row, col;
    UWORD8 *pu1_tmp_buf_luma;
    UWORD8 *pu1_tmp_buf_chroma;

    FILE *fp_luma, *fp_chroma;

    pu1_tmp_buf_luma = (UWORD8 *)calloc(4 * MAX_CTB_SIZE * MAX_CTB_SIZE, 1);
    pu1_tmp_buf_chroma = (UWORD8 *)calloc(4 * MAX_CTB_SIZE * MAX_CTB_SIZE, 1);

    for(row = 0; row < 2 * MAX_CTB_SIZE; row++)
    {
        for(col = 0; col < 2 * MAX_CTB_SIZE; col++)
        {
            if(0 != pu1_buf_luma[row * 2 * MAX_CTB_SIZE + col])
                pu1_tmp_buf_luma[row * 2 * MAX_CTB_SIZE + col] = 0xFF;
            if(0 != pu1_buf_chroma[row * 2 * MAX_CTB_SIZE + col])
                pu1_tmp_buf_chroma[row * 2 * MAX_CTB_SIZE + col] = 0xFF;
        }
    }

    fp_luma = fopen("D:\\dump\\win_sao_tmp_buf_luma.yuv", "wb");
    fp_chroma = fopen("D:\\dump\\win_sao_tmp_buf_chroma.yuv", "wb");

    fwrite(pu1_tmp_buf_luma, 4 * MAX_CTB_SIZE * MAX_CTB_SIZE, 1, fp_luma);
    fwrite(pu1_tmp_buf_chroma, 4 * MAX_CTB_SIZE * MAX_CTB_SIZE, 1, fp_chroma);

    fclose(fp_luma);
    fclose(fp_chroma);
}

void ihevcd_debug_print_struct_sizes()
{
    printf("sizeof(tu_t) %d\n", sizeof(tu_t));
    printf("sizeof(pu_t) %d\n", sizeof(pu_t));
    printf("sizeof(pu_mv_t) %d\n", sizeof(pu_mv_t));
    printf("sizeof(vps_t) %d\n", sizeof(vps_t));
    printf("sizeof(sps_t) %d\n", sizeof(sps_t));
    printf("sizeof(pps_t) %d\n", sizeof(pps_t));
    printf("sizeof(slice_header_t) %d\n", sizeof(slice_header_t));

    printf("sizeof(codec_t) %d\n", sizeof(codec_t));
    printf("sizeof(parse_ctxt_t) %d\n", sizeof(parse_ctxt_t));
    printf("sizeof(process_ctxt_t) %d\n", sizeof(process_ctxt_t));
    printf("sizeof(cab_ctxt_t) %d\n", sizeof(cab_ctxt_t));
    return;
}

void ihevcd_debug_dump_pic(UWORD8 *pu1_cur_pic_luma,
                           UWORD8 *pu1_cur_pic_chroma,
                           WORD32 pic_wd,
                           WORD32 pic_ht,
                           WORD32 pic_strd)
{
    FILE *fp;
    static WORD32 file_open = 0;
    WORD32 row;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_pre_ilf_dec_order.yuv", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_pre_ilf_dec_order.yuv", "ab");
    }

    for(row = 0; row < pic_ht; row++)
    {
        fwrite(pu1_cur_pic_luma, sizeof(UWORD8), pic_wd, fp);
        pu1_cur_pic_luma += pic_strd;
    }
    for(row = 0; row < pic_ht / 2; row++)
    {
        fwrite(pu1_cur_pic_chroma, sizeof(UWORD8), pic_wd, fp);
        pu1_cur_pic_chroma += pic_strd;
    }

    fclose(fp);
}

void ihevcd_debug_dump_bs(UWORD32 *pu4_pic_vert_bs,
                          UWORD32 *pu4_pic_horz_bs,
                          WORD32 vert_size_in_bytes,
                          WORD32 horz_size_in_bytes)
{
    FILE *fp_vert, *fp_horz;
    static WORD32 files_open = 0;

    if(files_open == 0)
    {
        fp_vert = fopen("D:\\dump\\win_vert_bs_dec_order.txt", "wb");
        fp_horz = fopen("D:\\dump\\win_horz_bs_dec_order.txt", "wb");
        files_open = 1;
    }
    else
    {
        fp_vert = fopen("D:\\dump\\win_vert_bs_dec_order.txt", "ab");
        fp_horz = fopen("D:\\dump\\win_horz_bs_dec_order.txt", "ab");
    }

    fwrite(pu4_pic_vert_bs, sizeof(UWORD8), vert_size_in_bytes, fp_vert);
    fwrite(pu4_pic_horz_bs, sizeof(UWORD8), horz_size_in_bytes, fp_horz);

    fclose(fp_vert);
    fclose(fp_horz);
}

void ihevcd_debug_dump_qp(UWORD8 *pu1_qp, WORD32 size_in_bytes)
{
    FILE *fp;
    static WORD32 file_open = 0;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_qp_dec_order.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_qp_dec_order.txt", "ab");
    }

    fwrite(pu1_qp, sizeof(UWORD8), size_in_bytes, fp);

    fclose(fp);
}

void ihevcs_dump_qp_const_in_ctb(UWORD8 *pu1_qp_const_in_ctb, WORD32 size_in_bytes)
{
    FILE *fp;
    static WORD32 file_open = 0;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_qp_const_ctb_dec_order.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_qp_const_ctb_dec_order.txt", "ab");
    }

    fwrite(pu1_qp_const_in_ctb, sizeof(UWORD8), size_in_bytes, fp);

    fclose(fp);
}


void ihevcd_debug_dump_no_loop_filter(UWORD8 *pu1_pic_no_loop_filter, WORD32 size_in_bytes)
{
    FILE *fp;
    static WORD32 file_open = 0;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_no_loop_filter_dec_order.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_no_loop_filter_dec_order.txt", "ab");
    }

    fwrite(pu1_pic_no_loop_filter, sizeof(UWORD8), size_in_bytes, fp);

    fclose(fp);
}

void ihevcd_debug_dump_offsets(WORD32 beta_offset_div_2, WORD32 tc_offset_div_2, WORD32 qp_offset_u, WORD32 qp_offset_v)
{
    FILE *fp;
    static WORD32 file_open = 0;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_offsets.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_offsets.txt", "ab");
    }

    fwrite(&beta_offset_div_2, sizeof(WORD32), 1, fp);
    fwrite(&tc_offset_div_2, sizeof(WORD32), 1, fp);
    fwrite(&qp_offset_u, sizeof(WORD32), 1, fp);
    fwrite(&qp_offset_v, sizeof(WORD32), 1, fp);

    fclose(fp);

}

/* Debugging POC values */
void ihevcd_debug_print_ref_list_pocs(WORD32 i4_pic_order_cnt_val,
                                      slice_header_t *ps_slice_hdr,
                                      dpb_mgr_t *ps_dpb_mgr,
                                      UWORD32 u4_num_st_curr_before,
                                      UWORD32 u4_num_st_curr_after,
                                      UWORD32 u4_num_st_foll,
                                      UWORD32 u4_num_lt_curr,
                                      UWORD32 u4_num_lt_foll,
                                      WORD32 *pi4_poc_st_curr_before,
                                      WORD32 *pi4_poc_st_curr_after,
                                      WORD32 *pi4_poc_st_foll,
                                      WORD32 *pi4_poc_lt_curr,
                                      WORD32 *pi4_poc_lt_foll)
{
    WORD32 i, j;
    pic_buf_t *ps_pic_buf;
    printf("\n------------------------\nCurrent POC: %d\n", i4_pic_order_cnt_val);
    printf("\nPOCs present in Reference List L0:\n");
    for(i = 0; i < ps_slice_hdr->i1_num_ref_idx_l0_active; i++)
    {
        ps_pic_buf = (pic_buf_t *)((ps_slice_hdr->as_ref_pic_list0[i].pv_pic_buf));
        printf("POC: %d\n", ps_pic_buf->i4_abs_poc);
        printf("Longterm Reference = %d\n", ps_pic_buf->u1_used_as_ref);
    }

    if(ps_slice_hdr->i1_slice_type  == BSLICE)
    {
        printf("\nPOCs present in Reference List L1:\n");
        for(i = 0; i < ps_slice_hdr->i1_num_ref_idx_l1_active; i++)
        {
            ps_pic_buf = (pic_buf_t *)((ps_slice_hdr->as_ref_pic_list1[i].pv_pic_buf));
            printf("POC: %d\n", ps_pic_buf->i4_abs_poc);
            printf("POC LSB: %d\n", ps_pic_buf->i4_poc_lsb);
            printf("Longterm Reference = %d\n", ps_pic_buf->u1_used_as_ref);
        }
    }

    printf("\nPOCs that are to be released from DPB:\n");
    for(i = 0; i < MAX_DPB_BUFS; i++)
    {
        if(ps_dpb_mgr->as_dpb_info[i].ps_pic_buf)
        {
            WORD32 poc_found = 0;
            ps_pic_buf = ps_dpb_mgr->as_dpb_info[i].ps_pic_buf;

            for(j = 0; j < u4_num_st_curr_before && 0 == poc_found; j++)
            {
                if(ps_pic_buf->i4_abs_poc == pi4_poc_st_curr_before[j])
                {
                    poc_found++;
                    break;
                }
            }
            for(j = 0; j < u4_num_st_curr_after && 0 == poc_found; j++)
            {
                if(ps_pic_buf->i4_abs_poc == pi4_poc_st_curr_after[j])
                {
                    poc_found++;
                    break;
                }
            }
            for(j = 0; j < u4_num_st_foll && 0 == poc_found; j++)
            {
                if(ps_pic_buf->i4_abs_poc == pi4_poc_st_foll[j])
                {
                    poc_found++;
                    break;
                }
            }
            for(j = 0; j < u4_num_lt_curr && 0 == poc_found; j++)
            {
                if(ps_pic_buf->i4_abs_poc == pi4_poc_lt_curr[j])
                {
                    poc_found++;
                    break;
                }
            }
            for(j = 0; j < u4_num_lt_foll && 0 == poc_found; j++)
            {
                if(ps_pic_buf->i4_abs_poc == pi4_poc_lt_foll[j])
                {
                    poc_found++;
                    break;
                }
            }

            if(0 == poc_found)
                printf("POC: %d\n", ps_pic_buf->i4_abs_poc);
        }
    }
}

void ihevcd_debug_validate_padded_region(process_ctxt_t *ps_proc)
{
    sps_t *ps_sps;
    codec_t *ps_codec;
    UWORD8 *pu1_src;
    UWORD16 *pu2_src;
    UWORD8 *pu1_validate;
    UWORD16 *pu2_validate;
    WORD32 i, j;
    WORD32 pic_ht, pic_wd;
    WORD32 src_strd;

    FILE *fp;
    static WORD32 file_open = 0;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\debug_padding.yuv", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\debug_padding.yuv", "ab");
    }


    if(NULL == fp)
    {
        printf("\nCannot Open file\n\n");
        return;
    }

    /* pu2_src and pu2_validate are for chroma */

    ps_sps = ps_proc->ps_sps;
    ps_codec = ps_proc->ps_codec;
    pu1_src = ps_proc->pu1_cur_pic_luma;
    pu2_src = (UWORD16 *)ps_proc->pu1_cur_pic_chroma;
    pic_ht = ps_sps->i2_pic_height_in_luma_samples;
    pic_wd = ps_sps->i2_pic_width_in_luma_samples;
    src_strd = ps_codec->i4_strd;

    pu1_validate = (UWORD8 *)calloc((pic_wd + PAD_LEFT + PAD_RIGHT) * (pic_ht + PAD_TOP + PAD_BOT) * 3 / 2, 1);
    pu2_validate = (UWORD16 *)(pu1_validate + (pic_wd + PAD_LEFT + PAD_RIGHT) * (pic_ht + PAD_TOP + PAD_BOT));

    for(i = 0; i < pic_ht; i++)
    {
        for(j = 0; j < PAD_LEFT; j++)
        {
            if(pu1_src[j - PAD_LEFT] != pu1_src[0])
            {
                pu1_validate[j + (PAD_TOP + i) * src_strd] = 255;
            }
        }

        for(j = 0; j < PAD_RIGHT; j++)
        {
            if(pu1_src[pic_wd + j] != pu1_src[pic_wd - 1])
            {
                pu1_validate[pic_wd + j + PAD_LEFT + (PAD_TOP + i) * src_strd] = 255;
            }
        }

        pu1_src += src_strd;
    }

    pu1_src = ps_proc->pu1_cur_pic_luma - PAD_LEFT;
    for(i = 0; i < pic_wd + PAD_LEFT + PAD_RIGHT; i++)
    {
        for(j = 0; j < PAD_TOP; j++)
        {
            if(pu1_src[(j - PAD_TOP) * src_strd] != pu1_src[0])
            {
                pu1_validate[i + j * src_strd] = 255;
            }
        }

        for(j = 0; j < PAD_BOT; j++)
        {
            if(pu1_src[(pic_ht + j) * src_strd] != pu1_src[(pic_ht - 1) * src_strd])
            {
                pu1_validate[i + (j + pic_ht + PAD_TOP) * src_strd] = 255;
            }
        }

        pu1_src += 1;
    }

    for(i = 0; i < pic_ht / 2; i++)
    {
        for(j = 0; j < PAD_LEFT / 2; j++)
        {
            if(pu2_src[j - PAD_LEFT / 2] != pu2_src[0])
            {
                pu2_validate[j + (PAD_TOP / 2 + i) * src_strd / 2] = 0xFFFF;
            }
        }

        for(j = 0; j < PAD_RIGHT / 2; j++)
        {
            if(pu2_src[pic_wd / 2 + j] != pu2_src[pic_wd / 2 - 1])
            {
                pu2_validate[pic_wd / 2 + j + PAD_LEFT / 2 + (PAD_TOP / 2 + i) * src_strd / 2] = 0xFFFF;
            }
        }

        pu2_src += src_strd / 2;
    }
    fwrite(pu1_validate, 1, (pic_wd + PAD_LEFT + PAD_RIGHT) * (pic_ht + PAD_TOP + PAD_BOT) * 3 / 2, fp);

    free(pu1_validate);
    fclose(fp);
}

void ihevcd_debug_print_nal_info(codec_t *ps_codec, WORD32 nal_type)
{
    FILE *fp;
    static WORD32 file_open = 0;
    slice_header_t *ps_slice_hdr = ps_codec->s_parse.ps_slice_hdr_base + (ps_codec->s_parse.i4_cur_slice_idx & (MAX_SLICE_HDR_CNT - 1));
    WORD32 frame_start_flag = 0;
    WORD32 frame_poc = 0;

    if(0 == file_open)
    {
        fp = fopen("nal_info.txt", "w");
        file_open = 1;
    }
    else
    {
        fp = fopen("nal_info.txt", "a");
    }

    if(NULL == fp)
    {
        printf("Cannot open NAL info file.. Exiting\n");
        exit(-1);
    }

    /* If slice NAL, update start of frame flag */
    switch(nal_type)
    {
        case NAL_BLA_W_LP    :
        case NAL_BLA_W_DLP   :
        case NAL_BLA_N_LP    :
        case NAL_IDR_W_LP    :
        case NAL_IDR_N_LP    :
        case NAL_CRA         :
        case NAL_TRAIL_N     :
        case NAL_TRAIL_R     :
        case NAL_TSA_N       :
        case NAL_TSA_R       :
        case NAL_STSA_N      :
        case NAL_STSA_R      :
        case NAL_RADL_N      :
        case NAL_RADL_R      :
        case NAL_RASL_N      :
        case NAL_RASL_R      :
            frame_start_flag = ps_slice_hdr->i1_first_slice_in_pic_flag;
            frame_poc = ps_slice_hdr->i4_abs_pic_order_cnt;
            ps_codec->i4_first_pic_done = 1;
            break;

        default:
            frame_start_flag = 0;
            frame_poc = 0;
            break;
    }
    fprintf(fp, "NALType=%d;NumBytes=%d;POC=%d;FrameStart=%d\n",
            nal_type,
            ps_codec->i4_nal_ofst + ps_codec->i4_nal_len,
            frame_poc,
            frame_start_flag);

    fclose(fp);
}

typedef struct
{
    UWORD8 au1_src[8 * 4];
    WORD32 src_strd;
    WORD32 bs;
    WORD32 qp_p;
    WORD32 qp_q;
    WORD32 beta_offset_div_2;
    WORD32 tc_offset_div_2;
    WORD32 filter_p;
    WORD32 filter_q;
}deblk_luma_t;

typedef struct
{
    UWORD8 au1_src[8 * 4];
    WORD32 src_strd;
    WORD32 bs;
    WORD32 qp_p;
    WORD32 qp_q;
    WORD32 qp_offset_u;
    WORD32 qp_offset_v;
    WORD32 tc_offset_div_2;
    WORD32 filter_p;
    WORD32 filter_q;
}deblk_chroma_t;


void ihevcd_debug_deblk_luma_vert(UWORD8 *pu1_src,
                                  WORD32 src_strd,
                                  WORD32 bs,
                                  WORD32 quant_param_p,
                                  WORD32 quant_param_q,
                                  WORD32 beta_offset_div2,
                                  WORD32 tc_offset_div2,
                                  WORD32 filter_flag_p,
                                  WORD32 filter_flag_q)
{
    FILE *fp;
    static WORD32 file_open = 0;
    WORD32 row, col;
    deblk_luma_t s_deblk_luma;

    pu1_src -= 4;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_deblk_luma_vert.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_deblk_luma_vert.txt", "ab");
    }

    for(row = 0; row < 4; row++)
    {
        for(col = 0; col < 8; col++)
        {
            s_deblk_luma.au1_src[row * 8 + col] = pu1_src[row * src_strd + col];
        }
    }
    s_deblk_luma.src_strd = src_strd;
    s_deblk_luma.bs = bs;
    s_deblk_luma.qp_p = quant_param_p;
    s_deblk_luma.qp_q = quant_param_q;
    s_deblk_luma.beta_offset_div_2 = beta_offset_div2;
    s_deblk_luma.tc_offset_div_2 = tc_offset_div2;
    s_deblk_luma.filter_p = filter_flag_p;
    s_deblk_luma.filter_q = filter_flag_q;

    fwrite(&s_deblk_luma, sizeof(deblk_luma_t), 1, fp);

    fclose(fp);
}

void ihevcd_debug_deblk_luma_horz(UWORD8 *pu1_src,
                                  WORD32 src_strd,
                                  WORD32 bs,
                                  WORD32 quant_param_p,
                                  WORD32 quant_param_q,
                                  WORD32 beta_offset_div2,
                                  WORD32 tc_offset_div2,
                                  WORD32 filter_flag_p,
                                  WORD32 filter_flag_q)
{
    FILE *fp;
    static WORD32 file_open = 0;
    WORD32 row, col;
    deblk_luma_t s_deblk_luma;

    pu1_src -= 4 * src_strd;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_deblk_luma_horz.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_deblk_luma_horz.txt", "ab");
    }

    for(row = 0; row < 8; row++)
    {
        for(col = 0; col < 4; col++)
        {
            s_deblk_luma.au1_src[row * 4 + col] = pu1_src[row * src_strd + col];
        }
    }
    s_deblk_luma.src_strd = src_strd;
    s_deblk_luma.bs = bs;
    s_deblk_luma.qp_p = quant_param_p;
    s_deblk_luma.qp_q = quant_param_q;
    s_deblk_luma.beta_offset_div_2 = beta_offset_div2;
    s_deblk_luma.tc_offset_div_2 = tc_offset_div2;
    s_deblk_luma.filter_p = filter_flag_p;
    s_deblk_luma.filter_q = filter_flag_q;

    fwrite(&s_deblk_luma, sizeof(deblk_luma_t), 1, fp);

    fclose(fp);
}

void ihevcd_debug_deblk_chroma_vert(UWORD8 *pu1_src,
                                    WORD32 src_strd,
                                    WORD32 bs,
                                    WORD32 quant_param_p,
                                    WORD32 quant_param_q,
                                    WORD32 qp_offset_u,
                                    WORD32 qp_offset_v,
                                    WORD32 tc_offset_div2,
                                    WORD32 filter_flag_p,
                                    WORD32 filter_flag_q)
{
    FILE *fp;
    static WORD32 file_open = 0;
    WORD32 row, col;
    deblk_chroma_t s_deblk_chroma;

    pu1_src -= 4;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_deblk_chroma_vert.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_deblk_chroma_vert.txt", "ab");
    }

    for(row = 0; row < 4; row++)
    {
        for(col = 0; col < 8; col++)
        {
            s_deblk_chroma.au1_src[row * 8 + col] = pu1_src[row * src_strd + col];
        }
    }
    s_deblk_chroma.src_strd = src_strd;
    s_deblk_chroma.bs = bs;
    s_deblk_chroma.qp_p = quant_param_p;
    s_deblk_chroma.qp_q = quant_param_q;
    s_deblk_chroma.qp_offset_u = qp_offset_u;
    s_deblk_chroma.qp_offset_v = qp_offset_v;
    s_deblk_chroma.tc_offset_div_2 = tc_offset_div2;
    s_deblk_chroma.filter_p = filter_flag_p;
    s_deblk_chroma.filter_q = filter_flag_q;

    fwrite(&s_deblk_chroma, sizeof(deblk_chroma_t), 1, fp);

    fclose(fp);
}


void ihevcd_debug_deblk_chroma_horz(UWORD8 *pu1_src,
                                    WORD32 src_strd,
                                    WORD32 bs,
                                    WORD32 quant_param_p,
                                    WORD32 quant_param_q,
                                    WORD32 qp_offset_u,
                                    WORD32 qp_offset_v,
                                    WORD32 tc_offset_div2,
                                    WORD32 filter_flag_p,
                                    WORD32 filter_flag_q)
{
    FILE *fp;
    static WORD32 file_open = 0;
    WORD32 row, col;
    deblk_chroma_t s_deblk_chroma;

    pu1_src -= 2 * src_strd;

    if(file_open == 0)
    {
        fp = fopen("D:\\dump\\win_deblk_chroma_horz.txt", "wb");
        file_open = 1;
    }
    else
    {
        fp = fopen("D:\\dump\\win_deblk_chroma_horz.txt", "ab");
    }

    for(row = 0; row < 4; row++)
    {
        for(col = 0; col < 8; col++)
        {
            s_deblk_chroma.au1_src[row * 8 + col] = pu1_src[row * src_strd + col];
        }
    }
    s_deblk_chroma.src_strd = src_strd;
    s_deblk_chroma.bs = bs;
    s_deblk_chroma.qp_p = quant_param_p;
    s_deblk_chroma.qp_q = quant_param_q;
    s_deblk_chroma.qp_offset_u = qp_offset_u;
    s_deblk_chroma.qp_offset_v = qp_offset_v;
    s_deblk_chroma.tc_offset_div_2 = tc_offset_div2;
    s_deblk_chroma.filter_p = filter_flag_p;
    s_deblk_chroma.filter_q = filter_flag_q;

    fwrite(&s_deblk_chroma, sizeof(deblk_chroma_t), 1, fp);

    fclose(fp);
}

#if DEBUG_PRINT_IQ_IT_RECON
void print_coeff(WORD16 *pi2_tu_coeff, WORD32 trans_size)
{
    WORD32 row, col;
    for(row = 0; row < trans_size; row++)
    {
        for(col = 0; col < trans_size; col++)
        {
            printf("%d\t", pi2_tu_coeff[row * trans_size + col]);
        }
        printf("\n");
    }
}

void print_dst(UWORD8 *pu1_dst,
               WORD32 dst_strd,
               WORD32 trans_size,
               WORD32 is_luma)
{
    WORD32 row, col;
    WORD32 inc;
    inc = is_luma == 1 ? 1 : 2;

    for(row = 0; row < trans_size; row++)
    {
        for(col = 0; col < trans_size; col++)
        {
            printf("%d\t", pu1_dst[row * dst_strd + inc * col]);
        }
        printf("\n");
    }
}
#endif
#endif
