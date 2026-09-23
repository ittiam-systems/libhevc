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
*  ihevc_hbd_sao_sse42_intr.c
*
* @brief
*  Contains function definitions for high bit depth Sample Adaptive Offset
*  (SAO) used in-loop filtering (SSE4.2).
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_sao_band_offset_luma_sse42()
*  - ihevc_hbd_sao_band_offset_chroma_sse42()
*  - ihevc_hbd_sao_edge_offset_class0_sse42()
*  - ihevc_hbd_sao_edge_offset_class0_chroma_sse42()
*  - ihevc_hbd_sao_edge_offset_class1_sse42()
*  - ihevc_hbd_sao_edge_offset_class1_chroma_sse42()
*  - ihevc_hbd_sao_edge_offset_class2_sse42()
*  - ihevc_hbd_sao_edge_offset_class2_chroma_sse42()
*  - ihevc_hbd_sao_edge_offset_class3_sse42()
*  - ihevc_hbd_sao_edge_offset_class3_chroma_sse42()
*
* @remarks
*  None
*
*******************************************************************************
*/

/**
*******************************************************************************
* @file
*  ihevc_sao_x86_intr.c
*
* @brief
*  Contains function definitions for Sample adaptive offset(SAO) used in-loop
* filtering
*
* @author
*  Ittiam
*
* @par List of Functions:
*   - ihevc_sao_band_offset_luma()
*   - ihevc_sao_band_offset_chroma()
*   - ihevc_sao_edge_offset_class0()
*   - ihevc_sao_edge_offset_class0_chroma()
*   - ihevc_sao_edge_offset_class1()
*   - ihevc_sao_edge_offset_class1_chroma()
*   - ihevc_sao_edge_offset_class2()
*   - ihevc_sao_edge_offset_class2_chroma()
*   - ihevc_sao_edge_offset_class3()
*   - ihevc_sao_edge_offset_class3_chroma()
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

#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"

#include "ihevc_sao.h"
#include "ihevc_hbd_tables_x86_intr.h"

#include <immintrin.h>

/*const static WORD8 gi1_table_edge_idx[5] = {1, 2, 0, 3, 4};
const static WORD8 gi1_table_band_idx[44] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             1, 2, 3, 4,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0};*/


/**
*******************************************************************************
*
* @brief
* Has two sets of functions : band offset and edge offset both for luma and chroma
* edge offset has horizontal ,vertical, 135 degree and 45 degree
*
* @par Description:
*
*
* @param[in-out] pu1_src
*  Pointer to the source
*
* @param[in] src_strd
*  Source stride
*
* @param[in-out] pu1_src_left
*  source left boundary
*
* @param[in-out] pu1_src_top
* Source top boundary
*
* @param[in-out] pu1_src_top_left
*  Source top left boundary
*
* @param[in] pu1_src_top_right
*  Source top right boundary
*
* @param[in] pu1_src_bot_left
*  Source bottom left boundary
*
* @param[in] pu1_avail
*  boundary availability flags
*
* @param[in] pi1_sao_offset_u
*  Chroma U sao offset values
*
* @param[in] pi1_sao_offset_v
*  Chroma V sao offset values
*
* @param[in] pi1_sao_offset
*  Luma sao offset values
*
* @param[in] wd
*  width of the source

* @param[in] ht
*  height of the source
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_sao_band_offset_luma_sse42(UWORD16 *pu2_src,
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
    WORD8 offset=0;
    WORD32 band_shift;
    UWORD16 *pu2_src_cpy;

    __m128i src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;
    __m128i band_table0_8x16b, band_table1_8x16b;
    __m128i tmp_set_128i_1, tmp_set_128i_2, tmp_set_128i_3, tmp_set_128i_4;
    __m128i m_max, m_min;
    __m128i sao_offset, const0_16x8b;
    __m128i cmp_mask, cmp_store;

    /* Updating left and top-left and top */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[row] = pu2_src[row * src_strd + (wd - 1)];
    }
    pu2_src_top_left[0] = pu2_src_top[wd - 1];
    for(col = 0; col < wd; col += 8)
    {
        tmp_set_128i_1= _mm_loadu_si128((__m128i *)(pu2_src +(ht - 1) * src_strd + offset));
        _mm_storeu_si128((__m128i *)(pu2_src_top + offset), tmp_set_128i_1);
        offset += 8;
    }

    band_shift = u4_bit_depth - 5;

    const0_16x8b = _mm_setzero_si128();
    sao_offset = _mm_loadl_epi64((__m128i *)pi1_sao_offset);
    cmp_mask = _mm_set1_epi8(15);
    m_max = _mm_set1_epi8(0xff);
    m_max = _mm_srli_epi16(m_max, (16 - u4_bit_depth));
    m_min = _mm_setzero_si128();

    band_table0_8x16b = _mm_setzero_si128();
    band_table1_8x16b = _mm_setzero_si128();

    // band table creation
    if(sao_band_pos > 15)
    {
        sao_band_pos = sao_band_pos - 16;
        band_table1_8x16b = _mm_loadu_si128((__m128i *)(gi1_table_band_idx_hbd + 16 - sao_band_pos));
        band_table0_8x16b = _mm_loadl_epi64((__m128i *)(gi1_table_band_idx_hbd + 32 - sao_band_pos));
    }
    else
    {
        band_table0_8x16b = _mm_loadu_si128((__m128i *)(gi1_table_band_idx_hbd + 16 - sao_band_pos));
        band_table1_8x16b = _mm_loadl_epi64((__m128i *)(gi1_table_band_idx_hbd + 32 - sao_band_pos));
    }

    for(col = wd; col >=8; col -= 8)
    {
        pu2_src_cpy = pu2_src;
        for(row = ht; row >0; row -= 2)
        {
            // row = 0 load 8 pixel values from 7:0 pos. relative to cur. pos.
            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(pu2_src_cpy));
            // row = 1
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));

            // shift to get 5 bit MSB
            src_temp2_8x16b = _mm_srli_epi16(src_temp0_8x16b, band_shift);
            src_temp3_8x16b = _mm_srli_epi16(src_temp1_8x16b, band_shift);

            src_temp2_8x16b = _mm_packus_epi16(src_temp2_8x16b, src_temp3_8x16b);

            // if the values >16 then put ff ,cmp_mask = dup16(15)
            cmp_store = _mm_cmpgt_epi8(src_temp2_8x16b, cmp_mask);
            // values 16 to 31 for row 0 & 1 but values <16 ==0
            tmp_set_128i_2 = _mm_and_si128(src_temp2_8x16b, cmp_store);
            // values 0 to 15 for row 0 & 1
            tmp_set_128i_1 =_mm_or_si128(src_temp2_8x16b, cmp_store);
            // values 16 to 31 for row 0 & 1 but values <16 masked to ff
            cmp_store = _mm_cmpeq_epi8(tmp_set_128i_2, const0_16x8b);
            tmp_set_128i_2 = _mm_or_si128(tmp_set_128i_2, cmp_store);

            // shuffle to get the band index
            tmp_set_128i_1 = _mm_shuffle_epi8(band_table0_8x16b, tmp_set_128i_1);
            tmp_set_128i_2 = _mm_shuffle_epi8(band_table1_8x16b, tmp_set_128i_2);

            // shuffle to get the band offset
            tmp_set_128i_1 = _mm_or_si128(tmp_set_128i_1, tmp_set_128i_2);
            tmp_set_128i_1 = _mm_shuffle_epi8(sao_offset, tmp_set_128i_1);

            // convert from 8 bit to 16 bit offset
            tmp_set_128i_2 =  _mm_cmpgt_epi8(const0_16x8b, tmp_set_128i_1);
            tmp_set_128i_3 = _mm_unpacklo_epi8(tmp_set_128i_1, tmp_set_128i_2);
            tmp_set_128i_4 = _mm_unpackhi_epi8(tmp_set_128i_1, tmp_set_128i_2);

            // add offset to the pixels
            src_temp0_8x16b = _mm_add_epi16(src_temp0_8x16b, tmp_set_128i_3);
            src_temp1_8x16b = _mm_add_epi16(src_temp1_8x16b, tmp_set_128i_4);

            // clip the pixels to appropriate range
            src_temp0_8x16b = _mm_max_epi16(src_temp0_8x16b, m_min);
            src_temp1_8x16b = _mm_max_epi16(src_temp1_8x16b, m_min);

            src_temp0_8x16b = _mm_min_epi16(src_temp0_8x16b, m_max);
            src_temp1_8x16b = _mm_min_epi16(src_temp1_8x16b, m_max);

            // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
            _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_8x16b);
            // row = 1
            _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_8x16b);

            pu2_src_cpy += (src_strd<<1);
        }
        pu2_src += 8;
    }
}
void ihevc_hbd_sao_band_offset_chroma_sse42(UWORD16 *pu2_src,
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
    WORD8 offset=0;
    WORD32 band_shift;
    UWORD16 *pu2_src_cpy;

    __m128i src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;
    __m128i band_table0_8x16b_u, band_table1_8x16b_u, band_table0_8x16b_v, band_table1_8x16b_v;
    __m128i tmp_set_128i_1, tmp_set_128i_2, tmp_set_128i_3, tmp_set_128i_0, tmp_set_128i_5, tmp_set_128i_6;
    __m128i m_max, m_min;
    __m128i sao_offset_u, sao_offset_v, const0_16x8b;
    __m128i cmp_mask, cmp_store;

    /* Updating left and top-left and top */
    /* Updating left and top and top-left */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[2 * row] = pu2_src[row * src_strd + (wd - 2)];
        pu2_src_left[2 * row + 1] = pu2_src[row * src_strd + (wd - 1)];
    }
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];
    for(col = 0; col < wd; col += 8)
    {
        tmp_set_128i_1= _mm_loadu_si128((__m128i *)(pu2_src +(ht - 1) * src_strd + offset));
        _mm_storeu_si128((__m128i *)(pu2_src_top+offset), tmp_set_128i_1);
        offset += 8;
    }

    band_shift = u4_bit_depth - 5;

    const0_16x8b = _mm_setzero_si128();
    sao_offset_u = _mm_loadl_epi64((__m128i *)pi1_sao_offset_u);
    sao_offset_v = _mm_loadl_epi64((__m128i *)pi1_sao_offset_v);
    cmp_mask = _mm_set1_epi8(15);
    m_max = _mm_set1_epi8(0xff);
    m_max = _mm_srli_epi16(m_max, (16 - u4_bit_depth));
    m_min = _mm_setzero_si128();

    // band table creation for chroma component u
    band_table0_8x16b_u = _mm_setzero_si128();
    band_table1_8x16b_u = _mm_setzero_si128();

    if(sao_band_pos_u > 15)
    {
        sao_band_pos_u = sao_band_pos_u - 16;
        band_table1_8x16b_u = _mm_loadu_si128((__m128i *)(gi1_table_band_idx_hbd + 16 - sao_band_pos_u));
        band_table0_8x16b_u = _mm_loadl_epi64((__m128i *)(gi1_table_band_idx_hbd + 32 - sao_band_pos_u));
    }
    else
    {
        band_table0_8x16b_u = _mm_loadu_si128((__m128i *)(gi1_table_band_idx_hbd + 16 - sao_band_pos_u));
        band_table1_8x16b_u = _mm_loadl_epi64((__m128i *)(gi1_table_band_idx_hbd + 32 - sao_band_pos_u));
    }

    // band table creation for chroma component v
    band_table0_8x16b_v = _mm_setzero_si128();
    band_table1_8x16b_v = _mm_setzero_si128();

    if(sao_band_pos_v > 15)
    {
        sao_band_pos_v = sao_band_pos_v - 16;
        band_table1_8x16b_v = _mm_loadu_si128((__m128i *)(gi1_table_band_idx_hbd + 16 - sao_band_pos_v));
        band_table0_8x16b_v = _mm_loadl_epi64((__m128i *)(gi1_table_band_idx_hbd + 32 - sao_band_pos_v));
    }
    else
    {
        band_table0_8x16b_v = _mm_loadu_si128((__m128i *)(gi1_table_band_idx_hbd + 16 - sao_band_pos_v));
        band_table1_8x16b_v = _mm_loadl_epi64((__m128i *)(gi1_table_band_idx_hbd + 32 - sao_band_pos_v));
    }

    for(col = wd; col >=8; col -= 8)
    {
        pu2_src_cpy = pu2_src;
        for(row = ht; row >0; row -= 4)
        {
            // row = 0 load 8 pixel values from 7:0 pos. relative to cur. pos.
            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(pu2_src_cpy));
            // row = 1
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
            // row = 2
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2 * src_strd));
            // row = 3
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 3 * src_strd));

            // shift to get 5 bit MSB
            tmp_set_128i_0 = _mm_srli_epi16(src_temp0_8x16b, band_shift);
            tmp_set_128i_1 = _mm_srli_epi16(src_temp1_8x16b, band_shift);
            tmp_set_128i_2 = _mm_srli_epi16(src_temp2_8x16b, band_shift);
            tmp_set_128i_3 = _mm_srli_epi16(src_temp3_8x16b, band_shift);

            tmp_set_128i_0 = _mm_packus_epi16(tmp_set_128i_0, tmp_set_128i_1);
            tmp_set_128i_1 = _mm_packus_epi16(tmp_set_128i_2, tmp_set_128i_3);

            // seperate u and v pixels
            // odd values
            tmp_set_128i_2 = _mm_srli_epi16(tmp_set_128i_0, 8);
            tmp_set_128i_3 = _mm_srli_epi16(tmp_set_128i_1, 8);
            // even values
            tmp_set_128i_0 = _mm_slli_epi16(tmp_set_128i_0, 8);
            tmp_set_128i_1 = _mm_slli_epi16(tmp_set_128i_1, 8);
            tmp_set_128i_0 = _mm_srli_epi16(tmp_set_128i_0, 8);
            tmp_set_128i_1 = _mm_srli_epi16(tmp_set_128i_1, 8);

            // combining even values
            tmp_set_128i_0 = _mm_packus_epi16(tmp_set_128i_0, tmp_set_128i_1);
            // combining odd values
            tmp_set_128i_1 = _mm_packus_epi16(tmp_set_128i_2, tmp_set_128i_3);

            // if the values >16 then put ff ,cmp_mask = dup16(15)
            cmp_store = _mm_cmpgt_epi8(tmp_set_128i_0, cmp_mask);
            // values 16 to 31 for row 0 & 1 but values <16 ==0
            tmp_set_128i_3 = _mm_and_si128(tmp_set_128i_0, cmp_store);
            // values 0 to 15 for row 0 & 1
            tmp_set_128i_2 =_mm_or_si128(tmp_set_128i_0, cmp_store);
            // values 16 to 31 for row 0 & 1 but values <16 masked to ff
            cmp_store = _mm_cmpeq_epi8(tmp_set_128i_3, const0_16x8b);
            tmp_set_128i_3 = _mm_or_si128(tmp_set_128i_3, cmp_store);

            // shuffle to get the band index of u
            tmp_set_128i_2 = _mm_shuffle_epi8(band_table0_8x16b_u, tmp_set_128i_2);
            tmp_set_128i_3 = _mm_shuffle_epi8(band_table1_8x16b_u, tmp_set_128i_3);

            // shuffle to get the band offset of u
            tmp_set_128i_2 = _mm_or_si128(tmp_set_128i_2, tmp_set_128i_3);
            tmp_set_128i_0 = _mm_shuffle_epi8(sao_offset_u, tmp_set_128i_2);

            // if the values >16 then put ff ,cmp_mask = dup16(15)
            cmp_store = _mm_cmpgt_epi8(tmp_set_128i_1, cmp_mask);
            // values 16 to 31 for row 0 & 1 but values <16 ==0
            tmp_set_128i_3 = _mm_and_si128(tmp_set_128i_1, cmp_store);
            // values 0 to 15 for row 0 & 1
            tmp_set_128i_2 =_mm_or_si128(tmp_set_128i_1, cmp_store);
            // values 16 to 31 for row 0 & 1 but values <16 masked to ff
            cmp_store = _mm_cmpeq_epi8(tmp_set_128i_3, const0_16x8b);
            tmp_set_128i_3 = _mm_or_si128(tmp_set_128i_3, cmp_store);

            // shuffle to get the band index of v
            tmp_set_128i_2 = _mm_shuffle_epi8(band_table0_8x16b_v, tmp_set_128i_2);
            tmp_set_128i_3 = _mm_shuffle_epi8(band_table1_8x16b_v, tmp_set_128i_3);

            // shuffle to get the band offset of v
            tmp_set_128i_2 = _mm_or_si128(tmp_set_128i_2, tmp_set_128i_3);
            tmp_set_128i_1 = _mm_shuffle_epi8(sao_offset_v, tmp_set_128i_2);

            // interleave u and v
            tmp_set_128i_2 = _mm_unpacklo_epi8(tmp_set_128i_0, tmp_set_128i_1);
            tmp_set_128i_3 = _mm_unpackhi_epi8(tmp_set_128i_0, tmp_set_128i_1);

            tmp_set_128i_5 =  _mm_cmpgt_epi8(const0_16x8b, tmp_set_128i_2);
            // SAO offset of row 0
            tmp_set_128i_0 = _mm_unpacklo_epi8(tmp_set_128i_2, tmp_set_128i_5);
            // SAO offset of row 1
            tmp_set_128i_1 = _mm_unpackhi_epi8(tmp_set_128i_2, tmp_set_128i_5);

            tmp_set_128i_6 =  _mm_cmpgt_epi8(const0_16x8b, tmp_set_128i_3);
            // SAO offset of row 2
            tmp_set_128i_2 = _mm_unpacklo_epi8(tmp_set_128i_3, tmp_set_128i_6);
            // SAO offset of row 3
            tmp_set_128i_3 = _mm_unpackhi_epi8(tmp_set_128i_3, tmp_set_128i_6);

            // add offset with input pixels
            src_temp0_8x16b = _mm_add_epi16(src_temp0_8x16b, tmp_set_128i_0);
            src_temp1_8x16b = _mm_add_epi16(src_temp1_8x16b, tmp_set_128i_1);
            src_temp2_8x16b = _mm_add_epi16(src_temp2_8x16b, tmp_set_128i_2);
            src_temp3_8x16b = _mm_add_epi16(src_temp3_8x16b, tmp_set_128i_3);

            // clip it to appropriate range
            src_temp0_8x16b = _mm_max_epi16(src_temp0_8x16b, m_min);
            src_temp1_8x16b = _mm_max_epi16(src_temp1_8x16b, m_min);
            src_temp2_8x16b = _mm_max_epi16(src_temp2_8x16b, m_min);
            src_temp3_8x16b = _mm_max_epi16(src_temp3_8x16b, m_min);

            src_temp0_8x16b = _mm_min_epi16(src_temp0_8x16b, m_max);
            src_temp1_8x16b = _mm_min_epi16(src_temp1_8x16b, m_max);
            src_temp2_8x16b = _mm_min_epi16(src_temp2_8x16b, m_max);
            src_temp3_8x16b = _mm_min_epi16(src_temp3_8x16b, m_max);

            // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
            _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_8x16b);
            // row = 1
            _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_8x16b);
            // row = 1
            _mm_storeu_si128((__m128i *)(pu2_src_cpy + 2 * src_strd), src_temp2_8x16b);
            // row = 1
            _mm_storeu_si128((__m128i *)(pu2_src_cpy + 3 * src_strd), src_temp3_8x16b);

            pu2_src_cpy += (src_strd << 2);
        }
        pu2_src += 8;
    }
}

void ihevc_hbd_sao_edge_offset_class0_sse42(UWORD16 *pu2_src,
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
    UWORD8 au1_mask[MAX_CTB_SIZE], *au1_mask_cpy;
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE+8];
    UWORD8 u1_avail0, u1_avail1;
    WORD32 offset=0 ;
    __m128i src_temp0_16x8b, src_temp1_16x8b;
    __m128i left0_16x8b, left1_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_gt1_16x8b, cmp_lt1_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i au1_mask8x16b;
    __m128i edge_idx_8x16b, sao_offset_8x16b;
    __m128i const2_16x8b, const0_16x8b, const1_16x8b;
    __m128i left_store_16x8b, left_store_16x8b_tmp, m_max, m_min;

    au1_mask8x16b = _mm_set1_epi8(0xff);
    m_max = _mm_srli_epi16(au1_mask8x16b, (16 - u4_bit_depth));
    m_min = _mm_setzero_si128();

    /* Update  top and top-left arrays */
    *pu2_src_top_left = pu2_src_top[wd - 1];

    for(col = wd; col >=8; col -= 8)
    {
        const0_16x8b= _mm_loadu_si128((__m128i *)(pu2_src + offset + (ht - 1) * src_strd));
        _mm_storeu_si128((__m128i *)(pu2_src_top + offset), const0_16x8b);
        offset += 8;
    }

    // setting availability mask to ff size MAX_CTB_SIZE
    for(col = 0 ; col < MAX_CTB_SIZE; col += 16)
        _mm_storeu_si128((__m128i *)(au1_mask+col), au1_mask8x16b);

    offset = 0;
    for(row = ht; row >= 8; row -= 8)
    {
        const0_16x8b= _mm_loadu_si128((__m128i *)(pu2_src_left + offset));
        _mm_storeu_si128((__m128i *)(au2_src_left_tmp + offset), const0_16x8b);
        offset += 8;
    }
    if(row) // This loop is entered if ht is multiple of 4 but not 8.
    {
        const0_16x8b= _mm_loadl_epi64((__m128i *)(pu2_src_left + offset));
        _mm_storel_epi64((__m128i *)(au2_src_left_tmp + offset), const0_16x8b);
    }

    edge_idx_8x16b   = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset);

    // availability mask creation
    u1_avail0 = pu1_avail[0];
    u1_avail1 = pu1_avail[1];
    au1_mask[0] = u1_avail0;
    au1_mask[wd-1] = u1_avail1;

    const2_16x8b = _mm_set1_epi8(2);
    const0_16x8b = _mm_setzero_si128();
    const1_16x8b = _mm_set1_epi8(1);

    {
        au1_mask_cpy = au1_mask;
        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            au1_mask8x16b = _mm_loadl_epi64((__m128i *)au1_mask_cpy);
            pu2_src_left_cpy =au2_src_left_tmp;
            for(row = ht; row > 0; row -= 4)
            {
                left_store_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src_left_cpy));
                // row = 0 load 8 pixel values from 7:0 pos. relative to cur. pos.
                src_temp0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy));
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+ src_strd));

                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, left_store_16x8b, 4);
                // row 1 left
                left1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, left_store_16x8b, 14);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);
                // row 0 left
                left0_16x8b = _mm_alignr_epi8(src_temp0_16x8b, left_store_16x8b, 14);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);

                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, left0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(left0_16x8b, src_temp0_16x8b);
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, left1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(left1_16x8b, src_temp1_16x8b);

                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                left0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // row = 0 right
                edge0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+1));
                // row = 1 right
                edge1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 1));

                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, edge0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(edge0_16x8b, src_temp0_16x8b);
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, edge1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(edge1_16x8b, src_temp1_16x8b);

                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // combining sign-left and sign_right
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, left0_16x8b);
                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // using availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                left0_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);

                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, left0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, left0_16x8b);

                // add offset with input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);

                // clip to appropriate range
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);

                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);
                left_store_16x8b_tmp = _mm_slli_si128(left_store_16x8b, 12);
                left_store_16x8b_tmp = _mm_srli_si128(left_store_16x8b_tmp, 12);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                pu2_src_cpy += (src_strd << 1);

                left_store_16x8b = _mm_srli_si128((left_store_16x8b), 4);
                // row = 0 load 8 pixel values from 7:0 pos. relative to cur. pos.
                src_temp0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy));
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+ src_strd));

                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, left_store_16x8b, 4);
                // row 1 left
                left1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, left_store_16x8b, 14);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);
                // row 0 left
                left0_16x8b = _mm_alignr_epi8(src_temp0_16x8b, left_store_16x8b, 14);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);

                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, left0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(left0_16x8b, src_temp0_16x8b);
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, left1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(left1_16x8b, src_temp1_16x8b);

                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                left0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // row = 0 right
                edge0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+1));
                // row = 1 right
                edge1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 1));

                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, edge0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(edge0_16x8b, src_temp0_16x8b);
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, edge1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(edge1_16x8b, src_temp1_16x8b);

                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // combining sign-left and sign_right
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, left0_16x8b);
                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // using availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                left0_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);

                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, left0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, left0_16x8b);

                // add offset with input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);

                // clip to appropriate range
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);

                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                left_store_16x8b = _mm_slli_si128(left_store_16x8b, 4);
                left_store_16x8b = _mm_or_si128(left_store_16x8b_tmp, left_store_16x8b);

                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                pu2_src_cpy += (src_strd << 1);
                pu2_src_left_cpy += 4;
            }

            if(row >= 2)
            {
                left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_left_cpy));
                // row = 0 load 8 pixel values from 7:0 pos. relative to cur. pos.
                src_temp0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy));
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+ src_strd));

                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, left_store_16x8b, 4);
                // row 1 left
                left1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, left_store_16x8b, 14);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);
                // row 0 left
                left0_16x8b = _mm_alignr_epi8(src_temp0_16x8b, left_store_16x8b, 14);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);

                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, left0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(left0_16x8b, src_temp0_16x8b);
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, left1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(left1_16x8b, src_temp1_16x8b);

                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                left0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // row = 0 right
                edge0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+1));
                // row = 1 right
                edge1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 1));

                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, edge0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(edge0_16x8b, src_temp0_16x8b);
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, edge1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(edge1_16x8b, src_temp1_16x8b);

                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // combining sign-left and sign_right
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, left0_16x8b);
                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // using availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                left0_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);

                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, left0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, left0_16x8b);

                // add offset with input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);

                // clip to appropriate range
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);

                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);
            }

            au1_mask_cpy += 8;
            pu2_src += 8;
        }

        offset = 0;
        for(row = ht; row >= 8; row -= 8)
        {
            const0_16x8b= _mm_loadu_si128((__m128i *)(au2_src_left_tmp + offset));
            _mm_storeu_si128((__m128i *)(pu2_src_left + offset), const0_16x8b);
            offset += 8;
        }
        if(row) // This loop is entered if ht is multiple of 4 but not 8.
        {
            const0_16x8b= _mm_loadl_epi64((__m128i *)(au2_src_left_tmp + offset));
            _mm_storel_epi64((__m128i *)(pu2_src_left + offset), const0_16x8b);
        }
    }
}
void ihevc_hbd_sao_edge_offset_class0_chroma_sse42(UWORD16 *pu2_src,
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
    UWORD8 au1_mask[2 * MAX_CTB_SIZE], *au1_mask_cpy;
    UWORD8 u1_avail0, u1_avail1;
    WORD32 offset=0;

    __m128i src_temp0_16x8b, src_temp1_16x8b;
    __m128i left0_16x8b, left1_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_gt1_16x8b, cmp_lt1_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i au1_mask8x16b, m_max, m_min;
    __m128i edge_idx_8x16b, sao_offset_u_8x16b, sao_offset_v_8x16b;
    __m128i const2_16x8b, const0_16x8b, const1_16x8b;
    __m128i left_store_16x8b, chroma_offset_8x16b, sao_offset_8x16b;

    au1_mask8x16b = _mm_set1_epi8(0xff);

    m_max = _mm_srli_epi16(au1_mask8x16b, (16 - u4_bit_depth));
    m_min = _mm_setzero_si128();

    /* Update  top and top-left arrays */
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];

    for(col = wd; col >=8; col -= 8)
    {
        const0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + offset + (ht - 1) * src_strd));
        _mm_storeu_si128((__m128i *)(pu2_src_top + offset), const0_16x8b);
        offset += 8;
    }

    // setting availability mask to ff size 2 * MAX_CTB_SIZE
    for(col = 0 ; col < 2 * MAX_CTB_SIZE; col += 16)
        _mm_storeu_si128((__m128i *)(au1_mask+col), au1_mask8x16b);

    edge_idx_8x16b   = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_u_8x16b = _mm_loadu_si128((__m128i *)pi1_sao_offset_u);
    sao_offset_v_8x16b = _mm_loadu_si128((__m128i *)pi1_sao_offset_v);
    sao_offset_8x16b = _mm_unpacklo_epi64(sao_offset_u_8x16b, sao_offset_v_8x16b);
    chroma_offset_8x16b = _mm_set1_epi16(0x0800);

    // availability mask creation
    u1_avail0 = pu1_avail[0];
    u1_avail1 = pu1_avail[1];
    au1_mask[0] = u1_avail0;
    au1_mask[1] = u1_avail0;
    au1_mask[wd-1] = u1_avail1;
    au1_mask[wd-2] = u1_avail1;
    const2_16x8b = _mm_set1_epi8(2);
    const0_16x8b = _mm_setzero_si128();
    const1_16x8b = _mm_set1_epi8(1);

    {
        au1_mask_cpy = au1_mask;
        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            au1_mask8x16b = _mm_loadl_epi64((__m128i *)au1_mask_cpy);
            pu2_src_left_cpy = pu2_src_left;
            for(row = ht; row > 0; row -= 2)
            {
                left_store_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src_left_cpy));
                // row = 0 load 8 pixel values from 7:0 pos. relative to cur. pos.
                src_temp0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy));
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));

                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, left_store_16x8b, 8);
                // row 1 left
                left1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, left_store_16x8b, 12);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 12);
                // row 0 left
                left0_16x8b = _mm_alignr_epi8(src_temp0_16x8b, left_store_16x8b, 12);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 12);

                // separating +ve and and -ve values.row 0 left
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, left0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(left0_16x8b, src_temp0_16x8b);

                // separating +ve and and -ve values.row 1 left
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, left1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(left1_16x8b, src_temp1_16x8b);

                // packing the signs of row1 and row0 into 8 bit number
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                left0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // row = 0 right
                edge0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2));
                // row = 1 right
                edge1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 2));
                // separating +ve and and -ve values.row 0 right
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, edge0_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(edge0_16x8b, src_temp0_16x8b);


                // separating +ve and and -ve values.row 1 right
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, edge1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(edge1_16x8b, src_temp1_16x8b);

                // packing the signs of row1 and row0 into 8 bit number
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // combining sign-left and sign_right
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, left0_16x8b);
                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get SAO index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // using availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // get SAO offset
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, chroma_offset_8x16b);
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                left0_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);

                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, left0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, left0_16x8b);
                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);

                // clipping to appropriate value
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);

                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                pu2_src_cpy += (src_strd << 1);
                pu2_src_left_cpy += 4;
            }
            au1_mask_cpy += 8;
            pu2_src += 8;
        }
    }
}

void ihevc_hbd_sao_edge_offset_class1_sse42(UWORD16 *pu2_src,
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
    UWORD16 *pu2_src_top_cpy;
    UWORD16 *pu2_src_cpy;

    __m128i src_top_16x8b, src_bottom_16x8b;
    __m128i src_temp0_16x8b, src_temp1_16x8b;
    __m128i signup0_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_gt1_16x8b, cmp_lt1_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i edge_idx_8x16b, sao_offset_8x16b;
    __m128i const2_16x8b, const0_16x8b, const1_16x8b;
    __m128i m_max, m_min, m_count;

    /* Updating left and top-left */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[row] = pu2_src[row * src_strd + (wd - 1)];
    }
    *pu2_src_top_left = pu2_src_top[wd - 1];

    pu2_src_top_cpy = pu2_src_top;
    edge_idx_8x16b   = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset);

    m_max =  _mm_cmpeq_epi16(edge_idx_8x16b, edge_idx_8x16b);
    m_count = _mm_cvtsi32_si128((16 - u4_bit_depth));
    m_max = _mm_srl_epi16(m_max, m_count);
    m_min = _mm_setzero_si128();

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht--;
    }
    if(0 == pu1_avail[3])
    {
        ht--;
    }

    const2_16x8b = _mm_set1_epi8(2);
    const1_16x8b =  _mm_set1_epi8(1);
    const0_16x8b = _mm_setzero_si128();

    {
        WORD32 ht_rem;
        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            src_top_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy + wd - col));
            // row = 0
            src_temp0_16x8b =_mm_loadu_si128((__m128i *)(pu2_src_cpy));
            // separating +ve and and -ve values.
            cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_top_16x8b);
            cmp_lt0_16x8b = _mm_subs_epu16(src_top_16x8b, src_temp0_16x8b);
            // creating mask 00 for +ve and -ve values and FF for zero.
            cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
            cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
            // combining the appropriate sign change
            signup0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
            signup0_16x8b = _mm_packs_epi16(signup0_16x8b, signup0_16x8b);
            for(row = ht; row >=2; row -= 2)
            {
                // row = 1 load 8 pixel values from 7:0 pos. relative to cur. pos.
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd));

                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_temp1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_temp0_16x8b);

                // row1 -bottom
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);

                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt1_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt1_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);

                cmp_gt0_16x8b = _mm_sub_epi8(cmp_gt1_16x8b, cmp_lt1_16x8b);
                cmp_lt0_16x8b = _mm_sub_epi8(cmp_lt1_16x8b, cmp_gt1_16x8b);

                edge0_16x8b = _mm_unpacklo_epi64(cmp_gt0_16x8b, cmp_lt0_16x8b);
                signup0_16x8b = _mm_unpackhi_epi64(signup0_16x8b, cmp_gt0_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                signup0_16x8b = _mm_unpackhi_epi64(cmp_lt0_16x8b, cmp_lt0_16x8b);

                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                src_top_16x8b = src_temp1_16x8b;

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);

                // clipping to appropriate value
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);

                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd << 1);
            }
            ht_rem = ht&0x1;

            if(ht_rem)
            {
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // current row -next row
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
                // combining the appropriate sign change
                edge0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                edge0_16x8b = _mm_packs_epi16(edge0_16x8b, edge0_16x8b);
                // adding top and botton and constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                src_top_16x8b = src_temp0_16x8b;

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);

                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
            }
            if(0 == pu1_avail[3])
            {
                src_top_16x8b = src_bottom_16x8b;
            }
            // updating top flag
            _mm_storeu_si128((__m128i *)(pu2_src_top + wd - col), src_top_16x8b);
            pu2_src += 8;
        }
    }
}

void ihevc_hbd_sao_edge_offset_class1_chroma_sse42(UWORD16 *pu2_src,
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
    UWORD16 *pu2_src_top_cpy;
    UWORD16 *pu2_src_cpy;

    __m128i src_top_16x8b, src_bottom_16x8b;
    __m128i src_temp0_16x8b, src_temp1_16x8b;
    __m128i signup0_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_gt1_16x8b, cmp_lt1_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i edge_idx_8x16b;
    __m128i const2_16x8b, const0_16x8b, const1_16x8b;
    __m128i sao_offset_u_8x16b, sao_offset_v_8x16b;
    __m128i m_min, m_max, m_count, sao_offset_8x16b, chroma_offset_8x16b;

    /* Updating left and top and top-left */
    for(row = 0; row < ht; row++)
    {
        pu2_src_left[2 * row] = pu2_src[row * src_strd + (wd - 2)];
        pu2_src_left[2 * row + 1] = pu2_src[row * src_strd + (wd - 1)];
    }
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];

    pu2_src_top_cpy = pu2_src_top;
    edge_idx_8x16b   = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_u_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset_u);
    sao_offset_v_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset_v);
    sao_offset_8x16b = _mm_unpacklo_epi64(sao_offset_u_8x16b, sao_offset_v_8x16b);
    chroma_offset_8x16b = _mm_set1_epi16(0x0800);

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht--;
    }
    if(0 == pu1_avail[3])
    {
        ht--;
    }

    m_max =  _mm_cmpeq_epi16(edge_idx_8x16b, edge_idx_8x16b);
    m_count = _mm_cvtsi32_si128((16 - u4_bit_depth));
    m_max = _mm_srl_epi16(m_max, m_count);
    m_min = _mm_setzero_si128();
    const2_16x8b = _mm_set1_epi8(2);
    const0_16x8b = _mm_setzero_si128();
    const1_16x8b = _mm_set1_epi8(1);

    {
        WORD32 ht_rem;

        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            src_top_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy + wd - col));
            // row = 0
            src_temp0_16x8b =_mm_loadu_si128((__m128i *)(pu2_src_cpy));
            // separating +ve and and -ve values.
            cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_top_16x8b);
            cmp_lt0_16x8b = _mm_subs_epu16(src_top_16x8b, src_temp0_16x8b);
            // creating mask 00 for +ve and -ve values and FF for zero.
            cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
            cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
            // combining the appropriate sign change
            signup0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
            signup0_16x8b = _mm_packs_epi16(signup0_16x8b, signup0_16x8b);

            for(row = ht; row >=2; row -= 2)
            {
                // row = 1 load 8 pixel values from 7:0 pos. relative to cur. pos.
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2 * src_strd));

                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_temp1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_temp0_16x8b);

                // row1 -bottom
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt1_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt1_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);

                cmp_gt0_16x8b = _mm_sub_epi8(cmp_gt1_16x8b, cmp_lt1_16x8b);
                cmp_lt0_16x8b = _mm_sub_epi8(cmp_lt1_16x8b, cmp_gt1_16x8b);

                edge0_16x8b = _mm_unpacklo_epi64(cmp_gt0_16x8b, cmp_lt0_16x8b);
                signup0_16x8b = _mm_unpackhi_epi64(signup0_16x8b, cmp_gt0_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                signup0_16x8b = _mm_unpackhi_epi64(cmp_lt0_16x8b, cmp_lt0_16x8b);

                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, chroma_offset_8x16b);
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);

                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);
            }
            ht_rem = ht&0x1;

            if(ht_rem)
            {
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // current row -next row
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
                // combining the appropriate sign change
                edge0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                edge0_16x8b = _mm_packs_epi16(edge0_16x8b, edge0_16x8b);
                // adding top and botton and constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // copying the next top
                src_top_16x8b = src_temp0_16x8b;

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, chroma_offset_8x16b);
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);

                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
            }
            if(0 == pu1_avail[3])
            {
                src_top_16x8b = src_bottom_16x8b;
            }
            // updating top flag
            _mm_storeu_si128((__m128i *)(pu2_src_top + wd - col), src_top_16x8b);
            pu2_src += 8;
        }
    }
}

/* 135 degree filtering */
void ihevc_hbd_sao_edge_offset_class2_sse42(UWORD16 *pu2_src,
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
    UWORD16 *pu2_src_top_cpy, *pu2_src_left_cpy, *pu2_src_left_cpy2;
    UWORD16 *pu2_firstleft;
    UWORD16 *pu2_src_cpy, *pu2_src_org;
    UWORD8 au1_mask[MAX_CTB_SIZE], *au1_mask_cpy;
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE+8];
    UWORD16 u2_pos_0_0_tmp, u2_pos_wd_ht_tmp;
    WORD32 ht_tmp, offset;

    WORD32 bit_depth;
    UWORD8 u1_avail0, u1_avail1;

    __m128i src_top_16x8b, src_bottom_16x8b;
    __m128i src_temp0_16x8b, src_temp1_16x8b;
    __m128i signup0_16x8b, signdwn1_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_lt1_16x8b, cmp_gt1_16x8b;
    __m128i cmp_gt0_u_16x8b, cmp_lt0_u_16x8b, cmp_lt1_b_16x8b, cmp_gt1_b_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i au1_mask8x16b;
    __m128i edge_idx_8x16b, sao_offset_8x16b;
    __m128i const2_16x8b, const0_16x8b, const1_16x8b;
    __m128i left_store_16x8b, left_store_16x8b_tmp;
    __m128i m_max, m_min, m_count;

    ht_tmp = ht;
    au1_mask8x16b = _mm_set1_epi8(0xff);

    // setting availability mask to ff size MAX_CTB_SIZE
    for(col = 0 ; col < MAX_CTB_SIZE; col += 16)
        _mm_storeu_si128((__m128i *)(au1_mask+col), au1_mask8x16b);

    offset = 0;
    for(row = ht; row >= 8; row -= 8)
    {
        const0_16x8b= _mm_loadu_si128((__m128i *)(pu2_src_left + offset));
        _mm_storeu_si128((__m128i *)(au2_src_left_tmp + offset), const0_16x8b);
        offset += 8;
    }
    if(row) // This loop is entered if ht is multiple of 4 but not 8.
    {
        const0_16x8b= _mm_loadl_epi64((__m128i *)(pu2_src_left + offset));
        _mm_storel_epi64((__m128i *)(au2_src_left_tmp + offset), const0_16x8b);
    }

    bit_depth = u4_bit_depth;
    pu2_src_org = pu2_src;
    pu2_src_top_cpy = pu2_src_top;
    pu2_src_left_cpy2 = au2_src_left_tmp;
    pu2_src_left_cpy = au2_src_left_tmp;
    edge_idx_8x16b   = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset);

    /* If top-left is available, process separately */
    if(0 != pu1_avail[4])
    {
        WORD8 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[0] - pu2_src_top_left[0]) +
            SIGN(pu2_src[0] - pu2_src[1 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_0_tmp = CLIP3(pu2_src[0] + pi1_sao_offset[edge_idx], 0, (1 << bit_depth) - 1);
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
        WORD8 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[wd - 1 + (ht - 1) * src_strd] - pu2_src[wd - 1 + (ht - 1) * src_strd- 1 - src_strd]) +
            SIGN(pu2_src[wd - 1 + (ht - 1) * src_strd] - pu2_src[wd - 1 + (ht - 1) * src_strd + 1 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_ht_tmp = CLIP3(pu2_src[wd - 1 + (ht - 1) * src_strd] + pi1_sao_offset[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_ht_tmp = pu2_src[wd - 1 + (ht - 1) * src_strd];
        }
    }
    else
    {
        u2_pos_wd_ht_tmp = pu2_src[wd - 1 + (ht - 1) * src_strd];
    }
    pu2_firstleft = pu2_src_top_left;

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_firstleft = pu2_src_left_cpy2;
        pu2_src_left_cpy2++;
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht--;
    }
    if(0 == pu1_avail[3])
    {
        ht--;
    }
    // storing top left in a mmx register
    left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_firstleft);
    m_max =  _mm_cmpeq_epi16(edge_idx_8x16b, edge_idx_8x16b);
    m_count = _mm_cvtsi32_si128((16 - u4_bit_depth));
    m_max = _mm_srl_epi16(m_max, m_count);
    m_min = _mm_setzero_si128();

    const2_16x8b = _mm_set1_epi8(2);
    const0_16x8b = _mm_setzero_si128();
    const1_16x8b = _mm_set1_epi8(1);
    left_store_16x8b = _mm_slli_si128(left_store_16x8b, 14);
    // update top -left
    *pu2_src_top_left = pu2_src_top[wd - 1];
    // availability mask creation
    u1_avail0 = pu1_avail[0];
    u1_avail1 = pu1_avail[1];
    au1_mask[0] = u1_avail0;
    au1_mask[wd-1] = u1_avail1;
    {
        WORD32 ht_rem;

        au1_mask_cpy = au1_mask;
        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            src_top_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy + wd-col));
            // row = 0
            src_temp0_16x8b =_mm_loadu_si128((__m128i *)(pu2_src_cpy));
            src_top_16x8b = _mm_alignr_epi8(src_top_16x8b, left_store_16x8b, 14);
            // loading the mask
            au1_mask8x16b = _mm_loadu_si128((__m128i *)au1_mask_cpy);
            // separating +ve and and -ve values.
            cmp_gt0_u_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_top_16x8b);
            cmp_lt0_u_16x8b = _mm_subs_epu16(src_top_16x8b, src_temp0_16x8b);
            // creating mask 00 for +ve and -ve values and FF for zero.

            pu2_src_left_cpy = pu2_src_left_cpy2;

            for(row = ht; row >=4; row -= 4)
            {
                left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_src_left_cpy);
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // row = 1 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 1));
                // to insert left in row 0
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 14);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp0_16x8b, signdwn1_16x8b, 14);
                // combining the appropriate sign change
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                // row = 2 right
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd + 1));

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.

                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd));

                // combining sign-left and sign_right

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                // storing the row 1 left for next row.
                signup0_16x8b = _mm_slli_si128(left_store_16x8b, 12);

                // combining sign-left and sign_right
                // manipulation for bottom - row 1
                signup0_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signup0_16x8b, 14);
                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signup0_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signup0_16x8b, src_bottom_16x8b);

                // row1  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);
                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // use availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // cnvert to 16 bit then add and then saturated pack
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                // store left boundary
                left_store_16x8b_tmp = _mm_slli_si128(left_store_16x8b, 12);
                left_store_16x8b_tmp = _mm_srli_si128(left_store_16x8b_tmp, 12);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);

                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // row = 1 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 1));
                // to insert left in row 0
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 14);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp0_16x8b, signdwn1_16x8b, 14);
                // combining the appropriate sign change
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                // row = 2 right
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd + 1));

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.

                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd));

                // combining sign-left and sign_right

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                // storing the row 1 left for next row.
                signup0_16x8b = _mm_slli_si128(left_store_16x8b, 12);

                // combining sign-left and sign_right
                // manipulation for bottom - row 1
                signup0_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signup0_16x8b, 14);
                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signup0_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signup0_16x8b, src_bottom_16x8b);

                // row1  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);
                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // use availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // cnvert to 16 bit then add and then saturated pack
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                left_store_16x8b = _mm_slli_si128(left_store_16x8b, 4);
                left_store_16x8b = _mm_or_si128(left_store_16x8b_tmp, left_store_16x8b);
                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);
                pu2_src_left_cpy += 4;
            }
            if(row >= 2)
            {
                left_store_16x8b = _mm_loadu_si128((__m128i *)pu2_src_left_cpy);
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // row = 1 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 1));
                // to insert left in row 0
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 14);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp0_16x8b, signdwn1_16x8b, 14);
                // combining the appropriate sign change
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                // row = 2 right
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd + 1));

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.

                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd));

                // combining sign-left and sign_right

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                // storing the row 1 left for next row.
                signup0_16x8b = _mm_slli_si128(left_store_16x8b, 12);

                // combining sign-left and sign_right
                // manipulation for bottom - row 1
                signup0_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signup0_16x8b, 14);
                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signup0_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signup0_16x8b, src_bottom_16x8b);

                // row1  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);
                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // use availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // cnvert to 16 bit then add and then saturated pack
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);
                pu2_src_left_cpy += 2;
            }

            ht_rem = ht&0x1;

            if(ht_rem)
            {
                left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_src_left_cpy);
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 1));
                // current row -next row
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
                // combining the appropriate sign change
                edge0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                edge0_16x8b = _mm_packs_epi16(edge0_16x8b, edge0_16x8b);
                // adding top and botton and constant 2
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_u_16x8b, const0_16x8b);
                // for the next iteration bottom -row1
                signup0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                signup0_16x8b = _mm_packs_epi16(signup0_16x8b, signup0_16x8b);
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 2);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // use availability mask
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp0_16x8b;

                // add offset + pixels and clip
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);

                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                pu2_src_cpy += (src_strd);
            }
            if(0 == pu1_avail[3])
            {
                src_top_16x8b = src_bottom_16x8b;
                au2_src_left_tmp[ht_tmp-1] = pu2_src_cpy[7];
            }
            if(0 == pu1_avail[2])
            {
                au2_src_left_tmp[0] = pu2_src[7 - src_strd];
            }

            // for the top left of next part of the block
            left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy + wd - col));
            // updating top flag
            _mm_storeu_si128((__m128i *)(pu2_src_top+ wd - col), src_top_16x8b);
            pu2_src += 8;
            au1_mask_cpy += 8;
        }

        pu2_src_org[0] = u2_pos_0_0_tmp;
        pu2_src_org[wd - 1 + (ht_tmp - 1) * src_strd] = u2_pos_wd_ht_tmp;

        offset = 0;
        for(row = ht_tmp; row >= 8; row -= 8)
        {
            const0_16x8b= _mm_loadu_si128((__m128i *)(au2_src_left_tmp + offset));
            _mm_storeu_si128((__m128i *)(pu2_src_left + offset), const0_16x8b);
            offset += 8;
        }
        if(row) // This loop is entered if ht_tmp is multiple of 4 but not 8.
        {
            const0_16x8b= _mm_loadl_epi64((__m128i *)(au2_src_left_tmp + offset));
            _mm_storel_epi64((__m128i *)(pu2_src_left + offset), const0_16x8b);
        }
    }
}

void ihevc_hbd_sao_edge_offset_class2_chroma_sse42(UWORD16 *pu2_src,
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
    UWORD16 *pu2_src_top_cpy, *pu2_src_left_cpy, *pu2_src_left_cpy2;
    UWORD16 *pu2_firstleft;
    UWORD16 *pu2_src_cpy, *pu2_src_org;
    UWORD8 au1_mask[2 * MAX_CTB_SIZE], *au1_mask_cpy;
    UWORD16 au2_src_left_tmp[2*(MAX_CTB_SIZE+8)];
    UWORD16 u2_pos_0_0_tmp_u, u2_pos_0_0_tmp_v, u2_pos_wd_ht_tmp_u, u2_pos_wd_ht_tmp_v;
    WORD32 ht_tmp;

    WORD32 bit_depth, offset;
    UWORD8 u1_avail0, u1_avail1;

    __m128i    src_temp0_16x8b, src_temp1_16x8b;
    __m128i signup0_16x8b, signdwn1_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_lt1_16x8b, cmp_gt1_16x8b;
    __m128i cmp_gt0_u_16x8b, cmp_lt0_u_16x8b, cmp_lt1_b_16x8b, cmp_gt1_b_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i src_top_16x8b, src_bottom_16x8b;
    __m128i au1_mask8x16b;
    __m128i edge_idx_8x16b, m_max, m_min, m_count;
    __m128i const2_16x8b, const0_16x8b, const1_16x8b;
    __m128i left_store_16x8b;
    __m128i chroma_offset_8x16b, sao_offset_u_8x16b, sao_offset_v_8x16b, sao_offset_8x16b;
    ht_tmp = ht;
    au1_mask8x16b = _mm_set1_epi8(0xff);
    /* Updating left and top-left */

    offset = 0;
    for(row = 2*ht; row >= 8; row -= 8)
    {
        const0_16x8b= _mm_loadu_si128((__m128i *)(pu2_src_left + offset));
        _mm_storeu_si128((__m128i *)(au2_src_left_tmp + offset), const0_16x8b);
        offset += 8;
    }
    // setting availability mask to ff size 2 * MAX_CTB_SIZE
    for(col = 0 ; col < 2 * MAX_CTB_SIZE; col += 16)
        _mm_storeu_si128((__m128i *)(au1_mask+col), au1_mask8x16b);
    bit_depth = u4_bit_depth;
    pu2_src_org = pu2_src;
    pu2_src_top_cpy = pu2_src_top;
    pu2_src_left_cpy2 = au2_src_left_tmp;
    pu2_src_left_cpy = au2_src_left_tmp;
    edge_idx_8x16b  = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_u_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset_u);
    sao_offset_v_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset_v);
    sao_offset_8x16b = _mm_unpacklo_epi64(sao_offset_u_8x16b, sao_offset_v_8x16b);
    chroma_offset_8x16b = _mm_set1_epi16(0x0800);

    /* If top-left is available, process separately */
    if(0 != pu1_avail[4])
    {
        WORD32 edge_idx;

        /* U */
        edge_idx = 2 + SIGN(pu2_src[0] - pu2_src_top_left[0]) +
            SIGN(pu2_src[0] - pu2_src[2 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_0_tmp_u = CLIP3(pu2_src[0] + pi1_sao_offset_u[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_0_0_tmp_u = pu2_src[0];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[1] - pu2_src_top_left[1]) +
            SIGN(pu2_src[1] - pu2_src[1 + 2 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_0_tmp_v = CLIP3(pu2_src[1] + pi1_sao_offset_v[edge_idx], 0, (1 << bit_depth) - 1);
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
        edge_idx = 2 + SIGN(pu2_src[wd - 2 + (ht - 1) * src_strd] - pu2_src[wd - 2 + (ht - 1) * src_strd - 2 - src_strd]) +
            SIGN(pu2_src[wd - 2 + (ht - 1) * src_strd] - pu2_src[wd - 2 + (ht - 1) * src_strd + 2 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_ht_tmp_u = CLIP3(pu2_src[wd - 2 + (ht - 1) * src_strd] + pi1_sao_offset_u[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_ht_tmp_u = pu2_src[wd - 2 + (ht - 1) * src_strd];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[wd - 1 + (ht - 1) * src_strd] - pu2_src[wd - 1 + (ht - 1) * src_strd - 2 - src_strd]) +
            SIGN(pu2_src[wd - 1 + (ht - 1) * src_strd] - pu2_src[wd - 1 + (ht - 1) * src_strd + 2 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_ht_tmp_v = CLIP3(pu2_src[wd - 1 + (ht - 1) * src_strd] + pi1_sao_offset_v[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_ht_tmp_v = pu2_src[wd - 1 + (ht - 1) * src_strd];
        }
    }
    else
    {
        u2_pos_wd_ht_tmp_u = pu2_src[wd - 2 + (ht - 1) * src_strd];
        u2_pos_wd_ht_tmp_v = pu2_src[wd - 1 + (ht - 1) * src_strd];
    }
    pu2_firstleft = pu2_src_top_left;

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_firstleft = pu2_src_left_cpy2;
        pu2_src_left_cpy2 += 2;
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht--;
    }
    if(0 == pu1_avail[3])
    {
        ht--;
    }
    // storing top left in a mmx register
    left_store_16x8b = _mm_loadu_si128((__m128i *)pu2_firstleft);

    m_max =  _mm_cmpeq_epi16(edge_idx_8x16b, edge_idx_8x16b);
    m_count = _mm_cvtsi32_si128((16 - u4_bit_depth));
    m_max = _mm_srl_epi16(m_max, m_count);
    m_min = _mm_setzero_si128();
    const2_16x8b = _mm_set1_epi8(2);
    const0_16x8b = _mm_setzero_si128();
    const1_16x8b = _mm_set1_epi8(1);

    left_store_16x8b = _mm_slli_si128(left_store_16x8b, 12);

    // availability mask creation
    u1_avail0 = pu1_avail[0];
    u1_avail1 = pu1_avail[1];
    au1_mask[0] = u1_avail0;
    au1_mask[1] = u1_avail0;
    au1_mask[wd-1] = u1_avail1;
    au1_mask[wd-2] = u1_avail1;

    /* top-left arrays */
    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];
    {
        WORD32 ht_rem;
        au1_mask_cpy = au1_mask;
        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            src_top_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy + wd - col));
            // row = 0
            src_temp0_16x8b =_mm_loadu_si128((__m128i *)(pu2_src_cpy));
            src_top_16x8b = _mm_alignr_epi8(src_top_16x8b, left_store_16x8b, 12);
            // loading the mask
            au1_mask8x16b = _mm_loadu_si128((__m128i *)au1_mask_cpy);
            // separating +ve and and -ve values.
            cmp_gt0_u_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_top_16x8b);
            cmp_lt0_u_16x8b = _mm_subs_epu16(src_top_16x8b, src_temp0_16x8b);

            pu2_src_left_cpy =pu2_src_left_cpy2;

            for(row = ht; row >=2; row -= 2)
            {
                left_store_16x8b = _mm_loadu_si128((__m128i *)pu2_src_left_cpy);
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd));
                // row = 1 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 2));
                // to insert left in row 0
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 12);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp0_16x8b, signdwn1_16x8b, 12);

                // separating +ve and and -ve values.
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                // row = 2 right
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd + 2));

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);

                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + 2*src_strd));

                // storing the row 1 left for next row.
                signup0_16x8b = _mm_slli_si128(left_store_16x8b, 8);

                // manipulation for bottom - row 1
                signup0_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signup0_16x8b, 12);
                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 8);
                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signup0_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signup0_16x8b, src_bottom_16x8b);

                // row1  getting it right for left of next iteration
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 12);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;
                // row0  getting its right for left of next iteration.
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 12);

                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // use availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // adding chroma offset to access U and V
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, chroma_offset_8x16b);
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // cnvert to 16 bit then add and then saturated pack
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);

                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy + src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);
                pu2_src_left_cpy += 4;
            }
            ht_rem = ht&0x1;

            if(ht_rem)
            {
                left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_src_left_cpy);
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy + src_strd + 2));
                // current row -next row
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_bottom_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp0_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
                // combining the appropriate sign change
                edge0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                edge0_16x8b = _mm_packs_epi16(edge0_16x8b, edge0_16x8b);
                // adding top and botton and constant 2
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_u_16x8b, const0_16x8b);
                // for the next iteration bottom -row1
                signup0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                signup0_16x8b = _mm_packs_epi16(signup0_16x8b, signup0_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);

                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                // copying the next top
                src_top_16x8b = src_temp0_16x8b;
                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 12);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // using availability mask
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // adding chroma offset to access U and V
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, chroma_offset_8x16b);
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);

                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // SAO offset + input pixels
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);

                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                pu2_src_cpy += (src_strd);
                pu2_src_left_cpy += 2;
            }
            if(0 == pu1_avail[3])
            {
                src_top_16x8b = src_bottom_16x8b;
                au2_src_left_tmp[2*ht_tmp - 1]=pu2_src_cpy[7];
                au2_src_left_tmp[2*ht_tmp - 2]=pu2_src_cpy[6];
            }
            if(0 == pu1_avail[2])
            {
                au2_src_left_tmp[0]=pu2_src[6 - src_strd];
                au2_src_left_tmp[1]=pu2_src[7 - src_strd];
            }

            // for the top left of next part of the block
            left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy + wd-col));
            // updating top flag
            _mm_storeu_si128((__m128i *)(pu2_src_top + wd-col), src_top_16x8b);
            pu2_src += 8;
            au1_mask_cpy += 8;
        }

        pu2_src_org[0] = u2_pos_0_0_tmp_u;
        pu2_src_org[1] = u2_pos_0_0_tmp_v;
        pu2_src_org[wd - 2 + (ht_tmp - 1) * src_strd] = u2_pos_wd_ht_tmp_u;
        pu2_src_org[wd - 1 + (ht_tmp - 1) * src_strd] = u2_pos_wd_ht_tmp_v;

        offset = 0;
        for(row = 2*ht_tmp; row >= 8; row -= 8)
        {
            const0_16x8b= _mm_loadu_si128((__m128i *)(au2_src_left_tmp + offset));
            _mm_storeu_si128((__m128i *)(pu2_src_left + offset), const0_16x8b);
            offset += 8;
        }
    }
}

void ihevc_hbd_sao_edge_offset_class3_sse42(UWORD16 *pu2_src,
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
    UWORD16 *pu2_src_top_cpy, *pu2_src_left_cpy, *pu2_src_left_cpy2;
    UWORD16 *pu2_src_cpy, *pu2_src_org;
    UWORD16 au2_src_left_tmp[MAX_CTB_SIZE+8];
    UWORD8 au1_mask[MAX_CTB_SIZE], *au1_mask_cpy;
    UWORD16 u2_pos_wd_0_tmp, u2_pos_0_ht_tmp;
    WORD32 ht_tmp;
    WORD32 bit_depth;
    UWORD8 u1_avail0, u1_avail1;

    __m128i src_top_16x8b, src_bottom_16x8b;
    __m128i src_temp0_16x8b, src_temp1_16x8b;
    __m128i signup0_16x8b, signdwn1_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_gt1_16x8b, cmp_lt1_16x8b;
    __m128i cmp_gt0_u_16x8b, cmp_lt0_u_16x8b, cmp_gt1_b_16x8b, cmp_lt1_b_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i au1_mask8x16b;
    __m128i edge_idx_8x16b, sao_offset_8x16b;
    __m128i const2_16x8b, const0_16x8b, const1_16x8b;
    __m128i left_store_16x8b, left_store_16x8b_tmp, m_max, m_min, m_count;

    ht_tmp = ht;
    au1_mask8x16b = _mm_set1_epi8(0xff);

    au2_src_left_tmp[0] =pu2_src[(wd - 1)];
    // manipulation for bottom left
    for(row = 1; row < ht; row++)
    {
        au2_src_left_tmp[row] =pu2_src_left[row];
    }
    au2_src_left_tmp[ht] = pu2_src_bot_left[0];

    *pu2_src_top_left = pu2_src_top[wd - 1];
    // setting availability mask to ff size MAX_CTB_SIZE
    for(col = 0 ; col < MAX_CTB_SIZE; col += 16)
        _mm_storeu_si128((__m128i *)(au1_mask+col), au1_mask8x16b);
    bit_depth = u4_bit_depth;
    pu2_src_org = pu2_src;
    pu2_src_top_cpy = pu2_src_top;
    pu2_src_left_cpy2 = au2_src_left_tmp;
    pu2_src_left_cpy = au2_src_left_tmp;
    edge_idx_8x16b   = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset);

    /* If top-right is available, process separately */
    if(0 != pu1_avail[5])
    {
        WORD32 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[wd - 1] - pu2_src_top_right[0]) +
            SIGN(pu2_src[wd - 1] - pu2_src[wd - 1 - 1 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_0_tmp = CLIP3(pu2_src[wd - 1] + pi1_sao_offset[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_0_tmp = pu2_src[wd - 1];
        }
    }
    else
    {
        u2_pos_wd_0_tmp = pu2_src[wd - 1];
    }

    /* If bottom-left is available, process separately */
    if(0 != pu1_avail[6])
    {
        WORD32 edge_idx;

        edge_idx = 2 + SIGN(pu2_src[(ht - 1) * src_strd] - pu2_src[(ht - 1) * src_strd + 1 - src_strd]) +
            SIGN(pu2_src[(ht - 1) * src_strd] - pu2_src_bot_left[0]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_ht_tmp = CLIP3(pu2_src[(ht - 1) * src_strd] + pi1_sao_offset[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_0_ht_tmp = pu2_src[(ht - 1) * src_strd];
        }
    }
    else
    {
        u2_pos_0_ht_tmp = pu2_src[(ht - 1) * src_strd];
    }

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_src_left_cpy2++;
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht--;
    }
    if(0 == pu1_avail[3])
    {
        ht--;
    }

    m_max =  _mm_cmpeq_epi16(edge_idx_8x16b, edge_idx_8x16b);
    m_count = _mm_cvtsi32_si128((16 - u4_bit_depth));
    m_max = _mm_srl_epi16(m_max, m_count);
    m_min = _mm_setzero_si128();

    const2_16x8b = _mm_set1_epi8(2);
    const0_16x8b = _mm_setzero_si128();
    const1_16x8b = _mm_set1_epi8(1);

    // availability mask creation
    u1_avail0 = pu1_avail[0];
    u1_avail1 = pu1_avail[1];
    au1_mask[0] = u1_avail0;
    au1_mask[wd-1] = u1_avail1;
    {
        WORD32 ht_rem;

        au1_mask_cpy = au1_mask;
        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            src_top_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy+wd-col+1));
            // row = 0
            src_temp0_16x8b =_mm_loadu_si128((__m128i *)(pu2_src_cpy));

            // loading the mask
            au1_mask8x16b = _mm_loadu_si128((__m128i *)au1_mask_cpy);
            // separating +ve and and -ve values.
            cmp_gt0_u_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_top_16x8b);
            cmp_lt0_u_16x8b = _mm_subs_epu16(src_top_16x8b, src_temp0_16x8b);

            pu2_src_left_cpy =pu2_src_left_cpy2;

            for(row = ht; row >=4; row -= 4)
            {
                left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_left_cpy));
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 12);
                // row = 0 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+1));

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signdwn1_16x8b, 14);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, signdwn1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp0_16x8b);

                // separating +ve and and -ve values.
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+2*src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 10);
                // manipulation for row 1 - bottom
                signdwn1_16x8b = _mm_alignr_epi8(src_bottom_16x8b, signdwn1_16x8b, 14);

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);

                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);

                // row1  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);

                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // using availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // row = 1 right
                signdwn1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd+1));

                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signdwn1_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_bottom_16x8b);

                // add offset + pixels and clip
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);
                // store left boundary
                left_store_16x8b_tmp = _mm_slli_si128(left_store_16x8b, 12);
                left_store_16x8b_tmp = _mm_srli_si128(left_store_16x8b_tmp, 12);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy+src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);

                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 12);
                // row = 0 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+1));

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signdwn1_16x8b, 14);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, signdwn1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp0_16x8b);

                // separating +ve and and -ve values.
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+2*src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 10);
                // manipulation for row 1 - bottom
                signdwn1_16x8b = _mm_alignr_epi8(src_bottom_16x8b, signdwn1_16x8b, 14);

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);

                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);

                // row1  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);

                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // using availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // row = 1 right
                signdwn1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd+1));

                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signdwn1_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_bottom_16x8b);

                // add offset + pixels and clip
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);
                // store left boundary
                left_store_16x8b = _mm_slli_si128(left_store_16x8b, 4);
                left_store_16x8b = _mm_or_si128(left_store_16x8b_tmp, left_store_16x8b);

                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy+src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);
                pu2_src_left_cpy += 4;
            }
            if(row >= 2)
            {
                left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_left_cpy));
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 12);
                // row = 0 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+1));

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signdwn1_16x8b, 14);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, signdwn1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp0_16x8b);

                // separating +ve and and -ve values.
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);

                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+2*src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 10);
                // manipulation for row 1 - bottom
                signdwn1_16x8b = _mm_alignr_epi8(src_bottom_16x8b, signdwn1_16x8b, 14);

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);

                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);

                // row1  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 14);

                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);

                // using availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // convert offset 8 bit to 16 bit
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // row = 1 right
                signdwn1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd+1));

                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signdwn1_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_bottom_16x8b);

                // add offset + pixels and clip
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clipping
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);
                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy+src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);
                pu2_src_left_cpy += 2;
            }
            ht_rem = ht&0x1;

            if(ht_rem)
            {
                left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_src_left_cpy);
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 12);
                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_bottom_16x8b, signdwn1_16x8b, 14);

                // current row -next row
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, signdwn1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp0_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
                // combining the appropriate sign change
                edge0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                // adding top and bottom and constant 2
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_u_16x8b, const0_16x8b);
                // for the next iteration bottom -row1
                signup0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);

                edge0_16x8b = _mm_add_epi16(edge0_16x8b, signup0_16x8b);

                edge0_16x8b = _mm_packs_epi16(edge0_16x8b, edge0_16x8b);
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 2);

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // using availability mask
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);
                // shuffle to get sao offset
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // cnvert to 16 bit then add and then saturated pack
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                // copying the next top
                src_top_16x8b = src_temp0_16x8b;

                // add offset + pixels and clip
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);

                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                pu2_src_cpy += (src_strd);
                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_left_cpy ++;
            }
            {   // for bottom right
                left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_src_left_cpy);
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 2);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 14);
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
            }
            if(0 == pu1_avail[3])
            {
                src_top_16x8b = src_bottom_16x8b;
            }
            // for the top left of next part of the block
            left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy+wd-col));
            // updating top flag
            _mm_storeu_si128((__m128i *)(pu2_src_top+wd-col), src_top_16x8b);
            pu2_src += 8;
            au1_mask_cpy += 8;
        }

        pu2_src_org[wd - 1] = u2_pos_wd_0_tmp;
        pu2_src_org[(ht_tmp - 1) * src_strd] = u2_pos_0_ht_tmp;
        for(row = 0; row < ht_tmp; row++)
        {
            pu2_src_left[row] = au2_src_left_tmp[row] ;
        }
    }
}

void ihevc_hbd_sao_edge_offset_class3_chroma_sse42(UWORD16 *pu2_src,
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
    UWORD16 *pu2_src_top_cpy, *pu2_src_left_cpy, *pu2_src_left_cpy2;
    UWORD16 *pu2_src_cpy, *pu2_src_org;
    UWORD16 au2_src_left_tmp[2*(MAX_CTB_SIZE+8)];
    UWORD8 au1_mask[2 * MAX_CTB_SIZE], *au1_mask_cpy;
    UWORD16 u2_pos_wd_0_tmp_u, u2_pos_wd_0_tmp_v, u2_pos_0_ht_tmp_u, u2_pos_0_ht_tmp_v;
    WORD32 ht_tmp;
    WORD32 bit_depth;
    UWORD8 u1_avail0, u1_avail1;

    __m128i src_top_16x8b, src_bottom_16x8b;
    __m128i src_temp0_16x8b, src_temp1_16x8b;
    __m128i signup0_16x8b, signdwn1_16x8b;
    __m128i cmp_gt0_16x8b, cmp_lt0_16x8b, cmp_gt1_16x8b, cmp_lt1_16x8b;
    __m128i cmp_gt0_u_16x8b, cmp_lt0_u_16x8b, cmp_gt1_b_16x8b, cmp_lt1_b_16x8b;
    __m128i edge0_16x8b, edge1_16x8b;
    __m128i au1_mask8x16b;
    __m128i edge_idx_8x16b;
    __m128i left_store_16x8b;
    __m128i const0_16x8b, const2_16x8b;
    __m128i sao_offset_u_8x16b, sao_offset_v_8x16b;
    __m128i m_min, m_max, m_count, const1_16x8b;
    __m128i sao_offset_8x16b, chroma_offset_8x16b;
    ht_tmp = ht;
    au1_mask8x16b = _mm_set1_epi8(0xff);

    au2_src_left_tmp[0] =pu2_src[(wd - 2)];
    au2_src_left_tmp[1] =pu2_src[(wd - 1)];
    // manipulation for bottom left
    for(row = 2; row < 2*ht; row++)
    {
        au2_src_left_tmp[row] =pu2_src_left[row];
    }
    au2_src_left_tmp[2*ht] = pu2_src_bot_left[0];
    au2_src_left_tmp[2*ht+1] = pu2_src_bot_left[1];

    pu2_src_top_left[0] = pu2_src_top[wd - 2];
    pu2_src_top_left[1] = pu2_src_top[wd - 1];
    // setting availability mask to ff size 2 * MAX_CTB_SIZE
    for(col = 0 ; col < 2 * MAX_CTB_SIZE; col += 16)
        _mm_storeu_si128((__m128i *)(au1_mask+col), au1_mask8x16b);
    bit_depth = u4_bit_depth;
    pu2_src_org = pu2_src;
    pu2_src_top_cpy = pu2_src_top;
    pu2_src_left_cpy2 = au2_src_left_tmp;
    pu2_src_left_cpy = au2_src_left_tmp;
    edge_idx_8x16b   = _mm_loadl_epi64((__m128i *)gi1_table_edge_idx_hbd);
    sao_offset_u_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset_u);
    sao_offset_v_8x16b = _mm_loadl_epi64((__m128i *)pi1_sao_offset_v);
    sao_offset_8x16b = _mm_unpacklo_epi64(sao_offset_u_8x16b, sao_offset_v_8x16b);
    chroma_offset_8x16b = _mm_set1_epi16(0x0800);
    /* If top-right is available, process separately */
    if(0 != pu1_avail[5])
    {
        WORD32 edge_idx;

        /* U */
        edge_idx = 2 + SIGN(pu2_src[wd - 2] - pu2_src_top_right[0]) +
            SIGN(pu2_src[wd - 2] - pu2_src[wd - 2 - 2 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_0_tmp_u = CLIP3(pu2_src[wd - 2] + pi1_sao_offset_u[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_0_tmp_u = pu2_src[wd - 2];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[wd - 1] - pu2_src_top_right[1]) +
            SIGN(pu2_src[wd - 1] - pu2_src[wd - 1 - 2 + src_strd]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_wd_0_tmp_v = CLIP3(pu2_src[wd - 1] + pi1_sao_offset_v[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_wd_0_tmp_v = pu2_src[wd - 1];
        }
    }
    else
    {
        u2_pos_wd_0_tmp_u = pu2_src[wd - 2];
        u2_pos_wd_0_tmp_v = pu2_src[wd - 1];
    }

    /* If bottom-left is available, process separately */
    if(0 != pu1_avail[6])
    {
        WORD32 edge_idx;

        /* U */
        edge_idx = 2 + SIGN(pu2_src[(ht - 1) * src_strd] - pu2_src[(ht - 1) * src_strd + 2 - src_strd]) +
            SIGN(pu2_src[(ht - 1) * src_strd] - pu2_src_bot_left[0]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_ht_tmp_u = CLIP3(pu2_src[(ht - 1) * src_strd] + pi1_sao_offset_u[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_0_ht_tmp_u = pu2_src[(ht - 1) * src_strd];
        }

        /* V */
        edge_idx = 2 + SIGN(pu2_src[(ht - 1) * src_strd + 1] - pu2_src[(ht - 1) * src_strd + 1 + 2 - src_strd]) +
            SIGN(pu2_src[(ht - 1) * src_strd + 1] - pu2_src_bot_left[1]);

        edge_idx = gi1_table_edge_idx_hbd[edge_idx];

        if(0 != edge_idx)
        {
            u2_pos_0_ht_tmp_v = CLIP3(pu2_src[(ht - 1) * src_strd + 1] + pi1_sao_offset_v[edge_idx], 0, (1 << bit_depth) - 1);
        }
        else
        {
            u2_pos_0_ht_tmp_v = pu2_src[(ht - 1) * src_strd + 1];
        }
    }
    else
    {
        u2_pos_0_ht_tmp_u = pu2_src[(ht - 1) * src_strd];
        u2_pos_0_ht_tmp_v = pu2_src[(ht - 1) * src_strd + 1];
    }

    /* Update height and source pointers based on the availability flags */
    if(0 == pu1_avail[2])
    {
        pu2_src_left_cpy2 += 2;
        pu2_src_top_cpy = pu2_src;
        pu2_src += src_strd;
        ht--;
    }
    if(0 == pu1_avail[3])
    {
        ht--;
    }

    m_max =  _mm_cmpeq_epi16(edge_idx_8x16b, edge_idx_8x16b);
    m_count = _mm_cvtsi32_si128((16 - u4_bit_depth));
    m_max = _mm_srl_epi16(m_max, m_count);
    m_min = _mm_setzero_si128();
    const2_16x8b = _mm_set1_epi8(2);
    const0_16x8b = _mm_setzero_si128();
    const1_16x8b = _mm_set1_epi8(1);

    // availability mask creation
    u1_avail0 = pu1_avail[0];
    u1_avail1 = pu1_avail[1];
    au1_mask[0] = u1_avail0;
    au1_mask[1] = u1_avail0;
    au1_mask[wd-1] = u1_avail1;
    au1_mask[wd-2] = u1_avail1;
    {
        WORD32 ht_rem;
        au1_mask_cpy = au1_mask;
        for(col = wd; col >= 8; col -= 8)
        {
            pu2_src_cpy = pu2_src;
            src_top_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy+wd-col+2));
            // row = 0
            src_temp0_16x8b =_mm_loadu_si128((__m128i *)(pu2_src_cpy));

            // loading the mask
            au1_mask8x16b = _mm_loadu_si128((__m128i *)au1_mask_cpy);
            // separating +ve and and -ve values.
            cmp_gt0_u_16x8b = _mm_subs_epu16(src_temp0_16x8b, src_top_16x8b);
            cmp_lt0_u_16x8b = _mm_subs_epu16(src_top_16x8b, src_temp0_16x8b);

            pu2_src_left_cpy =pu2_src_left_cpy2;

            for(row = ht; row >=2; row -= 2)
            {
                left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_left_cpy));
                // row = 1
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 8);
                // row = 0 right
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+2));

                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_temp1_16x8b, signdwn1_16x8b, 12);
                // row 0 -row1
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, signdwn1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp0_16x8b);

                // row1-row0
                // separating +ve and and -ve values.
                cmp_gt1_16x8b = _mm_subs_epu16(src_temp1_16x8b, src_bottom_16x8b);
                cmp_lt1_16x8b = _mm_subs_epu16(src_bottom_16x8b, src_temp1_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_packs_epi16(cmp_gt0_16x8b, cmp_gt1_16x8b);
                cmp_lt0_16x8b = _mm_packs_epi16(cmp_lt0_16x8b, cmp_lt1_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi8(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi8(cmp_lt0_16x8b, const0_16x8b);
                edge0_16x8b = _mm_sub_epi8(cmp_gt0_16x8b, cmp_lt0_16x8b);
                // row = 2
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+2*src_strd));
                // row = 1 right

                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 4);
                // manipulation for row 1 - bottom
                signdwn1_16x8b = _mm_alignr_epi8(src_bottom_16x8b, signdwn1_16x8b, 12);

                // row1 -bottom
                cmp_gt1_b_16x8b = _mm_subs_epu16(src_temp1_16x8b, signdwn1_16x8b);
                cmp_lt1_b_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp1_16x8b);

                cmp_gt0_u_16x8b = _mm_packs_epi16(cmp_gt0_u_16x8b, cmp_gt1_b_16x8b);
                cmp_lt0_u_16x8b = _mm_packs_epi16(cmp_lt0_u_16x8b, cmp_lt1_b_16x8b);

                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_u_16x8b = _mm_cmpeq_epi8(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_u_16x8b = _mm_cmpeq_epi8(cmp_lt0_u_16x8b, const0_16x8b);
                signup0_16x8b = _mm_sub_epi8(cmp_gt0_u_16x8b, cmp_lt0_u_16x8b);

                edge0_16x8b = _mm_add_epi8(edge0_16x8b, signup0_16x8b);

                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 8);
                // row1  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp1_16x8b, 12);
                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 12);
                // copying the next top
                src_top_16x8b = src_temp1_16x8b;

                // adding constant 2
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // use availability mask
                au1_mask8x16b = _mm_unpacklo_epi64(au1_mask8x16b, au1_mask8x16b);
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);
                // shuffle to get sao offset
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, chroma_offset_8x16b);
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                // cnvert to 16 bit then add and then saturated pack
                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge1_16x8b = _mm_unpackhi_epi8(edge0_16x8b, cmp_lt1_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                signdwn1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd+2));

                // bottom - row1
                cmp_gt0_u_16x8b = _mm_subs_epu16(src_bottom_16x8b, signdwn1_16x8b);
                cmp_lt0_u_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_bottom_16x8b);

                // add offset + pixels and clip
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp1_16x8b = _mm_add_epi16(src_temp1_16x8b, edge1_16x8b);
                // clip
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp1_16x8b = _mm_max_epi16(src_temp1_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);
                src_temp1_16x8b = _mm_min_epi16(src_temp1_16x8b, m_max);
                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                // row = 0 store 8 pixel values from 7:0 pos. relative to cur. pos.
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);
                // row = 1
                _mm_storeu_si128((__m128i *)(pu2_src_cpy+src_strd), src_temp1_16x8b);

                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_cpy += (src_strd<<1);
                pu2_src_left_cpy += 4;
            }
            ht_rem = ht&0x1;

            if(ht_rem)
            {
                left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_src_left_cpy);
                src_bottom_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_cpy+src_strd));
                // to insert left in row 1
                signdwn1_16x8b = _mm_slli_si128(left_store_16x8b, 8);
                // manipulation for row 1 - row 0
                signdwn1_16x8b = _mm_alignr_epi8(src_bottom_16x8b, signdwn1_16x8b, 12);

                // current row -next row
                // separating +ve and and -ve values.
                cmp_gt0_16x8b = _mm_subs_epu16(src_temp0_16x8b, signdwn1_16x8b);
                cmp_lt0_16x8b = _mm_subs_epu16(signdwn1_16x8b, src_temp0_16x8b);
                // creating mask 00 for +ve and -ve values and FF for zero.
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_16x8b, const0_16x8b);
                // combining the appropriate sign change
                edge0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);
                // adding top and bottom and constant 2
                cmp_gt0_16x8b = _mm_cmpeq_epi16(cmp_gt0_u_16x8b, const0_16x8b);
                cmp_lt0_16x8b = _mm_cmpeq_epi16(cmp_lt0_u_16x8b, const0_16x8b);
                // for the next iteration bottom -row1
                signup0_16x8b = _mm_sub_epi16(cmp_gt0_16x8b, cmp_lt0_16x8b);

                edge0_16x8b = _mm_add_epi16(edge0_16x8b, signup0_16x8b);
                edge0_16x8b = _mm_packs_epi16(edge0_16x8b, edge0_16x8b);
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, const2_16x8b);
                // eliminating old left for row 0 and row 1
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                // row0  getting it right for left of next block
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 12);
                // copying the next top
                src_top_16x8b = src_temp0_16x8b;

                // shuffle to get sao index
                edge0_16x8b = _mm_shuffle_epi8(edge_idx_8x16b, edge0_16x8b);
                // using availability mask
                edge0_16x8b = _mm_and_si128(edge0_16x8b, au1_mask8x16b);

                // shuffle to get sao offset
                edge0_16x8b = _mm_add_epi8(edge0_16x8b, chroma_offset_8x16b);
                edge0_16x8b = _mm_shuffle_epi8(sao_offset_8x16b, edge0_16x8b);

                cmp_lt1_16x8b =  _mm_cmpgt_epi8(const0_16x8b, edge0_16x8b);
                edge0_16x8b = _mm_unpacklo_epi8(edge0_16x8b, cmp_lt1_16x8b);

                // add offset + pixels and clip
                src_temp0_16x8b = _mm_add_epi16(src_temp0_16x8b, edge0_16x8b);
                src_temp0_16x8b = _mm_max_epi16(src_temp0_16x8b, m_min);
                src_temp0_16x8b = _mm_min_epi16(src_temp0_16x8b, m_max);

                // store left boundary
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
                _mm_storeu_si128((__m128i *)(pu2_src_cpy), src_temp0_16x8b);

                pu2_src_cpy += (src_strd);
                src_temp0_16x8b = src_bottom_16x8b;
                pu2_src_left_cpy += 2;
            }
            {   // for bottom right
                left_store_16x8b = _mm_loadl_epi64((__m128i *)pu2_src_left_cpy);
                left_store_16x8b = _mm_srli_si128(left_store_16x8b, 4);
                left_store_16x8b = _mm_alignr_epi8(left_store_16x8b, src_temp0_16x8b, 12);
                _mm_storel_epi64((__m128i *)(pu2_src_left_cpy), left_store_16x8b);
            }
            if(0 == pu1_avail[3])
            {
                src_top_16x8b = src_bottom_16x8b;
            }
            // for the top left of next part of the block
            left_store_16x8b = _mm_loadu_si128((__m128i *)(pu2_src_top_cpy + wd - col));
            // updating top flag
            _mm_storeu_si128((__m128i *)(pu2_src_top + wd - col), src_top_16x8b);
            pu2_src += 8;
            au1_mask_cpy += 8;
        }

        pu2_src_org[wd-2] = u2_pos_wd_0_tmp_u;
        pu2_src_org[wd-1] = u2_pos_wd_0_tmp_v;
        pu2_src_org[(ht_tmp - 1) * src_strd] = u2_pos_0_ht_tmp_u;
        pu2_src_org[(ht_tmp - 1) * src_strd+1] = u2_pos_0_ht_tmp_v;
        for(row = 0; row < 2*ht_tmp; row++)
        {
            pu2_src_left[row] = au2_src_left_tmp[row] ;
        }
    }
}
