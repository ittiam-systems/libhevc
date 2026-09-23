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
*  ihevc_hbd_chroma_intra_pred_filters_sse42_intr.c
*
* @brief
*  Contains function definitions for high bit depth chroma intra prediction
*  interpolation filters (SSE4.2).
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_intra_pred_chroma_planar_sse42()
*  - ihevc_hbd_intra_pred_chroma_ver_sse42()
*  - ihevc_hbd_intra_pred_chroma_horz_sse42()
*  - ihevc_hbd_intra_pred_chroma_dc_sse42()
*  - ihevc_hbd_intra_pred_chroma_mode2_sse42()
*  - ihevc_hbd_intra_pred_chroma_mode_18_34_sse42()
*  - ihevc_hbd_intra_pred_chroma_mode_3_to_9_sse42()
*  - ihevc_hbd_intra_pred_chroma_mode_11_to_17_sse42()
*  - ihevc_hbd_intra_pred_chroma_mode_19_to_25_sse42()
*  - ihevc_hbd_intra_pred_chroma_mode_27_to_33_sse42()
*
* @remarks
*  None
*
*******************************************************************************
*/

/**
*******************************************************************************
* @file
*  ihevc_hbd_chroma_intra_pred_filters_x86_intr.c
*
* @brief
*  Contains function Definition for intra prediction  interpolation filters
*
*
* @author
*  Ittiam
*
* @par List of Functions:
*  ihevc_hbd_intra_pred_chroma_planar()
*
*  ihevc_hbd_intra_pred_chroma_dc()
*
*  ihevc_hbd_intra_pred_chroma_horz()
*
*  ihevc_hbd_intra_pred_chroma_ver()
*
*  ihevc_hbd_intra_pred_chroma_mode2()
*
*  ihevc_hbd_intra_pred_chroma_mode_18_34()
*
*  ihevc_hbd_intra_pred_chroma_mode_3_to_9()
*
*  ihevc_hbd_intra_pred_chroma_mode_11_to_17()
*
*  ihevc_hbd_intra_pred_chroma_mode_19_to_25()
*
*  ihevc_hbd_intra_pred_chroma_mode_27_to_33()
*
*  ihevc_hbd_intra_pred_chroma_ref_substitution()
*
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
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_intra_pred.h"

#include "ihevc_chroma_intra_pred.h"
#include "ihevc_common_tables.h"
#include "ihevc_hbd_tables_x86_intr.h"


#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>


/****************************************************************************/
/* Constant Macros                                                          */
/****************************************************************************/
#define MAX_CU_SIZE 64
#define T32_4NT 128
#define T16_4NT 64
#define T16C_4NT 64
#define T8C_4NT 32

/**
*******************************************************************************
*
* @brief
*  Planar Intraprediction with reference neighboring samples location
* pointed by 'pu2_ref' to the TU block location  pointed by 'pu2_dst'  Refer
* to section 8.4.4.2.4 in the standard
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_planar_sse42(UWORD16 *pu2_ref,
                                              WORD32 src_strd,
                                              UWORD16 *pu2_dst,
                                              WORD32 dst_strd,
                                              WORD32 nt,
                                              WORD32 mode)
{
    WORD32 row, col;
    WORD32 log2nt = 5;
    WORD32 two_nt, three_nt;

    __m128i const_temp_4x32b, const_temp1_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b;
    __m128i col_8x16b, const_temp5_4x32b, const_temp6_4x32b, zero_8x16b, mask_4x32b, const_temp7_4x32b;
    __m128i const_temp8_4x32b, const_temp9_4x32b;
    __m128i col2_8x16b;

    switch(nt)
    {
        case 16:
            log2nt = 4;
            break;
        case 8:
            log2nt = 3;
            break;
        case 4:
            log2nt = 2;
            break;
        default:
            break;
    }
    two_nt = 2 * nt;
    three_nt = 3 * nt;

     /* Planar filtering */

    const_temp_4x32b  = _mm_set_epi32(pu2_ref[2 * (three_nt + 1)+1], pu2_ref[2 * (three_nt + 1)], pu2_ref[2 * (three_nt + 1)+1],
                                        pu2_ref[2 * (three_nt + 1)]);

    const_temp1_4x32b = _mm_set_epi32(pu2_ref[2 * (nt - 1)+1], pu2_ref[2 * (nt - 1)], pu2_ref[2 * (nt - 1)+1], pu2_ref[2 * (nt - 1)]);

    const_temp4_4x32b = _mm_set1_epi32(nt - 1);
    const_temp6_4x32b = _mm_set1_epi32(nt);
    const_temp7_4x32b = _mm_set1_epi32(4);

    zero_8x16b = _mm_set1_epi32(0);

    mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

    if(nt%4==0)
    {
        const_temp7_4x32b = _mm_set1_epi32(4);
        const_temp9_4x32b = _mm_set1_epi16(4);
        for(row = 0; row < nt; row++)
        {
            __m128i res_temp_8x16b, row_8x16b, res_temp1_8x16b, res_temp2_8x16b;
            __m128i res_temp3_8x16b, res_temp4_8x16b;

            const_temp2_4x32b  = _mm_set_epi32( pu2_ref[2*(two_nt - 1 - row)+1], pu2_ref[2*(two_nt - 1 - row)], pu2_ref[2*(two_nt - 1 - row)+1],
                pu2_ref[2*(two_nt - 1 - row)]);

            const_temp3_4x32b  = _mm_set1_epi32((row + 1));
            row_8x16b = _mm_set1_epi32((nt - 1 - row));

            const_temp5_4x32b = _mm_set_epi32(1, 1, 0, 0);
            const_temp8_4x32b = _mm_set_epi32(3, 3, 2, 2);
            col_8x16b = _mm_set_epi16(4, 4, 3, 3, 2, 2, 1, 1);

            const_temp5_4x32b = _mm_sub_epi32(const_temp4_4x32b, const_temp5_4x32b);
            const_temp8_4x32b = _mm_sub_epi32(const_temp4_4x32b, const_temp8_4x32b);

            /* (row + 1) * pu1_ref[nt - 1] */
            res_temp_8x16b  = _mm_mullo_epi32(const_temp3_4x32b,  const_temp1_4x32b);

            /* (row + 1) * pu1_ref[nt - 1] + nt) */
            res_temp_8x16b = _mm_add_epi32(res_temp_8x16b, const_temp6_4x32b);

            for(col = 0; col < 2*nt; col += 8)
            {
                __m128i src_temp_8x16b, src_temp1_8x16b, src_temp2_8x16b;

                /* loding 16bit 8 pixles */
                src_temp_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+2*(two_nt + 1) + col));
                src_temp1_8x16b = _mm_unpacklo_epi16(src_temp_8x16b, zero_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi16(src_temp_8x16b, zero_8x16b);

                /* ** first 4 coloumns **** */
                /* (nt - 1 - row) * pu1_ref[two_nt + 1 + col] */
                res_temp1_8x16b  = _mm_mullo_epi32(src_temp1_8x16b,  row_8x16b);

                col2_8x16b =  _mm_unpacklo_epi16(col_8x16b, zero_8x16b);

                /* (col + 1) * pu1_ref[three_nt + 1] */
                res_temp2_8x16b  = _mm_mullo_epi32(const_temp_4x32b,  col2_8x16b);

                /* (nt - 1 - col)* pu1_ref[two_nt - 1 - row] */
                res_temp3_8x16b  = _mm_mullo_epi32(const_temp2_4x32b,  const_temp5_4x32b);

                res_temp1_8x16b = _mm_add_epi32(res_temp_8x16b, res_temp1_8x16b);
                res_temp1_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp1_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp3_8x16b);

                res_temp1_8x16b = _mm_srli_epi32(res_temp1_8x16b, (log2nt + 1));

                /* ** next 4 coloumns **** */

                /* (nt - 1 - row) * pu2_ref[two_nt + 1 + col] */
                res_temp4_8x16b  = _mm_mullo_epi32(src_temp2_8x16b,  row_8x16b);

                col2_8x16b =  _mm_unpackhi_epi16(col_8x16b, zero_8x16b);
                /* (col + 1) * pu2_ref[three_nt + 1] */
                res_temp2_8x16b  = _mm_mullo_epi32(const_temp_4x32b,  col2_8x16b);

                /* (nt - 1 - col)* pu2_ref[two_nt - 1 - row] */
                res_temp3_8x16b  = _mm_mullo_epi32(const_temp2_4x32b,  const_temp8_4x32b);

                res_temp4_8x16b = _mm_add_epi32(res_temp_8x16b, res_temp4_8x16b);
                res_temp4_8x16b = _mm_add_epi32(res_temp4_8x16b, res_temp2_8x16b);
                res_temp4_8x16b = _mm_add_epi32(res_temp4_8x16b, res_temp3_8x16b);

                res_temp4_8x16b = _mm_srli_epi32(res_temp4_8x16b, (log2nt + 1));

                // Pack first 4 columns and next 4 coloumns together
                res_temp1_8x16b = _mm_packus_epi32(res_temp1_8x16b, res_temp4_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+(row * dst_strd) + col), res_temp1_8x16b);

                const_temp5_4x32b = _mm_sub_epi32(const_temp5_4x32b, const_temp7_4x32b);
                const_temp8_4x32b = _mm_sub_epi32(const_temp8_4x32b, const_temp7_4x32b);
                col_8x16b = _mm_add_epi16(col_8x16b, const_temp9_4x32b);
            } /* inner loop ends here */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Horizontal intraprediction with reference neighboring  samples location
* pointed by 'pu2_ref' to the TU block  location pointed by 'pu2_dst'  Refer
* to section 8.4.4.2.6 in the standard (Special case)
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_ver_sse42(UWORD16 *pu2_ref,
                                           WORD32 src_strd,
                                           UWORD16 *pu2_dst,
                                           WORD32 dst_strd,
                                           WORD32 nt,
                                           WORD32 mode)
{
    __m128i src_temp1;

    if(nt > 16)
    {
        ihevc_hbd_intra_pred_chroma_ver(pu2_ref, src_strd, pu2_dst, dst_strd, nt, mode);
        return;
    }

    /* Replication to next columns */
    if(nt==8)
    {
        src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) + 2 + 0));

        _mm_storeu_si128((__m128i *)(pu2_dst+((0) * dst_strd) ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd) ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd) ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd) ), src_temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd) ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd) ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd) ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd) ), src_temp1);

        src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) + 2 + 8));

        _mm_storeu_si128((__m128i *)(pu2_dst+((0) * dst_strd)+8 ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)+8 ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)+8 ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)+8 ), src_temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)+8 ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)+8 ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)+8 ), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)+8 ), src_temp1);
    }
    if(nt==16)
    {
        __m128i temp1, temp2;

        temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) + 2 + 0));
        temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) + 2 + 8));

        /* pu1_dst[(row * dst_strd) + col] = dc_val; */
        _mm_storeu_si128((__m128i *)(pu2_dst+((0) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((4) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((5) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((6) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((7) * dst_strd)), temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+8+((0)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((1)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((2)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((3)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((4)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((5)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((6)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((7)  * dst_strd)), temp2);

        _mm_storeu_si128((__m128i *)(pu2_dst+((8) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((9) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((10) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((11) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((12) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((13) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((14) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+((15) * dst_strd)), temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+8+((8)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((9)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((10) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((11) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((12) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((13) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((14) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+8+((15) * dst_strd)), temp2);

        temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) + 2 + 16));
        temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) + 2 + 24));

        /* pu1_dst[(row * dst_strd) + col] = dc_val; */
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((0) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((1) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((2) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((3) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((4) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((5) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((6) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((7) * dst_strd)), temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+24+((0)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((1)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((2)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((3)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((4)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((5)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((6)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((7)  * dst_strd)), temp2);

        _mm_storeu_si128((__m128i *)(pu2_dst+16+((8) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((9) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((10) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((11) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((12) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((13) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((14) * dst_strd)), temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+16+((15) * dst_strd)), temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+24+((8)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((9)  * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((10) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((11) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((12) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((13) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((14) * dst_strd)), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+24+((15) * dst_strd)), temp2);
    }
    else
    {
        src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) + 2 + 0));

        _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp1);
    }
}
/**
*******************************************************************************
*
* @brief
*  Horizontal intraprediction(mode 10) with reference  samples location
* pointed by 'pu2_ref' to the TU block  location pointed by 'pu2_dst'  Refer
* to section 8.4.4.2.6 in the standard (Special case)
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_horz_sse42(UWORD16 *pu2_ref,
                                            WORD32 src_strd,
                                            UWORD16 *pu2_dst,
                                            WORD32 dst_strd,
                                            WORD32 nt,
                                            WORD32 mode)
{
    WORD32 row;
    __m128i temp1, temp2, temp3, temp4, temp5, temp6,  temp7, temp8;

    if(nt > 16)
    {
        ihevc_hbd_intra_pred_chroma_horz(pu2_ref, src_strd, pu2_dst, dst_strd, nt, mode);
        return;
    }

    /* Replication to next rows */

    if(nt ==8)
    {
        for(row = 0; row < nt ; row += 4)
        {
            temp1 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+0)]);
            temp2 = _mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+0)]);
            temp3 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+1)]);
            temp4 = _mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+1)]);
            temp5 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+2)]);
            temp6 = _mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+2)]);
            temp7 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+3)]);
            temp8 = _mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+3)]);

            temp2= _mm_unpacklo_epi16(temp1, temp2);
            temp4= _mm_unpacklo_epi16(temp3, temp4);
            temp6= _mm_unpacklo_epi16(temp5, temp6);
            temp8= _mm_unpacklo_epi16(temp7, temp8);

            _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd) ), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd) ), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd) ), temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd) ), temp8);

            _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd)+8 ), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd)+8 ), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd)+8 ), temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd)+8 ), temp8);
        }
    }
    else if(nt ==16)
    {
        for(row = 0; row < nt ; row += 4)
        {
            temp1 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+0)]);
            temp2 =_mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+0)]);

            temp3 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+1)]);
            temp4 = _mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+1)]);

            temp5 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+2)]);
            temp6 =_mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+2)]);

            temp7 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * (row+3)]);
            temp8 = _mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * (row+3)]);

            temp2= _mm_unpacklo_epi16(temp1, temp2);
            temp4= _mm_unpacklo_epi16(temp3, temp4);
            temp6= _mm_unpacklo_epi16(temp5, temp6);
            temp8= _mm_unpacklo_epi16(temp7, temp8);

            _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd) + 0), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd) + 8), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd) + 16), temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+0) * dst_strd) + 24), temp2);

            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd) + 0), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd) + 8), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd) + 16), temp4);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+1) * dst_strd) + 24), temp4);

            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd) + 0), temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd) + 8), temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd) + 16), temp6);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+2) * dst_strd) + 24), temp6);

            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd) + 0), temp8);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd) + 8), temp8);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd) + 16), temp8);
            _mm_storeu_si128((__m128i *)(pu2_dst+((row+3) * dst_strd) + 24), temp8);
        }
    }
    else
    {
        temp1 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * 0]);
        temp2 =_mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * 0]);

        temp3 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * 1]);
        temp4 =_mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * 1]);

        temp5 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * 2]);
        temp6 =_mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * 2]);

        temp7 = _mm_set1_epi16(pu2_ref[(4 * nt) - 2 - 2 * 3]);
        temp8 =_mm_set1_epi16(pu2_ref[(4 * nt) - 1 - 2 * 3]);

        temp2 = _mm_unpacklo_epi16(temp1, temp2);
        temp4 = _mm_unpacklo_epi16(temp3, temp4);
        temp6 = _mm_unpacklo_epi16(temp5, temp6);
        temp8 = _mm_unpacklo_epi16(temp7, temp8);

        _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd) ), temp2);
        _mm_storeu_si128((__m128i *)(pu2_dst+(1 * dst_strd) ), temp4);
        _mm_storeu_si128((__m128i *)(pu2_dst+(2 * dst_strd) ), temp6);
        _mm_storeu_si128((__m128i *)(pu2_dst+(3 * dst_strd) ), temp8);
    }
}
/**
*******************************************************************************
*
* @brief
*  Intraprediction for DC mode with reference neighboring  samples location
* pointed by 'pu2_ref' to the TU block  location pointed by 'pu2_dst'  Refer
* to section 8.4.4.2.5 in the standard
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size (Chroma)
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_dc_sse42(UWORD16 *pu2_ref,
                                          WORD32 src_strd,
                                          UWORD16 *pu2_dst,
                                          WORD32 dst_strd,
                                          WORD32 nt,
                                          WORD32 mode)
{
    WORD32 acc_dc_u, acc_dc_v;
    WORD32 dc_val_u , dc_val_v;
    WORD32 row;
    WORD32 log2nt = 5;
    __m128i src_temp1, src_temp3, src_temp4, src_temp5, src_temp6, m_mask;
    __m128i src_temp7, src_temp8, src_temp9, src_temp10;
    __m128i m_zero =_mm_set1_epi32(0);
    __m128i const_16x8b;

    if(nt > 16)
    {
        ihevc_hbd_intra_pred_chroma_dc(pu2_ref, src_strd, pu2_dst, dst_strd, nt, mode);
        return;
    }

    switch(nt)
    {
        case 32:
            log2nt = 5;
            break;
        case 16:
            log2nt = 4;
            break;
        case 8:
            log2nt = 3;
            break;
        case 4:
            log2nt = 2;
            break;
        default:
            break;
    }

    acc_dc_u = 0;
    acc_dc_v = 0;

    /* Calculate DC value for the transform block */

    m_mask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY9_HBD[0]);

    if(nt==16)
    {
        const_16x8b = _mm_cmpeq_epi16(m_zero, m_zero);
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);
        src_temp3 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)));
        src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+8));
        src_temp5 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+16));
        src_temp6 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+24));
        src_temp7 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+32));
        src_temp8 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+40));
        src_temp9 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+48));
        src_temp10 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+56));

        src_temp3 = _mm_add_epi16(src_temp3, src_temp4);
        src_temp5 = _mm_add_epi16(src_temp5, src_temp6);
        src_temp7 = _mm_add_epi16(src_temp7, src_temp8);
        src_temp9 = _mm_add_epi16(src_temp9, src_temp10);

        src_temp3 = _mm_add_epi16(src_temp3, src_temp5);
        src_temp7 = _mm_add_epi16(src_temp7, src_temp9);

        src_temp3 = _mm_add_epi16(src_temp3, src_temp7);

        src_temp3 = _mm_shuffle_epi8(src_temp3, m_mask);

        src_temp3 = _mm_madd_epi16(src_temp3, const_16x8b);

        src_temp4 = _mm_shuffle_epi32(src_temp3, 0x01);
        src_temp5 = _mm_shuffle_epi32(src_temp3, 0x02);
        src_temp6 = _mm_shuffle_epi32(src_temp3, 0x03);

        src_temp3 = _mm_add_epi32(src_temp3, src_temp4);
        src_temp5 = _mm_add_epi32(src_temp5, src_temp6);

        acc_dc_u  = _mm_cvtsi128_si32(src_temp3);
        acc_dc_v  = _mm_cvtsi128_si32(src_temp5);
    }
    else if(nt==8)
    {
        src_temp3 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)));
        src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+8));
        src_temp5 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+16));
        src_temp6 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)+24));

        const_16x8b = _mm_cmpeq_epi16(src_temp3, src_temp3);
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);

        src_temp3 = _mm_add_epi16(src_temp3, src_temp4);
        src_temp5 = _mm_add_epi16(src_temp5, src_temp6);
        src_temp3 = _mm_add_epi16(src_temp3, src_temp5);

        src_temp3 = _mm_shuffle_epi8(src_temp3, m_mask);

        src_temp3 = _mm_madd_epi16(src_temp3, const_16x8b);

        src_temp4 = _mm_shuffle_epi32(src_temp3, 0x01);
        src_temp5 = _mm_shuffle_epi32(src_temp3, 0x02);
        src_temp6 = _mm_shuffle_epi32(src_temp3, 0x03);

        src_temp3 = _mm_add_epi32(src_temp3, src_temp4);
        src_temp5 = _mm_add_epi32(src_temp5, src_temp6);

        acc_dc_u  = _mm_cvtsi128_si32(src_temp3);
        acc_dc_v  = _mm_cvtsi128_si32(src_temp5);
    }
    else if(nt==4)
    {
        src_temp3 =  _mm_loadu_si128((__m128i *)(pu2_ref+(2*nt)));
        src_temp4 =  _mm_loadu_si128((__m128i *)(pu2_ref+(4*nt)));

        const_16x8b = _mm_cmpeq_epi16(src_temp3, src_temp3);
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);

        src_temp3 = _mm_add_epi16(src_temp3, src_temp4);

        src_temp3 = _mm_shuffle_epi8(src_temp3, m_mask);

        src_temp3 = _mm_madd_epi16(src_temp3, const_16x8b);

        src_temp4 = _mm_shuffle_epi32(src_temp3, 0x01);
        src_temp5 = _mm_shuffle_epi32(src_temp3, 0x02);
        src_temp6 = _mm_shuffle_epi32(src_temp3, 0x03);

        src_temp3 = _mm_add_epi32(src_temp3, src_temp4);
        src_temp5 = _mm_add_epi32(src_temp5, src_temp6);

        acc_dc_u  = _mm_cvtsi128_si32(src_temp3);
        acc_dc_v  = _mm_cvtsi128_si32(src_temp5);
    }

    acc_dc_u += pu2_ref[6*nt];
    acc_dc_v += pu2_ref[6*nt+1];

    acc_dc_u -= pu2_ref[4*nt];
    acc_dc_v -= pu2_ref[4*nt+1];

    dc_val_u = (acc_dc_u + nt) >> (log2nt + 1);
    dc_val_v = (acc_dc_v + nt) >> (log2nt + 1);

    dc_val_u = dc_val_u | (dc_val_v << 16);

    /* Fill the remaining rows with DC value */

    if(nt == 4)
    {
        src_temp1 = _mm_set1_epi32(dc_val_u);

        /* pu1_dst[(row * dst_strd) + col] = dc_val; */
        _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp1);
    }
    else if(nt == 8)
    {
        src_temp1 = _mm_set1_epi32(dc_val_u);

        /* pu1_dst[(row * dst_strd) + col] = dc_val; */
        _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+(4 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(5 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(6 * dst_strd)), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(7 * dst_strd)), src_temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd)+8), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(1 * dst_strd)+8), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(2 * dst_strd)+8), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(3 * dst_strd)+8), src_temp1);

        _mm_storeu_si128((__m128i *)(pu2_dst+(4 * dst_strd)+8), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(5 * dst_strd)+8), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(6 * dst_strd)+8), src_temp1);
        _mm_storeu_si128((__m128i *)(pu2_dst+(7 * dst_strd)+8), src_temp1);
    }
    else /* nt == 16 */
    {
        src_temp1 = _mm_set1_epi32(dc_val_u);

        for(row = 0; row < nt; row += 8)
        {
            /* pu1_dst[(row * dst_strd) + col] = dc_val; */
            _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+(1 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+(2 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+(3 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(0 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(1 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(2 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(3 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(0 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(1 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(2 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(3 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(0 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(1 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(2 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(3 * dst_strd)), src_temp1);

            _mm_storeu_si128((__m128i *)(pu2_dst+(4 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+(5 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+(6 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+(7 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(4 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(5 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(6 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+8+(7 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(4 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(5 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(6 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16+(7 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(4 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(5 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(6 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst+24+(7 * dst_strd)), src_temp1);

            pu2_dst += 8*dst_strd;
        }
    }
}
/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 2 (sw angle) with reference  neighboring samples
* location pointed by 'pu2_ref' to the  TU block location pointed by
* 'pu2_dst'  Refer to section 8.4.4.2.6 in the standard
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_intra_pred_chroma_mode2_sse42(UWORD16 *pu2_ref,
                                             WORD32 src_strd,
                                             UWORD16 *pu2_dst,
                                             WORD32 dst_strd,
                                             WORD32 nt,
                                             WORD32 mode)
{
    WORD32 row, col;

    __m128i src_temp1, src_temp2, src_temp3, src_temp4, src_temp5, src_temp6, src_temp7, src_temp8, sm2;

    sm2 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY7_HBD[0]);

    /* For the angle 45, replication is done from the corresponding angle */
    /* intra_pred_ang = tan(angle) in q5 format */

    if(nt==4)
    {
        /* pu1_ref[two_nt - row - (col+1) - 1] */
        src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 0 - 8 - 2));
        src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 1 - 8 - 2));
        src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 2 - 8 - 2));
        src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 3 - 8 - 2));

        _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), _mm_shuffle_epi8(src_temp1, sm2));
        _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), _mm_shuffle_epi8(src_temp2, sm2));
        _mm_storeu_si128((__m128i *)(pu2_dst + (2 * dst_strd)), _mm_shuffle_epi8(src_temp3, sm2));
        _mm_storeu_si128((__m128i *)(pu2_dst + (3 * dst_strd)), _mm_shuffle_epi8(src_temp4, sm2));
    }
    else if(nt==8)
    {
        for(col = 0; col <2*nt; col += 8)
        {
            /* pu1_ref[two_nt - row - (col+1) - 1] */
            src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 0 - (col+8) - 2));
            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 1 - (col+8) - 2));
            src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 2 - (col+8) - 2));
            src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 3 - (col+8) - 2));
            src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 4 - (col+8) - 2));
            src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 5 - (col+8) - 2));
            src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 6 - (col+8) - 2));
            src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * 7 - (col+8) - 2));

            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (0 * dst_strd)), _mm_shuffle_epi8(src_temp1, sm2));
            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (1 * dst_strd)), _mm_shuffle_epi8(src_temp2, sm2));
            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (2 * dst_strd)), _mm_shuffle_epi8(src_temp3, sm2));
            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (3 * dst_strd)), _mm_shuffle_epi8(src_temp4, sm2));
            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (4 * dst_strd)), _mm_shuffle_epi8(src_temp5, sm2));
            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (5 * dst_strd)), _mm_shuffle_epi8(src_temp6, sm2));
            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (6 * dst_strd)), _mm_shuffle_epi8(src_temp7, sm2));
            _mm_storeu_si128((__m128i *)(pu2_dst+ col + (7 * dst_strd)), _mm_shuffle_epi8(src_temp8, sm2));
        }
    }
    else
    {
        for(row = 0; row < nt; row += 8)
        {
            for(col = 0; col <2*nt; col += 8)
            {   /* pu1_ref[two_nt - row - (col+1) - 1] */
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+0)  - (col+8) - 2));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+1)  - (col+8) - 2));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+2)  - (col+8) - 2));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+3)  - (col+8) - 2));
                src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+4)  - (col+8) - 2));
                src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+5)  - (col+8) - 2));
                src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+6)  - (col+8) - 2));
                src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+(4 * nt) - 2 * (row+7)  - (col+8) - 2));

                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+0)  * dst_strd)), _mm_shuffle_epi8(src_temp1, sm2));
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+1)  * dst_strd)), _mm_shuffle_epi8(src_temp2, sm2));
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+2)  * dst_strd)), _mm_shuffle_epi8(src_temp3, sm2));
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+3)  * dst_strd)), _mm_shuffle_epi8(src_temp4, sm2));
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+4)  * dst_strd)), _mm_shuffle_epi8(src_temp5, sm2));
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+5)  * dst_strd)), _mm_shuffle_epi8(src_temp6, sm2));
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+6)  * dst_strd)), _mm_shuffle_epi8(src_temp7, sm2));
                _mm_storeu_si128((__m128i *)(pu2_dst + col + ( (row+7)  * dst_strd)), _mm_shuffle_epi8(src_temp8, sm2));
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 34 (ne angle) and  mode 18 (nw angle) with
* reference  neighboring samples location pointed by 'pu1_ref' to the  TU
* block location pointed by 'pu1_dst'
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_mode_18_34_sse42(UWORD16 *pu2_ref,
                                                  WORD32 src_strd,
                                                  UWORD16 *pu2_dst,
                                                  WORD32 dst_strd,
                                                  WORD32 nt,
                                                  WORD32 mode)
{
    WORD32 row, col;

    __m128i src_temp1, src_temp2, src_temp3, src_temp4;
    __m128i  src_temp5, src_temp6, src_temp7, src_temp8;

    if(mode == 34)
    {
        if(nt==4)
        {
            /* pu1_ref[two_nt + col + idx + 1] */
            src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(0 +1)+(4 * nt) + 2));
            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(1 +1)+(4 * nt) + 2));
            src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(2 +1)+(4 * nt) + 2));
            src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(3 +1)+(4 * nt) + 2));

            _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst + (2 * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst + (3 * dst_strd)), src_temp4);
        }
        else if(nt==8)
        {
            for(col=0;col<(2*nt);col += 8)
            {
                /* pu1_ref[two_nt + col + idx + 1] */
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(0 +1)+(4 * nt) + col + 2));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(1 +1)+(4 * nt) + col + 2));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(2 +1)+(4 * nt) + col + 2));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(3 +1)+(4 * nt) + col + 2));
                src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(4 +1)+(4 * nt) + col + 2));
                src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(5 +1)+(4 * nt) + col + 2));
                src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(6 +1)+(4 * nt) + col + 2));
                src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref+2*(7 +1)+(4 * nt) + col + 2));

                _mm_storeu_si128((__m128i *)(pu2_dst + col + (0 * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst + col + (1 * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst + col + (2 * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst + col + (3 * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst + col + (4 * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst + col + (5 * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst + col + (6 * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst + col + (7 * dst_strd)), src_temp8);
            }
        }
        else
        {
            __m128i src_temp9, src_temp10, src_temp11, src_temp12, src_temp13, src_temp14, src_temp15, src_temp16;
            for(row = 0; row < nt; row += 8)
            {
                for(col=0;col<(2*nt);col += 16)
                {
                    /* pu1_ref[two_nt + col + idx + 1] */
                    src_temp1  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(0+1)+ (col+0) +(4 * nt) + 2));
                    src_temp9  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(0+1)+ (col+8) +(4 * nt) + 2));
                    src_temp2  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(1+1)+ (col+0) +(4 * nt) + 2));
                    src_temp10 = _mm_loadu_si128((__m128i *)(pu2_ref +2*(1+1)+ (col+8) +(4 * nt) + 2));
                    src_temp3  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(2+1)+ (col+0) +(4 * nt) + 2));
                    src_temp11 = _mm_loadu_si128((__m128i *)(pu2_ref +2*(2+1)+ (col+8) +(4 * nt) + 2));
                    src_temp4  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(3+1)+ (col+0) +(4 * nt) + 2));
                    src_temp12 = _mm_loadu_si128((__m128i *)(pu2_ref +2*(3+1)+ (col+8) +(4 * nt) + 2));

                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (0 * dst_strd)), src_temp1 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (0 * dst_strd)), src_temp9 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (1 * dst_strd)), src_temp2 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (1 * dst_strd)), src_temp10);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (2 * dst_strd)), src_temp3 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (2 * dst_strd)), src_temp11);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (3 * dst_strd)), src_temp4 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (3 * dst_strd)), src_temp12);

                    src_temp5  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(4+1)+ (col+0) +(4 * nt) + 2));
                    src_temp13 = _mm_loadu_si128((__m128i *)(pu2_ref +2*(4+1)+ (col+8) +(4 * nt) + 2));
                    src_temp6  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(5+1)+ (col+0) +(4 * nt) + 2));
                    src_temp14 = _mm_loadu_si128((__m128i *)(pu2_ref +2*(5+1)+ (col+8) +(4 * nt) + 2));
                    src_temp7  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(6+1)+ (col+0) +(4 * nt) + 2));
                    src_temp15 = _mm_loadu_si128((__m128i *)(pu2_ref +2*(6+1)+ (col+8) +(4 * nt) + 2));
                    src_temp8  = _mm_loadu_si128((__m128i *)(pu2_ref +2*(7+1)+ (col+0) +(4 * nt) + 2));
                    src_temp16 = _mm_loadu_si128((__m128i *)(pu2_ref +2*(7+1)+ (col+8) +(4 * nt) + 2));

                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (4 * dst_strd)), src_temp5 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (4 * dst_strd)), src_temp13);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (5 * dst_strd)), src_temp6 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (5 * dst_strd)), src_temp14);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (6 * dst_strd)), src_temp7 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (6 * dst_strd)), src_temp15);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (7 * dst_strd)), src_temp8 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (7 * dst_strd)), src_temp16);
                }

                pu2_ref += 2*8;
                pu2_dst += 8*dst_strd;
            }
        }
    }
    else
    {
        if(nt==4)
        {
            /* pu1_ref[two_nt + col + idx + 1] */
            src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(0 +1)+(4 * nt) + 2));
            src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(1 +1)+(4 * nt) + 2));
            src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(2 +1)+(4 * nt) + 2));
            src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(3 +1)+(4 * nt) + 2));

            _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), src_temp1);
            _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), src_temp2);
            _mm_storeu_si128((__m128i *)(pu2_dst + (2 * dst_strd)), src_temp3);
            _mm_storeu_si128((__m128i *)(pu2_dst + (3 * dst_strd)), src_temp4);
        }
        else if(nt==8)
        {
            for(col=0;col<(2*nt);col += 8)
            {
                /* pu1_ref[two_nt + col + idx + 1] */
                src_temp1 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(0 +1)+(4 * nt) + col+ 2));
                src_temp2 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(1 +1)+(4 * nt) + col+ 2));
                src_temp3 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(2 +1)+(4 * nt) + col+ 2));
                src_temp4 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(3 +1)+(4 * nt) + col+ 2));
                src_temp5 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(4 +1)+(4 * nt) + col+ 2));
                src_temp6 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(5 +1)+(4 * nt) + col+ 2));
                src_temp7 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(6 +1)+(4 * nt) + col+ 2));
                src_temp8 = _mm_loadu_si128((__m128i *)(pu2_ref-2*(7 +1)+(4 * nt) + col+ 2));

                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (0 * dst_strd)), src_temp1);
                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (1 * dst_strd)), src_temp2);
                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (2 * dst_strd)), src_temp3);
                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (3 * dst_strd)), src_temp4);
                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (4 * dst_strd)), src_temp5);
                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (5 * dst_strd)), src_temp6);
                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (6 * dst_strd)), src_temp7);
                _mm_storeu_si128((__m128i *)(pu2_dst+ col + (7 * dst_strd)), src_temp8);
            }
        }
        else
        {
            __m128i src_temp9, src_temp10, src_temp11, src_temp12, src_temp13, src_temp14, src_temp15, src_temp16;
            for(row = 0; row < nt; row += 8)
            {
                for(col=0;col<(2*nt);col += 16)
                {
                    /* pu1_ref[two_nt + col + idx + 1] */
                    src_temp1  = _mm_loadu_si128((__m128i *)(pu2_ref -2*(0+1)+ (col+0) +(4 * nt) + 2));
                    src_temp9  = _mm_loadu_si128((__m128i *)(pu2_ref -2*(0+1)+ (col+8) +(4 * nt) + 2));
                    src_temp2  = _mm_loadu_si128((__m128i *)(pu2_ref -2*(1+1)+ (col+0) +(4 * nt) + 2));
                    src_temp10 = _mm_loadu_si128((__m128i *)(pu2_ref -2*(1+1)+ (col+8) +(4 * nt) + 2));
                    src_temp3  = _mm_loadu_si128((__m128i *)(pu2_ref -2*(2+1)+ (col+0) +(4 * nt) + 2));
                    src_temp11 = _mm_loadu_si128((__m128i *)(pu2_ref -2*(2+1)+ (col+8) +(4 * nt) + 2));
                    src_temp4  = _mm_loadu_si128((__m128i *)(pu2_ref -2*(3+1)+ (col+0) +(4 * nt) + 2));
                    src_temp12 = _mm_loadu_si128((__m128i *)(pu2_ref -2*(3+1)+ (col+8) +(4 * nt) + 2));

                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (0 * dst_strd)), src_temp1 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (0 * dst_strd)), src_temp9 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (1 * dst_strd)), src_temp2 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (1 * dst_strd)), src_temp10);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (2 * dst_strd)), src_temp3 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (2 * dst_strd)), src_temp11);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (3 * dst_strd)), src_temp4 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (3 * dst_strd)), src_temp12);

                    src_temp5  = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(4+1)+ (col+0) +(4 * nt) + 2));
                    src_temp13 = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(4+1)+ (col+8) +(4 * nt) + 2));
                    src_temp6  = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(5+1)+ (col+0) +(4 * nt) + 2));
                    src_temp14 = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(5+1)+ (col+8) +(4 * nt) + 2));
                    src_temp7  = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(6+1)+ (col+0) +(4 * nt) + 2));
                    src_temp15 = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(6+1)+ (col+8) +(4 * nt) + 2));
                    src_temp8  = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(7+1)+ (col+0) +(4 * nt) + 2));
                    src_temp16 = _mm_loadu_si128((__m128i *)(pu2_ref - 2*(7+1)+ (col+8) +(4 * nt) + 2));

                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (4 * dst_strd)), src_temp5 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (4 * dst_strd)), src_temp13);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (5 * dst_strd)), src_temp6 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (5 * dst_strd)), src_temp14);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (6 * dst_strd)), src_temp7 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (6 * dst_strd)), src_temp15);
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+0) + (7 * dst_strd)), src_temp8 );
                    _mm_storeu_si128((__m128i *)(pu2_dst + (col+8) + (7 * dst_strd)), src_temp16);
                }

                pu2_ref -= 2*8;
                pu2_dst += 8*dst_strd;
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 3 to 9  (positive angle, horizontal mode ) with
* reference  neighboring samples location pointed by 'pu2_ref' to the  TU
* block location pointed by 'pu2_dst'
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_mode_3_to_9_sse42(UWORD16 *pu2_ref,
                                                   WORD32 src_strd,
                                                   UWORD16 *pu2_dst,
                                                   WORD32 dst_strd,
                                                   WORD32 nt,
                                                   WORD32 mode)
{
    WORD32 row, col;
    WORD32 intra_pred_ang;

    __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
    __m128i fract_4x32b, zero_8x16b, intra_pred_ang_4x32b;
    __m128i row_4x32b, two_nt_4x32b, ref_main_idx_4x32b, res_temp5_4x32b, sm1;

    /* Intra Pred Angle according to the mode */
    intra_pred_ang = gai4_ihevc_ang_table[mode];

    /* For the angles other then 45 degree, interpolation btw 2 neighboring */
    /* samples dependent on distance to obtain destination sample */

    sm1 = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY7_HBD[0]);
    const_temp_4x32b  = _mm_set1_epi32(16);
    const_temp2_4x32b = _mm_set1_epi32(31);
    const_temp3_4x32b = _mm_set1_epi16(32);
    const_temp4_4x32b = _mm_set1_epi32(4);

    two_nt_4x32b = _mm_set1_epi32(1);

    zero_8x16b = _mm_set1_epi16(0);
    mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

    /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
    intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

    row_4x32b = _mm_set_epi32(4, 3, 2, 1);

    if(nt ==4)
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        const_temp4_4x32b = _mm_set1_epi16(4);
        two_nt_4x32b = _mm_set1_epi16((4 * nt)- 2);

        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD8  ai1_fract_temp_val[16], ai1_src_temp_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i src_values10;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            ref_main_idx_4x32b = _mm_srai_epi16(res_temp5_4x32b,  5);

            ref_main_idx_4x32b = _mm_add_epi16(ref_main_idx_4x32b,  ref_main_idx_4x32b);

            ref_main_idx_4x32b = _mm_sub_epi16(two_nt_4x32b , ref_main_idx_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);

            _mm_storel_epi64((__m128i *)(ai1_fract_temp_val), fract_4x32b);
            _mm_storel_epi64((__m128i *)(ai1_src_temp_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_set1_epi16(ai1_src_temp_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_src_temp_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_src_temp_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_src_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;
                __m128i src_5_8x16b, src_6_8x16b, src_7_8x16b, src_8_8x16b;

                /* loding 8-bit 16 pixels */
                src_temp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1-8)); /* col=0 */
                src_temp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2-8)); /* col=1 */
                src_temp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3-8)); /* col=2 */
                src_temp8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4-8)); /* col=3 */

                src_5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1-6)); /* col=0 */
                src_6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2-6)); /* col=1 */
                src_7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3-6)); /* col=2 */
                src_8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4-6)); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_5_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_6_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_7_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_8_8x16b, src_temp8_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_5_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_6_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_7_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_8_8x16b, src_temp8_8x16b); /* col=3 */

                /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);   /* col=0 */
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);   /* col=1 */
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);   /* col=2 */
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);   /* col=3 */

                /* converting 16 bit to 8 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                src_temp1_8x16b =_mm_shuffle_epi8(src_temp1_8x16b, sm1);
                src_temp2_8x16b =_mm_shuffle_epi8(src_temp2_8x16b, sm1);
                src_temp3_8x16b =_mm_shuffle_epi8(src_temp3_8x16b, sm1);
                src_temp4_8x16b =_mm_shuffle_epi8(src_temp4_8x16b, sm1);

                src_temp5_8x16b = _mm_unpacklo_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi32(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi32(src_temp3_8x16b, src_temp4_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi64(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi64(src_temp7_8x16b, src_temp8_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 0)), src_temp1_8x16b);             /* row=0 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * (1))), src_temp2_8x16b);       /* row=1 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * (2))), src_temp3_8x16b);       /* row=2 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * (3))), src_temp4_8x16b);       /* row=4 */
            }
        }
    }
    else
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        const_temp4_4x32b = _mm_set1_epi16(8);
        two_nt_4x32b = _mm_set1_epi16((4 * nt)- 2);

        for(col = 0; col < 2*nt; col += 16)
        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD16 pi2_ref_main_idx5, pi2_ref_main_idx6, pi2_ref_main_idx7, pi2_ref_main_idx8;
            WORD8  ai1_fract_temp_val[16], ai1_src_temp_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i fract5_8x16b, fract6_8x16b, fract7_8x16b, fract8_8x16b, src_values10;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i temp11_8x16b, temp12_8x16b, temp13_8x16b, temp14_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            ref_main_idx_4x32b = _mm_srai_epi16(res_temp5_4x32b,  5);

            ref_main_idx_4x32b = _mm_add_epi16(ref_main_idx_4x32b,  ref_main_idx_4x32b);

            ref_main_idx_4x32b = _mm_sub_epi16(two_nt_4x32b , ref_main_idx_4x32b);

            row_4x32b= _mm_add_epi16(row_4x32b, const_temp4_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);

            _mm_storeu_si128((__m128i *)(ai1_fract_temp_val), fract_4x32b);
            _mm_storeu_si128((__m128i *)(ai1_src_temp_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_set1_epi16(ai1_src_temp_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_src_temp_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_src_temp_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_src_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            fract5_8x16b = _mm_set1_epi16(ai1_fract_temp_val[8]);  /* col=5 */
            fract6_8x16b = _mm_set1_epi16(ai1_fract_temp_val[10]);  /* col=6 */
            fract7_8x16b = _mm_set1_epi16(ai1_fract_temp_val[12]);  /* col=7 */
            fract8_8x16b = _mm_set1_epi16(ai1_fract_temp_val[14]);  /* col=8 */

            temp11_8x16b = _mm_set1_epi16(ai1_src_temp_val[8]);  /* col=0 */
            temp12_8x16b = _mm_set1_epi16(ai1_src_temp_val[10]);  /* col=1 */
            temp13_8x16b = _mm_set1_epi16(ai1_src_temp_val[12]);  /* col=2 */
            temp14_8x16b = _mm_set1_epi16(ai1_src_temp_val[14]);  /* col=3 */

            temp11_8x16b = _mm_unpacklo_epi16(temp11_8x16b, fract5_8x16b);
            temp12_8x16b = _mm_unpacklo_epi16(temp12_8x16b, fract6_8x16b);
            temp13_8x16b = _mm_unpacklo_epi16(temp13_8x16b, fract7_8x16b);
            temp14_8x16b = _mm_unpacklo_epi16(temp14_8x16b, fract8_8x16b);

            pi2_ref_main_idx5 = _mm_extract_epi16(ref_main_idx_4x32b, 4);    /* col=5 */
            pi2_ref_main_idx6 = _mm_extract_epi16(ref_main_idx_4x32b, 5);    /* col=6 */
            pi2_ref_main_idx7 = _mm_extract_epi16(ref_main_idx_4x32b, 6);    /* col=7 */
            pi2_ref_main_idx8 = _mm_extract_epi16(ref_main_idx_4x32b, 7);    /* col=8 */

            for(row = 0; row < nt; row += 4)
            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;

                __m128i src_temp11_8x16b, src_temp12_8x16b, src_temp13_8x16b, src_temp14_8x16b;
                __m128i src_temp15_8x16b, src_temp16_8x16b, src_temp17_8x16b, src_temp18_8x16b;

                __m128i src_5_8x16b, src_6_8x16b, src_7_8x16b, src_8_8x16b;

                /* loding 8-bit 16 pixels */
                src_temp5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1-row-(8+row))); /* col=0 */
                src_temp6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2-row-(8+row))); /* col=1 */
                src_temp7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3-row-(8+row))); /* col=2 */
                src_temp8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4-row-(8+row))); /* col=3 */

                src_5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx1-row-(6+row))); /* col=0 */
                src_6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx2-row-(6+row))); /* col=1 */
                src_7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx3-row-(6+row))); /* col=2 */
                src_8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx4-row-(6+row))); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_5_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_6_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_7_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_8_8x16b, src_temp8_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_5_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_6_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_7_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_8_8x16b, src_temp8_8x16b); /* col=3 */

                /* loding 8-bit 16 pixels */
                src_temp15_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx5-row-row-8)); /* col=5 */
                src_temp16_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx6-row-row-8)); /* col=6 */
                src_temp17_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx7-row-row-8)); /* col=7 */
                src_temp18_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx8-row-row-8)); /* col=8 */

                src_5_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx5-row-row-6)); /* col=5 */
                src_6_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx6-row-row-6)); /* col=6 */
                src_7_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx7-row-row-6)); /* col=7 */
                src_8_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+pi2_ref_main_idx8-row-row-6)); /* col=8 */

                src_temp11_8x16b =  _mm_unpacklo_epi16(src_5_8x16b, src_temp15_8x16b); /* col=5 */
                src_temp12_8x16b =  _mm_unpacklo_epi16(src_6_8x16b, src_temp16_8x16b); /* col=6 */
                src_temp13_8x16b =  _mm_unpacklo_epi16(src_7_8x16b, src_temp17_8x16b); /* col=7 */
                src_temp14_8x16b =  _mm_unpacklo_epi16(src_8_8x16b, src_temp18_8x16b); /* col=8 */

                src_temp15_8x16b =  _mm_unpackhi_epi16(src_5_8x16b, src_temp15_8x16b); /* col=5 */
                src_temp16_8x16b =  _mm_unpackhi_epi16(src_6_8x16b, src_temp16_8x16b); /* col=6 */
                src_temp17_8x16b =  _mm_unpackhi_epi16(src_7_8x16b, src_temp17_8x16b); /* col=7 */
                src_temp18_8x16b =  _mm_unpackhi_epi16(src_8_8x16b, src_temp18_8x16b); /* col=8 */

                /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
                src_temp11_8x16b = _mm_madd_epi16(src_temp11_8x16b, temp11_8x16b);
                src_temp12_8x16b = _mm_madd_epi16(src_temp12_8x16b, temp12_8x16b);
                src_temp13_8x16b = _mm_madd_epi16(src_temp13_8x16b, temp13_8x16b);
                src_temp14_8x16b = _mm_madd_epi16(src_temp14_8x16b, temp14_8x16b);

                src_temp15_8x16b = _mm_madd_epi16(src_temp15_8x16b, temp11_8x16b);
                src_temp16_8x16b = _mm_madd_epi16(src_temp16_8x16b, temp12_8x16b);
                src_temp17_8x16b = _mm_madd_epi16(src_temp17_8x16b, temp13_8x16b);
                src_temp18_8x16b = _mm_madd_epi16(src_temp18_8x16b, temp14_8x16b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);   /* col=0 */
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);   /* col=1 */
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);   /* col=2 */
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);   /* col=3 */

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
                src_temp11_8x16b = _mm_add_epi32(src_temp11_8x16b, const_temp_4x32b);
                src_temp12_8x16b = _mm_add_epi32(src_temp12_8x16b, const_temp_4x32b);
                src_temp13_8x16b = _mm_add_epi32(src_temp13_8x16b, const_temp_4x32b);
                src_temp14_8x16b = _mm_add_epi32(src_temp14_8x16b, const_temp_4x32b);

                src_temp15_8x16b = _mm_add_epi32(src_temp15_8x16b, const_temp_4x32b);
                src_temp16_8x16b = _mm_add_epi32(src_temp16_8x16b, const_temp_4x32b);
                src_temp17_8x16b = _mm_add_epi32(src_temp17_8x16b, const_temp_4x32b);
                src_temp18_8x16b = _mm_add_epi32(src_temp18_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp11_8x16b = _mm_srai_epi32(src_temp11_8x16b,  5);   /* col=5 */
                src_temp12_8x16b = _mm_srai_epi32(src_temp12_8x16b,  5);   /* col=6 */
                src_temp13_8x16b = _mm_srai_epi32(src_temp13_8x16b,  5);   /* col=7 */
                src_temp14_8x16b = _mm_srai_epi32(src_temp14_8x16b,  5);   /* col=8 */

                src_temp15_8x16b = _mm_srai_epi32(src_temp15_8x16b,  5);   /* col=5 */
                src_temp16_8x16b = _mm_srai_epi32(src_temp16_8x16b,  5);   /* col=6 */
                src_temp17_8x16b = _mm_srai_epi32(src_temp17_8x16b,  5);   /* col=7 */
                src_temp18_8x16b = _mm_srai_epi32(src_temp18_8x16b,  5);   /* col=8 */

                /* converting 16 bit to 8 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                src_temp1_8x16b =_mm_shuffle_epi8(src_temp1_8x16b, sm1);
                src_temp2_8x16b =_mm_shuffle_epi8(src_temp2_8x16b, sm1);
                src_temp3_8x16b =_mm_shuffle_epi8(src_temp3_8x16b, sm1);
                src_temp4_8x16b =_mm_shuffle_epi8(src_temp4_8x16b, sm1);

                /* converting 16 bit to 8 bit */
                src_temp11_8x16b = _mm_packus_epi32(src_temp11_8x16b, src_temp15_8x16b); /* col=5 */
                src_temp12_8x16b = _mm_packus_epi32(src_temp12_8x16b, src_temp16_8x16b); /* col=6 */
                src_temp13_8x16b = _mm_packus_epi32(src_temp13_8x16b, src_temp17_8x16b); /* col=7 */
                src_temp14_8x16b = _mm_packus_epi32(src_temp14_8x16b, src_temp18_8x16b); /* col=8 */

                src_temp11_8x16b =_mm_shuffle_epi8(src_temp11_8x16b, sm1);
                src_temp12_8x16b =_mm_shuffle_epi8(src_temp12_8x16b, sm1);
                src_temp13_8x16b =_mm_shuffle_epi8(src_temp13_8x16b, sm1);
                src_temp14_8x16b =_mm_shuffle_epi8(src_temp14_8x16b, sm1);

                src_temp5_8x16b = _mm_unpacklo_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi32(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi32(src_temp3_8x16b, src_temp4_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi64(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi64(src_temp7_8x16b, src_temp8_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi32(src_temp11_8x16b, src_temp12_8x16b);
                src_temp16_8x16b = _mm_unpacklo_epi32(src_temp13_8x16b, src_temp14_8x16b);
                src_temp17_8x16b = _mm_unpackhi_epi32(src_temp11_8x16b, src_temp12_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi32(src_temp13_8x16b, src_temp14_8x16b);

                src_temp11_8x16b = _mm_unpacklo_epi64(src_temp15_8x16b, src_temp16_8x16b);
                src_temp12_8x16b = _mm_unpackhi_epi64(src_temp15_8x16b, src_temp16_8x16b);
                src_temp13_8x16b = _mm_unpacklo_epi64(src_temp17_8x16b, src_temp18_8x16b);
                src_temp14_8x16b = _mm_unpackhi_epi64(src_temp17_8x16b, src_temp18_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * row)),    src_temp1_8x16b);          /* row=0 */
                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+1))), src_temp2_8x16b);       /* row=1 */
                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+2))), src_temp3_8x16b);       /* row=2 */
                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+3))), src_temp4_8x16b);       /* row=4 */

                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * row)),    src_temp11_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * (row+1))), src_temp12_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * (row+2))), src_temp13_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * (row+3))), src_temp14_8x16b);
            }
        }
    }
}
/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 11 to 17  (negative angle, horizontal mode )
* with reference  neighboring samples location pointed by 'pu2_ref' to the
* TU block location pointed by 'pu2_dst'
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_mode_11_to_17_sse42(UWORD16 *pu2_ref,
                                                     WORD32 src_strd,
                                                     UWORD16 *pu2_dst,
                                                     WORD32 dst_strd,
                                                     WORD32 nt,
                                                     WORD32 mode)
{
    /* This function and ihevc_intra_pred_CHROMA_mode_19_to_25 are same except */
    /* for ref main & side samples assignment,can be combined for */
    /* optimzation */

    WORD32 row, col, k;
    WORD32 intra_pred_ang, inv_ang, inv_ang_sum;
    WORD32 ref_idx;

    __m128i const_temp_4x32b, const_temp2_4x32b, const_temp3_4x32b, const_temp4_4x32b, mask_4x32b;
    __m128i fract_4x32b, zero_8x16b, intra_pred_ang_4x32b;
    __m128i row_4x32b, two_nt_4x32b, ref_main_idx_4x32b, res_temp5_4x32b;

    UWORD16 ref_temp[2 * MAX_CU_SIZE + 2];
    UWORD16 *ref_main;
    inv_ang_sum = 128;

    intra_pred_ang = gai4_ihevc_ang_table[mode];

    inv_ang = gai4_ihevc_inv_ang_table[mode - 11];
    /* Intermediate reference samples for negative angle modes */
    /* This have to be removed during optimization */

    /* For horizontal modes, (ref main = ref left) (ref side = ref above) */

    ref_main = ref_temp + 2*nt;
    for(k = 0; k < (2 * (nt + 1)); k += 2)
    {
        ref_temp[k + (2* (nt - 1))] = pu2_ref[(4 * nt) - k];
        ref_temp[k + 1+ (2* (nt - 1))] = pu2_ref[(4 * nt) - k + 1];
    }

    ref_main = ref_temp + (2 * (nt - 1));
    ref_idx = (nt * intra_pred_ang) >> 5;

    /* SIMD Optimization can be done using look-up table for the loop */
    /* For negative angled derive the main reference samples from side */
    /* reference samples refer to section 8.4.4.2.6 */

    for(k = -2; k > (2 * ref_idx); k -= 2)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[(4 * nt) + ((inv_ang_sum >> 8) << 1)];
        ref_main[k + 1] = pu2_ref[((4 * nt) + 1) + ((inv_ang_sum >> 8) << 1)];
    }

    /* For the angles other then 45 degree, interpolation btw 2 neighboring */
    /* samples dependent on distance to obtain destination sample */

    const_temp_4x32b  = _mm_set1_epi32(16);
    const_temp3_4x32b = _mm_set1_epi16(32);

    two_nt_4x32b = _mm_set1_epi32(1);

    zero_8x16b = _mm_set1_epi16(0);
    mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

    /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
    intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

    row_4x32b = _mm_set_epi32(4, 3, 2, 1);

    if(nt ==4)
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        two_nt_4x32b = _mm_set1_epi16(1);

        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD8  ai1_fract_temp_val[16], ai1_src_temp_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i src_values10;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            ref_main_idx_4x32b = _mm_add_epi16(two_nt_4x32b , _mm_srai_epi16(res_temp5_4x32b,  5));
            ref_main_idx_4x32b = _mm_add_epi16(ref_main_idx_4x32b , ref_main_idx_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);

            _mm_storel_epi64((__m128i *)(ai1_fract_temp_val), fract_4x32b);
            _mm_storel_epi64((__m128i *)(ai1_src_temp_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_set1_epi16(ai1_src_temp_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_src_temp_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_src_temp_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_src_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;
                __m128i src_5_8x16b, src_6_8x16b, src_7_8x16b, src_8_8x16b;

                /* loding 16-bit 8 pixels */
                src_temp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1)); /* col=0 */
                src_temp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2)); /* col=1 */
                src_temp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3)); /* col=2 */
                src_temp8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4)); /* col=3 */

                src_5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1+2)); /* col=0 */
                src_6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2+2)); /* col=1 */
                src_7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3+2)); /* col=2 */
                src_8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4+2)); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_temp5_8x16b, src_5_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_temp6_8x16b, src_6_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_temp7_8x16b, src_7_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_temp8_8x16b, src_8_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_temp5_8x16b, src_5_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_temp6_8x16b, src_6_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_temp7_8x16b, src_7_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_temp8_8x16b, src_8_8x16b); /* col=3 */

                /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);   /* col=0 */
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);   /* col=1 */
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);   /* col=2 */
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);   /* col=3 */

                /* converting 16 bit to 8 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                src_temp5_8x16b = _mm_unpacklo_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi32(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi32(src_temp3_8x16b, src_temp4_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi64(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi64(src_temp7_8x16b, src_temp8_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * 0)), src_temp1_8x16b);    /* row=0 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * (1))), src_temp2_8x16b);  /* row=1 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * (2))), src_temp3_8x16b);  /* row=2 */

                _mm_storeu_si128((__m128i *)(pu2_dst+(dst_strd * (3))), src_temp4_8x16b);  /* row=3 */

            }
        }
    }
    else
    {
        intra_pred_ang_4x32b = _mm_set1_epi16(intra_pred_ang);
        row_4x32b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        const_temp2_4x32b = _mm_set1_epi16(31);
        const_temp4_4x32b = _mm_set1_epi16(8);
        two_nt_4x32b = _mm_set1_epi16(1);

        for(col = 0; col < 2*nt; col += 16)
        {
            WORD16 pi2_ref_main_idx1, pi2_ref_main_idx2, pi2_ref_main_idx3, pi2_ref_main_idx4;
            WORD16 pi2_ref_main_idx5, pi2_ref_main_idx6, pi2_ref_main_idx7, pi2_ref_main_idx8;
            WORD8  ai1_fract_temp_val[16], ai1_src_temp_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b;
            __m128i fract5_8x16b, fract6_8x16b, fract7_8x16b, fract8_8x16b, src_values10;

            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i temp11_8x16b, temp12_8x16b, temp13_8x16b, temp14_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi16(row_4x32b, intra_pred_ang_4x32b);

            /* fract = pos & (31); */
            fract_4x32b= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            ref_main_idx_4x32b = _mm_add_epi16(two_nt_4x32b , _mm_srai_epi16(res_temp5_4x32b,  5));
            ref_main_idx_4x32b = _mm_add_epi16(ref_main_idx_4x32b , ref_main_idx_4x32b);

            row_4x32b= _mm_add_epi16(row_4x32b, const_temp4_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi16(const_temp3_4x32b , fract_4x32b);

            _mm_storeu_si128((__m128i *)(ai1_fract_temp_val), fract_4x32b);
            _mm_storeu_si128((__m128i *)(ai1_src_temp_val),  src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_fract_temp_val[0]);  /* col=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_fract_temp_val[2]);  /* col=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_fract_temp_val[4]);  /* col=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_fract_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_set1_epi16(ai1_src_temp_val[0]);  /* col=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_src_temp_val[2]);  /* col=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_src_temp_val[4]);  /* col=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_src_temp_val[6]);  /* col=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            pi2_ref_main_idx1 = _mm_extract_epi16(ref_main_idx_4x32b, 0);    /* col=0 */
            pi2_ref_main_idx2 = _mm_extract_epi16(ref_main_idx_4x32b, 1);    /* col=1 */
            pi2_ref_main_idx3 = _mm_extract_epi16(ref_main_idx_4x32b, 2);    /* col=2 */
            pi2_ref_main_idx4 = _mm_extract_epi16(ref_main_idx_4x32b, 3);    /* col=3 */

            fract5_8x16b = _mm_set1_epi16(ai1_fract_temp_val[8]);  /* col=5 */
            fract6_8x16b = _mm_set1_epi16(ai1_fract_temp_val[10]);  /* col=6 */
            fract7_8x16b = _mm_set1_epi16(ai1_fract_temp_val[12]);  /* col=7 */
            fract8_8x16b = _mm_set1_epi16(ai1_fract_temp_val[14]);  /* col=8 */

            temp11_8x16b = _mm_set1_epi16(ai1_src_temp_val[8]);  /* col=0 */
            temp12_8x16b = _mm_set1_epi16(ai1_src_temp_val[10]);  /* col=1 */
            temp13_8x16b = _mm_set1_epi16(ai1_src_temp_val[12]);  /* col=2 */
            temp14_8x16b = _mm_set1_epi16(ai1_src_temp_val[14]);  /* col=3 */

            temp11_8x16b = _mm_unpacklo_epi16(temp11_8x16b, fract5_8x16b);
            temp12_8x16b = _mm_unpacklo_epi16(temp12_8x16b, fract6_8x16b);
            temp13_8x16b = _mm_unpacklo_epi16(temp13_8x16b, fract7_8x16b);
            temp14_8x16b = _mm_unpacklo_epi16(temp14_8x16b, fract8_8x16b);

            pi2_ref_main_idx5 = _mm_extract_epi16(ref_main_idx_4x32b, 4);    /* col=5 */
            pi2_ref_main_idx6 = _mm_extract_epi16(ref_main_idx_4x32b, 5);    /* col=6 */
            pi2_ref_main_idx7 = _mm_extract_epi16(ref_main_idx_4x32b, 6);    /* col=7 */
            pi2_ref_main_idx8 = _mm_extract_epi16(ref_main_idx_4x32b, 7);    /* col=8 */

            for(row = 0; row < nt; row += 4)
            {
                __m128i src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b, src_temp4_8x16b;
                __m128i src_temp5_8x16b, src_temp6_8x16b, src_temp7_8x16b, src_temp8_8x16b;

                __m128i src_temp11_8x16b, src_temp12_8x16b, src_temp13_8x16b, src_temp14_8x16b;
                __m128i src_temp15_8x16b, src_temp16_8x16b, src_temp17_8x16b, src_temp18_8x16b;

                __m128i src_5_8x16b, src_6_8x16b, src_7_8x16b, src_8_8x16b;

                /* loding 16-bit 8 pixels */
                src_temp5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1+row+row)); /* col=0 */
                src_temp6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2+row+row)); /* col=1 */
                src_temp7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3+row+row)); /* col=2 */
                src_temp8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4+row+row)); /* col=3 */

                src_5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx1+row+row+2)); /* col=0 */
                src_6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx2+row+row+2)); /* col=1 */
                src_7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx3+row+row+2)); /* col=2 */
                src_8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx4+row+row+2)); /* col=3 */

                src_temp1_8x16b =  _mm_unpacklo_epi16(src_temp5_8x16b, src_5_8x16b); /* col=0 */
                src_temp2_8x16b =  _mm_unpacklo_epi16(src_temp6_8x16b, src_6_8x16b); /* col=1 */
                src_temp3_8x16b =  _mm_unpacklo_epi16(src_temp7_8x16b, src_7_8x16b); /* col=2 */
                src_temp4_8x16b =  _mm_unpacklo_epi16(src_temp8_8x16b, src_8_8x16b); /* col=3 */

                src_temp5_8x16b =  _mm_unpackhi_epi16(src_temp5_8x16b, src_5_8x16b); /* col=0 */
                src_temp6_8x16b =  _mm_unpackhi_epi16(src_temp6_8x16b, src_6_8x16b); /* col=1 */
                src_temp7_8x16b =  _mm_unpackhi_epi16(src_temp7_8x16b, src_7_8x16b); /* col=2 */
                src_temp8_8x16b =  _mm_unpackhi_epi16(src_temp8_8x16b, src_8_8x16b); /* col=3 */

                /* loding 8-bit 16 pixels */
                src_temp15_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx5+row+row)); /* col=5 */
                src_temp16_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx6+row+row)); /* col=6 */
                src_temp17_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx7+row+row)); /* col=7 */
                src_temp18_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx8+row+row)); /* col=8 */

                src_5_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx5+row+row+2)); /* col=5 */
                src_6_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx6+row+row+2)); /* col=6 */
                src_7_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx7+row+row+2)); /* col=7 */
                src_8_8x16b = _mm_loadu_si128((__m128i *)(ref_main+pi2_ref_main_idx8+row+row+2)); /* col=8 */

                src_temp11_8x16b =  _mm_unpacklo_epi16(src_temp15_8x16b, src_5_8x16b); /* col=0 */
                src_temp12_8x16b =  _mm_unpacklo_epi16(src_temp16_8x16b, src_6_8x16b); /* col=1 */
                src_temp13_8x16b =  _mm_unpacklo_epi16(src_temp17_8x16b, src_7_8x16b); /* col=2 */
                src_temp14_8x16b =  _mm_unpacklo_epi16(src_temp18_8x16b, src_8_8x16b); /* col=3 */

                src_temp15_8x16b =  _mm_unpackhi_epi16(src_temp15_8x16b, src_5_8x16b); /* col=0 */
                src_temp16_8x16b =  _mm_unpackhi_epi16(src_temp16_8x16b, src_6_8x16b); /* col=1 */
                src_temp17_8x16b =  _mm_unpackhi_epi16(src_temp17_8x16b, src_7_8x16b); /* col=2 */
                src_temp18_8x16b =  _mm_unpackhi_epi16(src_temp18_8x16b, src_8_8x16b); /* col=3 */

                /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
                src_temp1_8x16b = _mm_madd_epi16(src_temp1_8x16b, temp1_8x16b);
                src_temp2_8x16b = _mm_madd_epi16(src_temp2_8x16b, temp2_8x16b);
                src_temp3_8x16b = _mm_madd_epi16(src_temp3_8x16b, temp3_8x16b);
                src_temp4_8x16b = _mm_madd_epi16(src_temp4_8x16b, temp4_8x16b);

                src_temp5_8x16b = _mm_madd_epi16(src_temp5_8x16b, temp1_8x16b);
                src_temp6_8x16b = _mm_madd_epi16(src_temp6_8x16b, temp2_8x16b);
                src_temp7_8x16b = _mm_madd_epi16(src_temp7_8x16b, temp3_8x16b);
                src_temp8_8x16b = _mm_madd_epi16(src_temp8_8x16b, temp4_8x16b);

                /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
                src_temp11_8x16b = _mm_madd_epi16(src_temp11_8x16b, temp11_8x16b);
                src_temp12_8x16b = _mm_madd_epi16(src_temp12_8x16b, temp12_8x16b);
                src_temp13_8x16b = _mm_madd_epi16(src_temp13_8x16b, temp13_8x16b);
                src_temp14_8x16b = _mm_madd_epi16(src_temp14_8x16b, temp14_8x16b);

                src_temp15_8x16b = _mm_madd_epi16(src_temp15_8x16b, temp11_8x16b);
                src_temp16_8x16b = _mm_madd_epi16(src_temp16_8x16b, temp12_8x16b);
                src_temp17_8x16b = _mm_madd_epi16(src_temp17_8x16b, temp13_8x16b);
                src_temp18_8x16b = _mm_madd_epi16(src_temp18_8x16b, temp14_8x16b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
                src_temp1_8x16b = _mm_add_epi32(src_temp1_8x16b, const_temp_4x32b);
                src_temp2_8x16b = _mm_add_epi32(src_temp2_8x16b, const_temp_4x32b);
                src_temp3_8x16b = _mm_add_epi32(src_temp3_8x16b, const_temp_4x32b);
                src_temp4_8x16b = _mm_add_epi32(src_temp4_8x16b, const_temp_4x32b);

                src_temp5_8x16b = _mm_add_epi32(src_temp5_8x16b, const_temp_4x32b);
                src_temp6_8x16b = _mm_add_epi32(src_temp6_8x16b, const_temp_4x32b);
                src_temp7_8x16b = _mm_add_epi32(src_temp7_8x16b, const_temp_4x32b);
                src_temp8_8x16b = _mm_add_epi32(src_temp8_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp1_8x16b = _mm_srai_epi32(src_temp1_8x16b,  5);   /* col=0 */
                src_temp2_8x16b = _mm_srai_epi32(src_temp2_8x16b,  5);   /* col=1 */
                src_temp3_8x16b = _mm_srai_epi32(src_temp3_8x16b,  5);   /* col=2 */
                src_temp4_8x16b = _mm_srai_epi32(src_temp4_8x16b,  5);   /* col=3 */

                src_temp5_8x16b = _mm_srai_epi32(src_temp5_8x16b,  5);   /* col=0 */
                src_temp6_8x16b = _mm_srai_epi32(src_temp6_8x16b,  5);   /* col=1 */
                src_temp7_8x16b = _mm_srai_epi32(src_temp7_8x16b,  5);   /* col=2 */
                src_temp8_8x16b = _mm_srai_epi32(src_temp8_8x16b,  5);   /* col=3 */

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
                src_temp11_8x16b = _mm_add_epi32(src_temp11_8x16b, const_temp_4x32b);
                src_temp12_8x16b = _mm_add_epi32(src_temp12_8x16b, const_temp_4x32b);
                src_temp13_8x16b = _mm_add_epi32(src_temp13_8x16b, const_temp_4x32b);
                src_temp14_8x16b = _mm_add_epi32(src_temp14_8x16b, const_temp_4x32b);

                src_temp15_8x16b = _mm_add_epi32(src_temp15_8x16b, const_temp_4x32b);
                src_temp16_8x16b = _mm_add_epi32(src_temp16_8x16b, const_temp_4x32b);
                src_temp17_8x16b = _mm_add_epi32(src_temp17_8x16b, const_temp_4x32b);
                src_temp18_8x16b = _mm_add_epi32(src_temp18_8x16b, const_temp_4x32b);

                /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
                src_temp11_8x16b = _mm_srai_epi32(src_temp11_8x16b,  5);   /* col=5 */
                src_temp12_8x16b = _mm_srai_epi32(src_temp12_8x16b,  5);   /* col=6 */
                src_temp13_8x16b = _mm_srai_epi32(src_temp13_8x16b,  5);   /* col=7 */
                src_temp14_8x16b = _mm_srai_epi32(src_temp14_8x16b,  5);   /* col=8 */

                src_temp15_8x16b = _mm_srai_epi32(src_temp15_8x16b,  5);   /* col=5 */
                src_temp16_8x16b = _mm_srai_epi32(src_temp16_8x16b,  5);   /* col=6 */
                src_temp17_8x16b = _mm_srai_epi32(src_temp17_8x16b,  5);   /* col=7 */
                src_temp18_8x16b = _mm_srai_epi32(src_temp18_8x16b,  5);   /* col=8 */

                /* converting 32 bit to 16 bit */
                src_temp1_8x16b = _mm_packus_epi32(src_temp1_8x16b, src_temp5_8x16b); /* col=0 */
                src_temp2_8x16b = _mm_packus_epi32(src_temp2_8x16b, src_temp6_8x16b); /* col=1 */
                src_temp3_8x16b = _mm_packus_epi32(src_temp3_8x16b, src_temp7_8x16b); /* col=2 */
                src_temp4_8x16b = _mm_packus_epi32(src_temp4_8x16b, src_temp8_8x16b); /* col=3 */

                /* converting 16 bit to 8 bit */
                src_temp11_8x16b = _mm_packus_epi32(src_temp11_8x16b, src_temp15_8x16b); /* col=5 */
                src_temp12_8x16b = _mm_packus_epi32(src_temp12_8x16b, src_temp16_8x16b); /* col=6 */
                src_temp13_8x16b = _mm_packus_epi32(src_temp13_8x16b, src_temp17_8x16b); /* col=7 */
                src_temp14_8x16b = _mm_packus_epi32(src_temp14_8x16b, src_temp18_8x16b); /* col=8 */

                src_temp5_8x16b = _mm_unpacklo_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp6_8x16b = _mm_unpacklo_epi32(src_temp3_8x16b, src_temp4_8x16b);
                src_temp7_8x16b = _mm_unpackhi_epi32(src_temp1_8x16b, src_temp2_8x16b);
                src_temp8_8x16b = _mm_unpackhi_epi32(src_temp3_8x16b, src_temp4_8x16b);

                src_temp1_8x16b = _mm_unpacklo_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp2_8x16b = _mm_unpackhi_epi64(src_temp5_8x16b, src_temp6_8x16b);
                src_temp3_8x16b = _mm_unpacklo_epi64(src_temp7_8x16b, src_temp8_8x16b);
                src_temp4_8x16b = _mm_unpackhi_epi64(src_temp7_8x16b, src_temp8_8x16b);

                src_temp15_8x16b = _mm_unpacklo_epi32(src_temp11_8x16b, src_temp12_8x16b);
                src_temp16_8x16b = _mm_unpacklo_epi32(src_temp13_8x16b, src_temp14_8x16b);
                src_temp17_8x16b = _mm_unpackhi_epi32(src_temp11_8x16b, src_temp12_8x16b);
                src_temp18_8x16b = _mm_unpackhi_epi32(src_temp13_8x16b, src_temp14_8x16b);

                src_temp11_8x16b = _mm_unpacklo_epi64(src_temp15_8x16b, src_temp16_8x16b);
                src_temp12_8x16b = _mm_unpackhi_epi64(src_temp15_8x16b, src_temp16_8x16b);
                src_temp13_8x16b = _mm_unpacklo_epi64(src_temp17_8x16b, src_temp18_8x16b);
                src_temp14_8x16b = _mm_unpackhi_epi64(src_temp17_8x16b, src_temp18_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * row)),    src_temp1_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+1))), src_temp2_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+2))), src_temp3_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst+col+(dst_strd * (row+3))), src_temp4_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * row)),    src_temp11_8x16b);          /* row=0 */
                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * (row+1))), src_temp12_8x16b);       /* row=1 */
                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * (row+2))), src_temp13_8x16b);       /* row=2 */
                _mm_storeu_si128((__m128i *)(pu2_dst+col+8+(dst_strd * (row+3))), src_temp14_8x16b);       /* row=4 */
            }
        }
    }
}
/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 19 to 25  (negative angle, vertical mode ) with
* reference  neighboring samples location pointed by 'pu2_ref' to the  TU
* block location pointed by 'pu2_dst'
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_mode_19_to_25_sse42(UWORD16 *pu2_ref,
                                                     WORD32 src_strd,
                                                     UWORD16 *pu2_dst,
                                                     WORD32 dst_strd,
                                                     WORD32 nt,
                                                     WORD32 mode)
{
    WORD32 row, k;
    WORD32 intra_pred_ang, idx;
    WORD32 inv_ang, inv_ang_sum, pos, fract;
    WORD32 ref_main_idx, ref_idx;
    UWORD16 ref_temp[(2 * MAX_CU_SIZE) + 2];
    UWORD16 *ref_main;

    __m128i zero_8x16b, fract_8x16b, const_temp_8x16b;

    if(nt > 16)
    {
        ihevc_hbd_intra_pred_chroma_mode_19_to_25(pu2_ref, src_strd, pu2_dst, dst_strd, nt, mode);
        return;
    }

    intra_pred_ang = gai4_ihevc_ang_table_chroma[mode];
    inv_ang = gai4_ihevc_inv_ang_table_chroma[mode - 12];

    /* Intermediate reference samples for negative angle modes */
    /* This have to be removed during optimization */
    /* For horizontal modes, (ref main = ref above) (ref side = ref left) */
    ref_main = ref_temp + 2*nt;
    for(k = 0; k < (2 * (nt + 1)); k += 2)
    {
        ref_temp[k + (2* (nt - 1))] = pu2_ref[(4 * nt) + k];
        ref_temp[k + 1 + (2* (nt - 1))] = pu2_ref[(4 * nt) + k + 1];
    }

    ref_idx = (nt * intra_pred_ang) >> 5;
    inv_ang_sum = 128;
    ref_main = ref_temp + (2 *( nt - 1));
    /* SIMD Optimization can be done using look-up table for the loop */
    /* For negative angled derive the main reference samples from side */
    /* reference samples refer to section 8.4.4.2.6 */
    for(k = -2; k > (2 * ref_idx) ; k -= 2)
    {
        inv_ang_sum += inv_ang;
        ref_main[k] = pu2_ref[(4 * nt) - (inv_ang_sum >> 8) * 2];
        ref_main[k + 1] = pu2_ref[((4 * nt) + 1) - (inv_ang_sum >> 8) * 2];
    }

    const_temp_8x16b = _mm_set1_epi32(16);

    if(nt==4) /* if nt =4 */
    {
        __m128i const_temp2_4x32b, const_temp3_4x32b, mask_4x32b;
        __m128i src_values10, src_values11, zero_8x16b, intra_pred_ang_4x32b;
        __m128i row_4x32b, two_nt_4x32b, src_values12;

        const_temp2_4x32b = _mm_set1_epi32(31);
        const_temp3_4x32b = _mm_set1_epi32(32);

        two_nt_4x32b = _mm_set1_epi32(2);

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
        intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

        row_4x32b = _mm_set_epi32(4, 3, 2, 1);
        {
            WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;
            WORD8  ai1_src_temp0_val[16], ai1_src_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b, res_temp5_4x32b;
            __m128i src_values0, src_values1, src_values2, src_values3, src_values13;
            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;
            __m128i src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

            src_values12 = _mm_add_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));
            src_values12 = _mm_add_epi32(src_values12 , _mm_srai_epi32(res_temp5_4x32b,  5));

            ref_main_temp0 = _mm_srli_si128(src_values12 , 4);  /* next 32 bit values */
            ref_main_temp1 = _mm_srli_si128(src_values12 , 8);  /* next 32 bit values */
            ref_main_temp2 = _mm_srli_si128(src_values12 , 12); /* next 32 bit values */
            ref_main_idx1  = _mm_cvtsi128_si32(src_values12);    /* row=0 */
            ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* row=1 */
            ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* row=2 */
            ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* row=3 */

            /* fract = pos & (31); */
            src_values11= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi32(const_temp3_4x32b , src_values11);

            _mm_storeu_si128((__m128i *)(ai1_src_temp1_val), src_values11);
            _mm_storeu_si128((__m128i *)(ai1_src_temp0_val), src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_src_temp1_val[0]);  /* row=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_src_temp1_val[4]);  /* row=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_src_temp1_val[8]);  /* row=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_src_temp1_val[12]);  /* row=3 */

            temp1_8x16b = _mm_set1_epi16(ai1_src_temp0_val[0]);  /* row=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_src_temp0_val[4]);  /* row=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_src_temp0_val[8]);  /* row=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_src_temp0_val[12]);  /* row=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            // inner loop starts from here
            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1));  /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx2));  /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx3));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx4));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+2));
            src_values11 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx2+2));
            src_values12 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx3+2));
            src_values13 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx4+2));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp1_8x16b);
            src_values1 = _mm_madd_epi16(src_values1, temp2_8x16b);
            src_values2 = _mm_madd_epi16(src_values2, temp3_8x16b);
            src_values3 = _mm_madd_epi16(src_values3, temp4_8x16b);

            src_values10 = _mm_madd_epi16(src_values10, temp1_8x16b);
            src_values11 = _mm_madd_epi16(src_values11, temp2_8x16b);
            src_values12 = _mm_madd_epi16(src_values12, temp3_8x16b);
            src_values13 = _mm_madd_epi16(src_values13, temp4_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd)), src_values0);       /* row=0 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), src_values1);   /* row=1 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), src_values2);   /* row=2 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), src_values3);   /* row=3 */
        }
    }
    else if(nt == 8) /* for nt = 16 case */
    {
        WORD32 ref_main_idx1 , fract1, temp, temp1;
        __m128i fract1_8x16b, temp_8x16b, temp1_8x16b;

        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 2)
        {
            __m128i src_values0, src_values1, src_values2, src_values3;
            __m128i  src_values10, src_values11, src_values12, src_values13;
            __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp= 32-fract;
            ref_main_idx = 2 * idx + 2; /* col from 0-15 */

            pos = ((row + 2) * intra_pred_ang);
            idx = pos >> 5;
            fract1 = pos & (31);
            temp1= 32-fract1;
            ref_main_idx1 = 2 * idx + 2; /* col from 0-15 */

            fract_8x16b  = _mm_set1_epi16(fract);
            fract1_8x16b = _mm_set1_epi16(fract1);
            temp_8x16b   = _mm_set1_epi16(temp);
            temp1_8x16b  = _mm_set1_epi16(temp1);

            temp_8x16b = _mm_unpacklo_epi16(temp_8x16b, fract_8x16b);
            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);

            /* row=0 */
            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+8));   /* col = 8-15 */

            /* row=1 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1));   /* col = 0-7 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+8));  /* col = 8-15 */

            src_values10 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+2));
            src_values11 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+8+2));
            src_values12 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+2));
            src_values13 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx1+8+2));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp_8x16b);
            src_values1 = _mm_madd_epi16(src_values1, temp_8x16b);

            src_values2 = _mm_madd_epi16(src_values2, temp1_8x16b);
            src_values3 = _mm_madd_epi16(src_values3, temp1_8x16b);

            src_values10 = _mm_madd_epi16(src_values10, temp_8x16b);
            src_values11 = _mm_madd_epi16(src_values11, temp_8x16b);

            src_values12 = _mm_madd_epi16(src_values12, temp1_8x16b);
            src_values13 = _mm_madd_epi16(src_values13, temp1_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);

            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);

            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);
            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);

            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);

            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);

            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 8-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_values0);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_values1);

            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd), src_values2);
            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd+8), src_values3);

            pu2_dst += 2*dst_strd;
        }
    }
    else if(nt==16)
    {
        WORD32 temp;
        /* unroll the col loop (inner) */
        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 1)
        {
            __m128i  src_values0, src_values1, src_values2, src_values3, temp_8x16b;
            __m128i  src_values10, src_values11, src_values12, src_values13;
            __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp = 32-fract;
            ref_main_idx =2 * idx + 2; /* col from 0-31 */

            fract_8x16b = _mm_set1_epi16(fract);
            temp_8x16b  = _mm_set1_epi16(temp);

            temp_8x16b = _mm_unpacklo_epi16(temp_8x16b, fract_8x16b);

            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+8));   /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+16));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+24));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+2));
            src_values11 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+8+2));
            src_values12 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+16+2));
            src_values13 = _mm_loadu_si128((__m128i *)(ref_main+ref_main_idx+24+2));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
            src_values0 = _mm_madd_epi16(src_values0, temp_8x16b);
            src_values1 = _mm_madd_epi16(src_values1, temp_8x16b);
            src_values2 = _mm_madd_epi16(src_values2, temp_8x16b);
            src_values3 = _mm_madd_epi16(src_values3, temp_8x16b);

            src_values10 = _mm_madd_epi16(src_values10, temp_8x16b);
            src_values11 = _mm_madd_epi16(src_values11, temp_8x16b);
            src_values12 = _mm_madd_epi16(src_values12, temp_8x16b);
            src_values13 = _mm_madd_epi16(src_values13, temp_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 16-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_values0);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_values1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16), src_values2);
            _mm_storeu_si128((__m128i *)(pu2_dst+24), src_values3);

            pu2_dst += dst_strd;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*  Intraprediction for mode 27 to 33  (positive angle, vertical mode ) with
* reference  neighboring samples location pointed by 'pu2_ref' to the  TU
* block location pointed by 'pu2_dst'
*
* @par Description:
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[in] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] nt
*  integer Transform Block size
*
* @param[in] mode
*  integer intraprediction mode
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_intra_pred_chroma_mode_27_to_33_sse42(UWORD16 *pu2_ref,
                                                     WORD32 src_strd,
                                                     UWORD16 *pu2_dst,
                                                     WORD32 dst_strd,
                                                     WORD32 nt,
                                                     WORD32 mode)
{
    WORD32 row ;
    WORD32 pos, fract;
    WORD32 intra_pred_ang;
    WORD32 idx, ref_main_idx;

    __m128i zero_8x16b, fract_8x16b, const_temp_8x16b;

    if(nt > 16)
    {
        ihevc_hbd_intra_pred_chroma_mode_27_to_33(pu2_ref, src_strd, pu2_dst, dst_strd, nt, mode);
        return;
    }

    intra_pred_ang = gai4_ihevc_ang_table_chroma[mode];
    const_temp_8x16b = _mm_set1_epi32(16);

    if(nt==4) /* if nt =4 */
    {
        __m128i const_temp2_4x32b, const_temp3_4x32b, mask_4x32b;
        __m128i src_values10, src_values11, zero_8x16b, intra_pred_ang_4x32b;
        __m128i row_4x32b, two_nt_4x32b, src_values12;
        __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

        const_temp2_4x32b = _mm_set1_epi32(31);
        const_temp3_4x32b = _mm_set1_epi32(32);

        two_nt_4x32b = _mm_set1_epi32((4*nt)+2);

        zero_8x16b = _mm_set1_epi16(0);
        mask_4x32b = _mm_set_epi32(0, 0, 0, 0x80808080);  /* Mask register */

        /* intra_pred_ang = gai4_ihevc_ang_table[mode]; */
        intra_pred_ang_4x32b = _mm_set1_epi32(intra_pred_ang);

        row_4x32b = _mm_set_epi32(4, 3, 2, 1);
        {
            WORD32 ref_main_idx1, ref_main_idx2, ref_main_idx3, ref_main_idx4;
            WORD8  ai1_src_temp0_val[16], ai1_src_temp1_val[16];

            __m128i fract1_8x16b, fract2_8x16b, fract3_8x16b, fract4_8x16b, res_temp5_4x32b;
            __m128i src_values0, src_values1, src_values2, src_values3, src_values13;
            __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
            __m128i ref_main_temp0, ref_main_temp1, ref_main_temp2;

            /* pos = ((row + 1) * intra_pred_ang); */
            res_temp5_4x32b  = _mm_mullo_epi32(row_4x32b, intra_pred_ang_4x32b);

            src_values12 = _mm_add_epi32(two_nt_4x32b , _mm_srai_epi32(res_temp5_4x32b,  5));
            src_values12 = _mm_add_epi32(src_values12 , _mm_srai_epi32(res_temp5_4x32b,  5));

            ref_main_temp0 = _mm_srli_si128(src_values12 , 4);  /* next 32 bit values */
            ref_main_temp1 = _mm_srli_si128(src_values12 , 8);  /* next 32 bit values */
            ref_main_temp2 = _mm_srli_si128(src_values12 , 12); /* next 32 bit values */
            ref_main_idx1  = _mm_cvtsi128_si32(src_values12);    /* row=0 */
            ref_main_idx2  = _mm_cvtsi128_si32(ref_main_temp0);  /* row=1 */
            ref_main_idx3  = _mm_cvtsi128_si32(ref_main_temp1);  /* row=2 */
            ref_main_idx4  = _mm_cvtsi128_si32(ref_main_temp2);  /* row=3 */

            /* fract = pos & (31); */
            src_values11= _mm_and_si128(res_temp5_4x32b, const_temp2_4x32b);

            /* (32 - fract) */
            src_values10 = _mm_sub_epi32(const_temp3_4x32b , src_values11);

            _mm_storeu_si128((__m128i *)(ai1_src_temp1_val), src_values11);
            _mm_storeu_si128((__m128i *)(ai1_src_temp0_val), src_values10);

            fract1_8x16b = _mm_set1_epi16(ai1_src_temp1_val[0]);  /* row=0 */
            fract2_8x16b = _mm_set1_epi16(ai1_src_temp1_val[4]);  /* row=1 */
            fract3_8x16b = _mm_set1_epi16(ai1_src_temp1_val[8]);  /* row=2 */
            fract4_8x16b = _mm_set1_epi16(ai1_src_temp1_val[12]);  /* row=3 */

            temp1_8x16b = _mm_set1_epi16(ai1_src_temp0_val[0]);  /* row=0 */
            temp2_8x16b = _mm_set1_epi16(ai1_src_temp0_val[4]);  /* row=1 */
            temp3_8x16b = _mm_set1_epi16(ai1_src_temp0_val[8]);  /* row=2 */
            temp4_8x16b = _mm_set1_epi16(ai1_src_temp0_val[12]);  /* row=3 */

            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);
            temp2_8x16b = _mm_unpacklo_epi16(temp2_8x16b, fract2_8x16b);
            temp3_8x16b = _mm_unpacklo_epi16(temp3_8x16b, fract3_8x16b);
            temp4_8x16b = _mm_unpacklo_epi16(temp4_8x16b, fract4_8x16b);

            // inner loop starts from here
            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1));  /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx2));  /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx3));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx4));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+2));
            src_values11 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx2+2));
            src_values12 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx3+2));
            src_values13 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx4+2));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp1_8x16b);
            src_values1 = _mm_madd_epi16(src_values1, temp2_8x16b);
            src_values2 = _mm_madd_epi16(src_values2, temp3_8x16b);
            src_values3 = _mm_madd_epi16(src_values3, temp4_8x16b);

            src_values10 = _mm_madd_epi16(src_values10, temp1_8x16b);
            src_values11 = _mm_madd_epi16(src_values11, temp2_8x16b);
            src_values12 = _mm_madd_epi16(src_values12, temp3_8x16b);
            src_values13 = _mm_madd_epi16(src_values13, temp4_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 16 bit to 8 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            _mm_storeu_si128((__m128i *)(pu2_dst+(0 * dst_strd)), src_values0);       /* row=0 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((1) * dst_strd)), src_values1);   /* row=1 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((2) * dst_strd)), src_values2);   /* row=2 */
            _mm_storeu_si128((__m128i *)(pu2_dst+((3) * dst_strd)), src_values3);   /* row=3 */
        }
    }
    else if(nt == 8) /* for nt = 16 case */
    {
        WORD32 ref_main_idx1 , fract1, temp, temp1;
        __m128i fract1_8x16b, temp_8x16b, temp1_8x16b;
        __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 2)
        {
            __m128i src_values0, src_values1, src_values2, src_values3;
            __m128i  src_values10, src_values11, src_values12, src_values13;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp= 32-fract;
            ref_main_idx = (4 * nt) + 2 * idx + 2; /* col from 0-15 */

            pos = ((row + 2) * intra_pred_ang);
            idx = pos >> 5;
            fract1 = pos & (31);
            temp1= 32-fract1;
            ref_main_idx1 = (4 * nt) + 2 * idx + 2; /* col from 0-15 */

            fract_8x16b  = _mm_set1_epi16(fract);
            fract1_8x16b = _mm_set1_epi16(fract1);
            temp_8x16b   = _mm_set1_epi16(temp);
            temp1_8x16b  = _mm_set1_epi16(temp1);

            temp_8x16b = _mm_unpacklo_epi16(temp_8x16b, fract_8x16b);
            temp1_8x16b = _mm_unpacklo_epi16(temp1_8x16b, fract1_8x16b);

            /* row=0 */
            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+8));   /* col = 8-15 */

            /* row=1 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1));    /* col = 0-7 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+8));  /* col = 8-15 */

            src_values10 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+2));
            src_values11 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+8+2));
            src_values12 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+2));
            src_values13 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx1+8+2));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            src_values0 = _mm_madd_epi16(src_values0, temp_8x16b);
            src_values1 = _mm_madd_epi16(src_values1, temp_8x16b);

            src_values2 = _mm_madd_epi16(src_values2, temp1_8x16b);
            src_values3 = _mm_madd_epi16(src_values3, temp1_8x16b);

            src_values10 = _mm_madd_epi16(src_values10, temp_8x16b);
            src_values11 = _mm_madd_epi16(src_values11, temp_8x16b);
            src_values12 = _mm_madd_epi16(src_values12, temp1_8x16b);
            src_values13 = _mm_madd_epi16(src_values13, temp1_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);

            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);

            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);

            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);

            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);

            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 8-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_values0);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_values1);

            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd), src_values2);
            _mm_storeu_si128((__m128i *)(pu2_dst+dst_strd+8), src_values3);

            pu2_dst += 2*dst_strd;
        }
    }
    else if(nt==16)
    {
        WORD32 temp;
        /* unroll the col loop (inner) */
        zero_8x16b = _mm_set1_epi16(0);

        for(row = 0; row < nt; row += 1)
        {
            __m128i  src_values0, src_values1, src_values2, src_values3, temp_8x16b;
            __m128i  src_values10, src_values11, src_values12, src_values13;
            __m128i  src_temp0_8x16b, src_temp1_8x16b, src_temp2_8x16b, src_temp3_8x16b;

            pos = ((row + 1) * intra_pred_ang);
            idx = pos >> 5;
            fract = pos & (31);
            temp = 32-fract;
            ref_main_idx =(4 * nt) + 2 * idx + 2; /* col from 0-31 */

            fract_8x16b = _mm_set1_epi16(fract);
            temp_8x16b  = _mm_set1_epi16(temp);

            temp_8x16b = _mm_unpacklo_epi16(temp_8x16b, fract_8x16b);

            src_temp0_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx));     /* col = 0-7 */
            src_temp1_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+8));   /* col = 8-15 */
            src_temp2_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+16));  /* col = 16-23 */
            src_temp3_8x16b = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+24));  /* col = 24-31 */

            src_values10 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+2));
            src_values11 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+8+2));
            src_values12 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+16+2));
            src_values13 = _mm_loadu_si128((__m128i *)(pu2_ref+ref_main_idx+24+2));

            src_values0 = _mm_unpacklo_epi16(src_temp0_8x16b, src_values10);
            src_values1 = _mm_unpacklo_epi16(src_temp1_8x16b, src_values11);
            src_values2 = _mm_unpacklo_epi16(src_temp2_8x16b, src_values12);
            src_values3 = _mm_unpacklo_epi16(src_temp3_8x16b, src_values13);

            src_values10 = _mm_unpackhi_epi16(src_temp0_8x16b, src_values10);
            src_values11 = _mm_unpackhi_epi16(src_temp1_8x16b, src_values11);
            src_values12 = _mm_unpackhi_epi16(src_temp2_8x16b, src_values12);
            src_values13 = _mm_unpackhi_epi16(src_temp3_8x16b, src_values13);

            /* fract*(pu1_ref[ref_main_idx + 1]- pu1_ref[ref_main_idx]) */
            src_values0 = _mm_madd_epi16(src_values0, temp_8x16b);
            src_values1 = _mm_madd_epi16(src_values1, temp_8x16b);
            src_values2 = _mm_madd_epi16(src_values2, temp_8x16b);
            src_values3 = _mm_madd_epi16(src_values3, temp_8x16b);

            src_values10 = _mm_madd_epi16(src_values10, temp_8x16b);
            src_values11 = _mm_madd_epi16(src_values11, temp_8x16b);
            src_values12 = _mm_madd_epi16(src_values12, temp_8x16b);
            src_values13 = _mm_madd_epi16(src_values13, temp_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) */
            src_values0 = _mm_add_epi32(src_values0, const_temp_8x16b);
            src_values1 = _mm_add_epi32(src_values1, const_temp_8x16b);
            src_values2 = _mm_add_epi32(src_values2, const_temp_8x16b);
            src_values3 = _mm_add_epi32(src_values3, const_temp_8x16b);

            src_values10 = _mm_add_epi32(src_values10, const_temp_8x16b);
            src_values11 = _mm_add_epi32(src_values11, const_temp_8x16b);
            src_values12 = _mm_add_epi32(src_values12, const_temp_8x16b);
            src_values13 = _mm_add_epi32(src_values13, const_temp_8x16b);

            /* ((32 - fract)* pu1_ref[ref_main_idx]+ fract * pu1_ref[ref_main_idx + 1] + 16) >>5 */
            src_values0 = _mm_srai_epi32(src_values0,  5);
            src_values1 = _mm_srai_epi32(src_values1,  5);
            src_values2 = _mm_srai_epi32(src_values2,  5);
            src_values3 = _mm_srai_epi32(src_values3,  5);

            src_values10 = _mm_srai_epi32(src_values10,  5);
            src_values11 = _mm_srai_epi32(src_values11,  5);
            src_values12 = _mm_srai_epi32(src_values12,  5);
            src_values13 = _mm_srai_epi32(src_values13,  5);

            /* converting 32 bit to 16 bit */
            src_values0 = _mm_packus_epi32(src_values0, src_values10);
            src_values1 = _mm_packus_epi32(src_values1, src_values11);
            src_values2 = _mm_packus_epi32(src_values2, src_values12);
            src_values3 = _mm_packus_epi32(src_values3, src_values13);

            /* loding 8-bit 8 pixels values */
            _mm_storeu_si128((__m128i *)(pu2_dst), src_values0);
            _mm_storeu_si128((__m128i *)(pu2_dst+8), src_values1);
            _mm_storeu_si128((__m128i *)(pu2_dst+16), src_values2);
            _mm_storeu_si128((__m128i *)(pu2_dst+24), src_values3);

            pu2_dst += dst_strd;
        }
    }
}
