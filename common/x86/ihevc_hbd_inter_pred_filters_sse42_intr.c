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
*  ihevc_hbd_inter_pred_filters_sse42_intr.c
*
* @brief
*  Contains function definitions for high bit depth inter prediction
*  interpolation filters (SSE4.2).
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_inter_pred_luma_copy_sse42()
*  - ihevc_hbd_inter_pred_luma_horz_sse42()
*  - ihevc_hbd_inter_pred_luma_vert_sse42()
*  - ihevc_hbd_inter_pred_luma_horz_w16out_sse42()
*  - ihevc_hbd_inter_pred_luma_vert_w16out_sse42()
*  - ihevc_hbd_inter_pred_luma_vert_w16inp_sse42()
*  - ihevc_hbd_inter_pred_luma_vert_w16inp_w16out_sse42()
*  - ihevc_hbd_inter_pred_luma_copy_w16out_sse42()
*  - ihevc_hbd_inter_pred_chroma_horz_sse42()
*  - ihevc_hbd_inter_pred_chroma_vert_sse42()
*  - ihevc_hbd_inter_pred_chroma_horz_w16out_sse42()
*  - ihevc_hbd_inter_pred_chroma_vert_w16out_sse42()
*  - ihevc_hbd_inter_pred_chroma_vert_w16inp_sse42()
*  - ihevc_hbd_inter_pred_chroma_vert_w16inp_w16out_sse42()
*  - ihevc_hbd_inter_pred_chroma_copy_sse42()
*  - ihevc_hbd_inter_pred_chroma_copy_w16out_sse42()
*
* @remarks
*  None
*
*******************************************************************************
*/

/**
*******************************************************************************
* @file
*  ihevc_hbd_inter_pred_filters_x86_intr.c
*
* @brief
*  Contains function definitions for inter prediction  interpolation filters
*  coded in x86 intrinsics
*
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_inter_pred_luma_copy()
*  - ihevc_hbd_inter_pred_luma_horz()
*  - ihevc_hbd_inter_pred_luma_vert()
*  - ihevc_hbd_inter_pred_luma_copy_w16out()
*  - ihevc_hbd_inter_pred_luma_horz_w16out()
*  - ihevc_hbd_inter_pred_luma_vert_w16out()
*  - ihevc_hbd_inter_pred_luma_vert_w16inp()
*  - ihevc_hbd_inter_pred_luma_vert_w16inp_w16out()
*  - ihevc_hbd_inter_pred_chroma_copy()
*  - ihevc_hbd_inter_pred_chroma_horz()
*  - ihevc_hbd_inter_pred_chroma_vert()
*  - ihevc_hbd_inter_pred_chroma_copy_w16out()
*  - ihevc_hbd_inter_pred_chroma_horz_w16out()
*  - ihevc_hbd_inter_pred_chroma_vert_w16out()
*  - ihevc_hbd_inter_pred_chroma_vert_w16inp()
*  - ihevc_hbd_inter_pred_chroma_vert_w16inp_w16out()
*
* @remarks
*  None
*
*******************************************************************************
*/


/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <assert.h>

#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
#include "ihevc_inter_pred.h"
#include "ihevc_platform_macros.h"
#include "ihevc_macros.h"
#include "ihevc_debug.h"
#include <immintrin.h>


/*****************************************************************************/
/* Function Definitions                                                      */
/*****************************************************************************/

/**
*******************************************************************************
*
* @brief
*       Interprediction luma function for copy
*
* @par Description:
*    Copies the array of width 'wd' and height 'ht' from the  location pointed
*    by 'src' to the location pointed by 'dst'
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
* Assumption : ht%4 == 0, wd%4 == 0
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_copy_sse42(UWORD16 *pu2_src,
                                          UWORD16 *pu2_dst,
                                          WORD32 src_strd,
                                          WORD32 dst_strd,
                                          WORD8 *pi1_coeff,
                                          WORD32 ht,
                                          WORD32 wd,
                                          UWORD8 bit_depth)
{
    WORD32 row, col;
    __m128i  src0_16x8b, src1_16x8b, src2_16x8b, src3_16x8b;

    ASSERT(wd%4 == 0); /* checking assumption */
    ASSERT(ht%4 == 0); /* checking assumption */

    /* outer for loop starts from here */
    if(0 == (wd & 7)) /* multiple of 8 case */
    {
        for(row = 0; row < ht; row += 4)
        {
            for(col = 0; col < wd; col += 8)
            {
                /* load 8 pixel values from 8:0 pos. relative to cur. pos. */
                src0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));            /* row =0 */
                src1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                src2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                src3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                /* storing 8 16-bit output values */
                _mm_storeu_si128((__m128i *)(pu2_dst), src0_16x8b);               /* row =0 */
                _mm_storeu_si128((__m128i *)(pu2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                _mm_storeu_si128((__m128i *)(pu2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                _mm_storeu_si128((__m128i *)(pu2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                pu2_src += 8; /* pointer update */
                pu2_dst += 8; /* pointer update */
            } /* inner for loop ends here(8-output values in single iteration) */

            pu2_src += 4*src_strd - wd; /* pointer update */
            pu2_dst += 4*dst_strd - wd; /* pointer update */
        }
    }
    else /* wd = multiple of 4 case */
    {
        for(row = 0; row < ht; row += 4)
        {
            for(col = 0; col < wd; col += 4)
            {
                /* load 4 pixel values from 4:0 pos. relative to cur. pos. */
                src0_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src));            /* row =0 */
                src1_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                src2_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                src3_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                _mm_storel_epi64((__m128i *)(pu2_dst), src0_16x8b);               /* row =0 */
                _mm_storel_epi64((__m128i *)(pu2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                _mm_storel_epi64((__m128i *)(pu2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                _mm_storel_epi64((__m128i *)(pu2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                pu2_src += 4; /* pointer update */
                pu2_dst += 4; /* pointer update */
            } /* inner for loop ends here(4- output values in single iteration) */

            pu2_src += 4*src_strd - wd; /* pointer update */
            pu2_dst += 4*dst_strd - wd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     Interprediction luma filter for horizontal input
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to  by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and  writes to the location pointed
*    by 'pu2_dst'  The output is downshifted by 6 and clipped to 8 bits
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_luma_horz_sse42(UWORD16 *pu2_src,
                                          UWORD16 *pu2_dst,
                                          WORD32 src_strd,
                                          WORD32 dst_strd,
                                          WORD8 *pi1_coeff,
                                          WORD32 ht,
                                          WORD32 wd,
                                          UWORD8 bit_depth)
{
    WORD32 row, col;

    /* all 128 bit registers are named with a suffix mxnb, where m is the */
    /* number of n bits packed in the register */
    __m128i zero_8x16b, offset_8x16b;
    __m128i src_temp1_16x8b, src_temp2_16x8b, src_temp3_16x8b, src_temp4_16x8b, src_temp5_16x8b, src_temp6_16x8b;
    __m128i src_temp7_16x8b, src_temp8_16x8b, src_temp9_16x8b, src_temp0_16x8b;
    __m128i src_temp11_16x8b, src_temp12_16x8b, src_temp13_16x8b, src_temp14_16x8b, src_temp15_16x8b, src_temp16_16x8b;
    __m128i res_temp1_8x16b, res_temp2_8x16b, res_temp3_8x16b, res_temp4_8x16b, res_temp5_8x16b, res_temp6_8x16b, res_temp7_8x16b, res_temp8_8x16b;
    __m128i res_temp9_8x16b, res_temp0_8x16b;
    __m128i res_temp11_8x16b, res_temp12_8x16b, res_temp13_8x16b, res_temp14_8x16b, res_temp15_8x16b, res_temp16_8x16b;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b, coeff4_5_8x16b, coeff6_7_8x16b;
    __m128i clip_tmp_8x16b, sign_reg;
    ASSERT(wd%4 == 0); /* checking assumption */
    PREFETCH((char const*)(pu2_src+(0 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(1 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(2 * src_strd)), _MM_HINT_T0);
    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    src_temp1_16x8b = _mm_loadl_epi64((__m128i *)pi1_coeff);
    zero_8x16b = _mm_set1_epi32(0);
    offset_8x16b = _mm_set1_epi32((1<<(FILTER_PREC-1))); /* for offset addition */

    clip_tmp_8x16b = _mm_cmpeq_epi16(zero_8x16b, zero_8x16b);
    clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);

    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, src_temp1_16x8b);
    src_temp2_16x8b  = _mm_unpacklo_epi8(src_temp1_16x8b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    coeff4_5_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(2, 2, 2, 2));  /* pi1_coeff[4] */
    coeff6_7_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(3, 3, 3, 3));  /* pi1_coeff[4] */

    if(0 == (ht & 1)) /* ht multiple of 2 case */
    {
        if(0 == (wd & 7)) /* wd = multiple of 8 case */
        {
            for(row = 0; row < ht; row += 1 /* 2 */)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(3 * src_strd)), _MM_HINT_T0);

                for(col = 0; col < wd; col += 8)
                {
                    /* load 8 pixel values from row 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));   /* row = 0 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp7_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */
                    res_temp7_8x16b = _mm_madd_epi16(src_temp7_16x8b, coeff0_1_8x16b);
                    /* row = 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp8_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);
                    res_temp8_8x16b = _mm_madd_epi16(src_temp8_16x8b, coeff2_3_8x16b); /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp9_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */
                    res_temp9_8x16b = _mm_madd_epi16(src_temp9_16x8b, coeff4_5_8x16b);

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b = _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp0_16x8b = _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */
                    res_temp0_8x16b = _mm_madd_epi16(src_temp0_16x8b, coeff6_7_8x16b);

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp7_8x16b, res_temp8_8x16b);
                    res_temp7_8x16b = _mm_add_epi32(res_temp9_8x16b, res_temp0_8x16b);
                    res_temp8_8x16b = _mm_add_epi32(res_temp6_8x16b, res_temp7_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp5_8x16b, offset_8x16b);            /* row = 0 */
                    res_temp7_8x16b = _mm_add_epi32(res_temp8_8x16b, offset_8x16b);
                    res_temp6_8x16b = _mm_srai_epi32(res_temp6_8x16b, FILTER_PREC); /* row = 0 */
                    res_temp7_8x16b = _mm_srai_epi32(res_temp7_8x16b, FILTER_PREC);
                    res_temp5_8x16b = _mm_packus_epi32(res_temp6_8x16b, res_temp7_8x16b);       /* row = 0 */

                    // Clipping
                    res_temp5_8x16b = _mm_min_epi16(res_temp5_8x16b, clip_tmp_8x16b);

                    res_temp5_8x16b = _mm_max_epi16(res_temp5_8x16b, zero_8x16b);

                    /* to store the 8 pixels res. */
                    _mm_storeu_si128((__m128i *)(pu2_dst + offset), res_temp5_8x16b);

                    offset += 8; /* To pointer updates */
                }

                pu2_src += 1*src_strd; /* pointer updates */
                pu2_dst += 1*dst_strd; /* pointer updates */
            }
        }
        else /* wd = multiple of 4 case */
        {
            for(row = 0; row < ht; row += 2)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                for(col = 0; col < wd; col += 4)
                {
                    /* load 16 pixel values from 12:-3 pos. relative to cur. pos. */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));         /* row = 0 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset)); /* row = 1 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));                 /* row = 0 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp5_8x16b, offset_8x16b);            /* row = 0 */
                    res_temp6_8x16b = _mm_srai_epi32(res_temp6_8x16b, FILTER_PREC);          /* row = 0 */
                    res_temp5_8x16b = _mm_packus_epi32(res_temp6_8x16b, res_temp6_8x16b);       /* row = 0 */

                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+1));                 /* row =1 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row =1 */
                    src_temp13_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp11_8x16b = _mm_madd_epi16(src_temp13_16x8b, coeff0_1_8x16b);    /* row = 1 */
                    /* row =1 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+2));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+3));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp14_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp12_8x16b = _mm_madd_epi16(src_temp14_16x8b, coeff2_3_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+4));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+5));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp13_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff4_5_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+6));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+7));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp16_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp14_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff6_7_8x16b);    /* row = 1 */

                    res_temp15_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);
                    res_temp16_8x16b = _mm_add_epi32(res_temp13_8x16b, res_temp14_8x16b);
                    res_temp15_8x16b = _mm_add_epi32(res_temp15_8x16b, res_temp16_8x16b);

                    res_temp16_8x16b = _mm_add_epi32(res_temp15_8x16b, offset_8x16b);            /* row = 1 */
                    res_temp16_8x16b = _mm_srai_epi32(res_temp16_8x16b, FILTER_PREC);           /* row = 1 */
                    res_temp15_8x16b = _mm_packus_epi32(res_temp16_8x16b, res_temp16_8x16b);       /* row = 1 */

                    // Clipping
                    res_temp5_8x16b = _mm_min_epi16(res_temp5_8x16b, clip_tmp_8x16b);
                    res_temp15_8x16b = _mm_min_epi16(res_temp15_8x16b, clip_tmp_8x16b);

                    res_temp5_8x16b = _mm_max_epi16(res_temp5_8x16b, zero_8x16b);
                    res_temp15_8x16b = _mm_max_epi16(res_temp15_8x16b, zero_8x16b);

                    /* to store the 1st 4 pixels res. */
                    _mm_storel_epi64((__m128i *)(pu2_dst + offset), res_temp5_8x16b);
                    _mm_storel_epi64((__m128i *)(pu2_dst + dst_strd + offset), res_temp15_8x16b);

                    offset += 4; /* To pointer updates */
                }
                pu2_src += 2*src_strd;  /* Pointer update */
                pu2_dst += 2*dst_strd;  /* Pointer update */
            }
        }
    }
    else /* odd ht */
    {
        if(0 == (wd & 7)) /* multiple of 8 case */
        {
            for(row = 0; row < ht; row++)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                for(col = 0; col < wd; col += 8)
                {
                    /* load 8 pixel values from row 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));   /* row = 0 */

                    /* load 8 pixel values from row 1 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp7_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */
                    res_temp7_8x16b = _mm_madd_epi16(src_temp7_16x8b, coeff0_1_8x16b);
                    /* row = 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp8_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);
                    res_temp8_8x16b = _mm_madd_epi16(src_temp8_16x8b, coeff2_3_8x16b); /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp9_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */
                    res_temp9_8x16b = _mm_madd_epi16(src_temp9_16x8b, coeff4_5_8x16b);

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b = _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp0_16x8b = _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */
                    res_temp0_8x16b = _mm_madd_epi16(src_temp0_16x8b, coeff6_7_8x16b);

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp7_8x16b, res_temp8_8x16b);
                    res_temp7_8x16b = _mm_add_epi32(res_temp9_8x16b, res_temp0_8x16b);
                    res_temp8_8x16b = _mm_add_epi32(res_temp6_8x16b, res_temp7_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp5_8x16b, offset_8x16b);            /* row = 0 */
                    res_temp7_8x16b = _mm_add_epi32(res_temp8_8x16b, offset_8x16b);
                    res_temp6_8x16b = _mm_srai_epi32(res_temp6_8x16b, FILTER_PREC); /* row = 0 */
                    res_temp7_8x16b = _mm_srai_epi32(res_temp7_8x16b, FILTER_PREC);
                    res_temp5_8x16b = _mm_packus_epi32(res_temp6_8x16b, res_temp7_8x16b);       /* row = 0 */

                    // Clipping
                    res_temp5_8x16b = _mm_min_epi16(res_temp5_8x16b, clip_tmp_8x16b);

                    res_temp5_8x16b = _mm_max_epi16(res_temp5_8x16b, zero_8x16b);

                    _mm_storeu_si128((__m128i *)(pu2_dst + offset), res_temp5_8x16b);

                    offset += 8; /* To pointer updates */
                }
                pu2_src += src_strd;    /* pointer updates */
                pu2_dst += dst_strd;    /* pointer updates */
            }
        }
        else  /* wd = multiple of 4 case */
        {
            for(row = 0; row < (ht-1); row += 2)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);


                for(col = 0; col < wd; col += 4)
                {
                    /* load 16 pixel values from 12:-3 pos. relative to cur. pos. */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));         /* row = 0 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset)); /* row = 1 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));                 /* row = 0 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp5_8x16b, offset_8x16b);            /* row = 0 */
                    res_temp6_8x16b = _mm_srai_epi32(res_temp6_8x16b, FILTER_PREC);          /* row = 0 */
                    res_temp5_8x16b = _mm_packus_epi32(res_temp6_8x16b, res_temp6_8x16b);       /* row = 0 */

                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+1));                 /* row =1 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row =1 */
                    src_temp13_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp11_8x16b = _mm_madd_epi16(src_temp13_16x8b, coeff0_1_8x16b);    /* row = 1 */
                    /* row =1 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+2));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+3));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp14_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp12_8x16b = _mm_madd_epi16(src_temp14_16x8b, coeff2_3_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+4));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+5));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp13_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff4_5_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+6));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+7));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp16_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp14_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff6_7_8x16b);    /* row = 1 */

                    res_temp15_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);
                    res_temp16_8x16b = _mm_add_epi32(res_temp13_8x16b, res_temp14_8x16b);
                    res_temp15_8x16b = _mm_add_epi32(res_temp15_8x16b, res_temp16_8x16b);

                    res_temp16_8x16b = _mm_add_epi32(res_temp15_8x16b, offset_8x16b);            /* row = 1 */
                    res_temp16_8x16b = _mm_srai_epi32(res_temp16_8x16b, FILTER_PREC);           /* row = 1 */
                    res_temp15_8x16b = _mm_packus_epi32(res_temp16_8x16b, res_temp16_8x16b);       /* row = 1 */

                    // Clipping
                    res_temp5_8x16b = _mm_min_epi16(res_temp5_8x16b, clip_tmp_8x16b);
                    res_temp15_8x16b = _mm_min_epi16(res_temp15_8x16b, clip_tmp_8x16b);

                    res_temp5_8x16b = _mm_max_epi16(res_temp5_8x16b, zero_8x16b);
                    res_temp15_8x16b = _mm_max_epi16(res_temp15_8x16b, zero_8x16b);

                    /* to store the 1st 4 pixels res. */
                    _mm_storel_epi64((__m128i *)(pu2_dst + offset), res_temp5_8x16b);
                    _mm_storel_epi64((__m128i *)(pu2_dst + dst_strd + offset), res_temp15_8x16b);

                    offset += 4; /* To pointer updates */
                }
                pu2_src += 2*src_strd;  /* Pointer update */
                pu2_dst += 2*dst_strd;  /* Pointer update */
            }
            {
                /* last repeat at outside the loop */
                int offset =0;
                for(col = 0; col < wd; col += 4)
                {
                    /* load 16 pixel values from 12:-3 pos. relative to cur. pos. */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));         /* row = 0 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));                 /* row = 0 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp5_8x16b, offset_8x16b);            /* row = 0 */
                    res_temp6_8x16b = _mm_srai_epi32(res_temp6_8x16b, FILTER_PREC);          /* row = 0 */
                    res_temp5_8x16b = _mm_packus_epi32(res_temp6_8x16b, res_temp6_8x16b);       /* row = 0 */

                    // Clipping
                    res_temp5_8x16b = _mm_min_epi16(res_temp5_8x16b, clip_tmp_8x16b);

                    res_temp5_8x16b = _mm_max_epi16(res_temp5_8x16b, zero_8x16b);

                    _mm_storel_epi64((__m128i *)(pu2_dst + offset), res_temp5_8x16b);

                    offset += 4; /* To pointer updates */
                }
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*    Interprediction luma filter for vertical input
*
* @par Description:
*   Applies a vertcal filter with coefficients pointed to  by 'pi1_coeff' to
*   the elements pointed by 'pu1_src' and  writes to the location pointed by
*   'pu1_dst'  The output is downshifted by 6 and clipped to 8 bits
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_luma_vert_sse42(UWORD16 *pu2_src,
                                          UWORD16 *pu2_dst,
                                          WORD32 src_strd,
                                          WORD32 dst_strd,
                                          WORD8 *pi1_coeff,
                                          WORD32 ht,
                                          WORD32 wd,
                                          UWORD8 bit_depth)
{
    WORD32 row, col;
    UWORD16 *pu2_src_copy;
    UWORD16 *pu2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b, coeff4_5_8x16b, coeff6_7_8x16b;
    __m128i s0_8x16b, s1_8x16b, s2_8x16b, s3_8x16b, s4_8x16b, s5_8x16b, s6_8x16b, s7_8x16b, s8_8x16b, s9_8x16b;
    __m128i s2_0_16x8b, s2_1_16x8b, s2_2_16x8b, s2_3_16x8b, s2_4_16x8b, s2_5_16x8b, s2_6_16x8b, s2_7_16x8b, s2_8_16x8b, s2_9_16x8b, s2_10_16x8b;
    __m128i s3_0_16x8b, s3_1_16x8b, s3_2_16x8b, s3_3_16x8b, s3_4_16x8b, s3_5_16x8b, s3_6_16x8b, s3_7_16x8b, s3_8_16x8b;
    __m128i s4_0_16x8b, s4_1_16x8b, s4_2_16x8b, s4_3_16x8b, s4_4_16x8b, s4_5_16x8b, s4_6_16x8b, s4_7_16x8b, s4_8_16x8b;
    __m128i s10_8x16b, s11_8x16b, s12_8x16b, s13_8x16b, s14_8x16b, s15_8x16b, s16_8x16b, s17_8x16b, s18_8x16b, s19_8x16b;
    __m128i s20_8x16b, s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b, s26_8x16b, s27_8x16b, s28_8x16b, s29_8x16b;
    __m128i s30_8x16b, s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b, s36_8x16b, s37_8x16b, s38_8x16b, s39_8x16b;

    __m128i zero_8x16b, offset_8x16b;
    __m128i clip_tmp_8x16b, sign_reg;

    PREFETCH((char const*)(pu2_src+(0 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(1 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(2 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(3 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(4 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(5 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128(); /* for saturated clipping */
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    clip_tmp_8x16b = _mm_cmpeq_epi16(zero_8x16b, zero_8x16b);
    clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    coeff4_5_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(2, 2, 2, 2));  /* pi1_coeff[4] */
    coeff6_7_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(3, 3, 3, 3));  /* pi1_coeff[4] */
    /* seting  values in register */
    offset_8x16b = _mm_set1_epi32((1<<(FILTER_PREC-1))); /* for offset addition */

    /* outer for loop starts from here */
    if(wd %8 == 0)
    { /* wd = multiple of 8 case */

        pu2_src_copy = pu2_src;
        pu2_dst_copy = pu2_dst;

        for(col = 0; col < wd; col += 8)
        {

            pu2_src = pu2_src_copy + col;
            pu2_dst = pu2_dst_copy + col;

            PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);

            /* load 8 pixel values. */
            s2_0_16x8b  = _mm_loadu_si128((__m128i *)(pu2_src+(-3 * src_strd)));

            /* load 8 pixel values */
            s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-2 * src_strd)));

            s3_0_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
            s3_4_16x8b = _mm_unpackhi_epi16(s2_0_16x8b, s2_1_16x8b);

            s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
            s4_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff0_1_8x16b );
            /* load 8 pixel values */
            s2_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

            /* load 8 pixel values */
            s2_3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

            s3_1_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);
            s3_5_16x8b = _mm_unpackhi_epi16(s2_2_16x8b, s2_3_16x8b);

            s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
            s5_8x16b = _mm_madd_epi16(s3_5_16x8b, coeff2_3_8x16b );

            /* load 8 pixel values */
            s2_4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

            /* load 8 pixel values */
            s2_5_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

            s3_2_16x8b = _mm_unpacklo_epi16(s2_4_16x8b, s2_5_16x8b);
            s3_6_16x8b = _mm_unpackhi_epi16(s2_4_16x8b, s2_5_16x8b);

            s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );
            s6_8x16b = _mm_madd_epi16(s3_6_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_6_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

            /* load 8 pixel values */
            s2_7_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(4 * src_strd)));

            s3_3_16x8b = _mm_unpacklo_epi16(s2_6_16x8b, s2_7_16x8b);
            s3_7_16x8b = _mm_unpackhi_epi16(s2_6_16x8b, s2_7_16x8b);

            s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );
            s7_8x16b = _mm_madd_epi16(s3_7_16x8b, coeff6_7_8x16b );

            s0_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
            s2_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
            s4_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);
            s6_8x16b = _mm_add_epi32(s6_8x16b, s7_8x16b);
            s0_8x16b = _mm_add_epi32(s0_8x16b, s2_8x16b);
            s4_8x16b = _mm_add_epi32(s4_8x16b, s6_8x16b);

            s0_8x16b = _mm_add_epi32(s0_8x16b, offset_8x16b);
            s4_8x16b = _mm_add_epi32(s4_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s7_8x16b = _mm_srai_epi32(s0_8x16b,  FILTER_PREC);
            s8_8x16b = _mm_srai_epi32(s4_8x16b,  FILTER_PREC);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s9_8x16b = _mm_packus_epi32(s7_8x16b, s8_8x16b);

            s9_8x16b = _mm_min_epi16(s9_8x16b, clip_tmp_8x16b);
            s9_8x16b = _mm_max_epi16(s9_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 0 */
            _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), s9_8x16b);

            /* ROW 2 */
            s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
            s24_8x16b = _mm_madd_epi16(s3_5_16x8b, coeff0_1_8x16b );
            s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
            s25_8x16b = _mm_madd_epi16(s3_6_16x8b, coeff2_3_8x16b );
            s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );
            s26_8x16b = _mm_madd_epi16(s3_7_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_8_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(5 * src_strd)));

            /* load 8 pixel values */
            s2_9_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(6 * src_strd)));

            s3_4_16x8b = _mm_unpacklo_epi16(s2_8_16x8b, s2_9_16x8b);
            s3_8_16x8b = _mm_unpackhi_epi16(s2_8_16x8b, s2_9_16x8b);

            s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );
            s27_8x16b = _mm_madd_epi16(s3_8_16x8b, coeff6_7_8x16b );

            s20_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
            s22_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
            s24_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);
            s26_8x16b = _mm_add_epi32(s26_8x16b, s27_8x16b);
            s20_8x16b = _mm_add_epi32(s20_8x16b, s22_8x16b);
            s24_8x16b = _mm_add_epi32(s24_8x16b, s26_8x16b);

            s20_8x16b = _mm_add_epi32(s20_8x16b, offset_8x16b);
            s24_8x16b = _mm_add_epi32(s24_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s27_8x16b = _mm_srai_epi32(s20_8x16b,  FILTER_PREC);
            s28_8x16b = _mm_srai_epi32(s24_8x16b,  FILTER_PREC);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s29_8x16b = _mm_packus_epi32(s27_8x16b, s28_8x16b);

            s29_8x16b = _mm_min_epi16(s29_8x16b, clip_tmp_8x16b);

            s29_8x16b = _mm_max_epi16(s29_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 2 */
            _mm_storeu_si128((__m128i *)(pu2_dst + (2 * dst_strd)), s29_8x16b);
            /* ROW 1 */
            s4_0_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);
            s4_4_16x8b = _mm_unpackhi_epi16(s2_1_16x8b, s2_2_16x8b);

            s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
            s14_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff0_1_8x16b );

            s4_1_16x8b = _mm_unpacklo_epi16(s2_3_16x8b, s2_4_16x8b);
            s4_5_16x8b = _mm_unpackhi_epi16(s2_3_16x8b, s2_4_16x8b);

            s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
            s15_8x16b = _mm_madd_epi16(s4_5_16x8b, coeff2_3_8x16b );

            s4_2_16x8b = _mm_unpacklo_epi16(s2_5_16x8b, s2_6_16x8b);
            s4_6_16x8b = _mm_unpackhi_epi16(s2_5_16x8b, s2_6_16x8b);

            s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );
            s16_8x16b = _mm_madd_epi16(s4_6_16x8b, coeff4_5_8x16b );

            s4_3_16x8b = _mm_unpacklo_epi16(s2_7_16x8b, s2_8_16x8b);
            s4_7_16x8b = _mm_unpackhi_epi16(s2_7_16x8b, s2_8_16x8b);

            s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );
            s17_8x16b = _mm_madd_epi16(s4_7_16x8b, coeff6_7_8x16b );

            s10_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
            s12_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
            s14_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);
            s16_8x16b = _mm_add_epi32(s16_8x16b, s17_8x16b);
            s10_8x16b = _mm_add_epi32(s10_8x16b, s12_8x16b);
            s14_8x16b = _mm_add_epi32(s14_8x16b, s16_8x16b);

            s10_8x16b = _mm_add_epi32(s10_8x16b, offset_8x16b);
            s14_8x16b = _mm_add_epi32(s14_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s17_8x16b = _mm_srai_epi32(s10_8x16b,  FILTER_PREC);
            s18_8x16b = _mm_srai_epi32(s14_8x16b,  FILTER_PREC);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s19_8x16b = _mm_packus_epi32(s17_8x16b, s18_8x16b);

            s19_8x16b = _mm_min_epi16(s19_8x16b, clip_tmp_8x16b);

            s19_8x16b = _mm_max_epi16(s19_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 1 */
            _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), s19_8x16b);
            /* ROW 3 */
            s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
            s34_8x16b = _mm_madd_epi16(s4_5_16x8b, coeff0_1_8x16b );
            s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
            s35_8x16b = _mm_madd_epi16(s4_6_16x8b, coeff2_3_8x16b );
            s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );
            s36_8x16b = _mm_madd_epi16(s4_7_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_10_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(7 * src_strd)));

            s4_4_16x8b = _mm_unpacklo_epi16(s2_9_16x8b, s2_10_16x8b);
            s4_8_16x8b = _mm_unpackhi_epi16(s2_9_16x8b, s2_10_16x8b);

            s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );
            s37_8x16b = _mm_madd_epi16(s4_8_16x8b, coeff6_7_8x16b );

            s30_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
            s32_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
            s34_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);
            s36_8x16b = _mm_add_epi32(s36_8x16b, s37_8x16b);
            s30_8x16b = _mm_add_epi32(s30_8x16b, s32_8x16b);
            s34_8x16b = _mm_add_epi32(s34_8x16b, s36_8x16b);

            s30_8x16b = _mm_add_epi32(s30_8x16b, offset_8x16b);
            s34_8x16b = _mm_add_epi32(s34_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s37_8x16b = _mm_srai_epi32(s30_8x16b,  FILTER_PREC);
            s38_8x16b = _mm_srai_epi32(s34_8x16b,  FILTER_PREC);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s39_8x16b = _mm_packus_epi32(s37_8x16b, s38_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 2 */

            // Clipping
            s39_8x16b = _mm_min_epi16(s39_8x16b, clip_tmp_8x16b);

            s39_8x16b = _mm_max_epi16(s39_8x16b, zero_8x16b);

            _mm_storeu_si128((__m128i *)(pu2_dst + (3 * dst_strd)), s39_8x16b);

            pu2_src += (8 * src_strd);
            pu2_dst += (4 * dst_strd);

            s3_0_16x8b = s3_2_16x8b;
            s3_1_16x8b = s3_3_16x8b;
            s3_2_16x8b = s3_4_16x8b;
            s3_4_16x8b = s3_6_16x8b;
            s3_5_16x8b = s3_7_16x8b;
            s3_6_16x8b = s3_8_16x8b;

            s4_0_16x8b = s4_2_16x8b;
            s4_1_16x8b = s4_3_16x8b;
            s4_2_16x8b = s4_4_16x8b;
            s4_4_16x8b = s4_6_16x8b;
            s4_5_16x8b = s4_7_16x8b;
            s4_6_16x8b = s4_8_16x8b;

            for(row = 4; row < ht; row += 2 /* 4 */)
            {
                PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);

                s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
                s4_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff0_1_8x16b );
                s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
                s5_8x16b = _mm_madd_epi16(s3_5_16x8b, coeff2_3_8x16b );
                s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );
                s6_8x16b = _mm_madd_epi16(s3_6_16x8b, coeff4_5_8x16b );

                /* load 8 pixel values from (cur_row + 4)th row */
                s2_0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));

                s3_3_16x8b = _mm_unpacklo_epi16(s2_10_16x8b, s2_0_16x8b);
                s3_7_16x8b = _mm_unpackhi_epi16(s2_10_16x8b, s2_0_16x8b);
                s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );
                s7_8x16b = _mm_madd_epi16(s3_7_16x8b, coeff6_7_8x16b );

                s0_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
                s2_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
                s4_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);
                s6_8x16b = _mm_add_epi32(s6_8x16b, s7_8x16b);
                s0_8x16b = _mm_add_epi32(s0_8x16b, s2_8x16b);
                s4_8x16b = _mm_add_epi32(s4_8x16b, s6_8x16b);

                s0_8x16b = _mm_add_epi32(s0_8x16b, offset_8x16b);
                s4_8x16b = _mm_add_epi32(s4_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s7_8x16b = _mm_srai_epi32(s0_8x16b,  FILTER_PREC);
                s8_8x16b = _mm_srai_epi32(s4_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s9_8x16b = _mm_packus_epi32(s7_8x16b, s8_8x16b);

                s9_8x16b = _mm_min_epi16(s9_8x16b, clip_tmp_8x16b);
                s9_8x16b = _mm_max_epi16(s9_8x16b, zero_8x16b);
                /* store 8 8-bit output values */
                /* Store the output pixels of row 0 */
                _mm_storeu_si128((__m128i *)(pu2_dst + (0 * dst_strd)), s9_8x16b);

                /* load 8 pixel values from (cur_row + 5)th row */
                s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + src_strd));

                s3_0_16x8b = s3_1_16x8b;
                s3_1_16x8b = s3_2_16x8b;
                s3_2_16x8b = s3_3_16x8b;
                s3_4_16x8b = s3_5_16x8b;
                s3_5_16x8b = s3_6_16x8b;
                s3_6_16x8b = s3_7_16x8b;

                /* row + 1 */
                s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
                s14_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff0_1_8x16b );
                s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
                s15_8x16b = _mm_madd_epi16(s4_5_16x8b, coeff2_3_8x16b );
                s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );
                s16_8x16b = _mm_madd_epi16(s4_6_16x8b, coeff4_5_8x16b );

                /* unpacking (cur_row + 4)th row and (cur_row + 5)th row */
                s4_3_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
                s4_7_16x8b = _mm_unpackhi_epi16(s2_0_16x8b, s2_1_16x8b);
                s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );
                s17_8x16b = _mm_madd_epi16(s4_7_16x8b, coeff6_7_8x16b );

                s10_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
                s12_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
                s14_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);
                s16_8x16b = _mm_add_epi32(s16_8x16b, s17_8x16b);
                s10_8x16b = _mm_add_epi32(s10_8x16b, s12_8x16b);
                s14_8x16b = _mm_add_epi32(s14_8x16b, s16_8x16b);

                s10_8x16b = _mm_add_epi32(s10_8x16b, offset_8x16b);
                s14_8x16b = _mm_add_epi32(s14_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s17_8x16b = _mm_srai_epi32(s10_8x16b,  FILTER_PREC);
                s18_8x16b = _mm_srai_epi32(s14_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s19_8x16b = _mm_packus_epi32(s17_8x16b, s18_8x16b);

                s19_8x16b = _mm_min_epi16(s19_8x16b, clip_tmp_8x16b);

                s19_8x16b = _mm_max_epi16(s19_8x16b, zero_8x16b);

                /* store 8 8-bit output values */
                /* Store the output pixels of (cur_row + 1) */
                _mm_storeu_si128((__m128i *)(pu2_dst + (1 * dst_strd)), s19_8x16b);

                s2_10_16x8b = s2_1_16x8b;

                s4_0_16x8b = s4_1_16x8b;
                s4_1_16x8b = s4_2_16x8b;
                s4_2_16x8b = s4_3_16x8b;
                s4_4_16x8b = s4_5_16x8b;
                s4_5_16x8b = s4_6_16x8b;
                s4_6_16x8b = s4_7_16x8b;

                pu2_src += 2 * src_strd; /* pointer update */
                pu2_dst += 2 * dst_strd; /* pointer update */
            }
        }
    }
    else
    {
        /* wd = multiple of 4 case */

        pu2_src_copy = pu2_src;
        pu2_dst_copy = pu2_dst;

        for(col = 0; col < wd; col += 4)
        {
            pu2_src = pu2_src_copy + col;
            pu2_dst = pu2_dst_copy + col;

            PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);

            /* load 8 pixel values */
            s2_0_16x8b  = _mm_loadu_si128((__m128i *)(pu2_src+(-3 * src_strd)));

            /* load 8 pixel values */
            s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-2 * src_strd)));

            s3_0_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);

            s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
            /* load 8 pixel values */
            s2_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

            /* load 8 pixel values */
            s2_3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

            s3_1_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

            s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
            /* load 8 pixel values */
            s2_4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

            /* load 8 pixel values */
            s2_5_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

            s3_2_16x8b = _mm_unpacklo_epi16(s2_4_16x8b, s2_5_16x8b);

            s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );
            /* load 8 pixel values */
            s2_6_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

            /* load 8 pixel values */
            s2_7_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(4 * src_strd)));

            s3_3_16x8b = _mm_unpacklo_epi16(s2_6_16x8b, s2_7_16x8b);

            s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

            s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
            s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
            s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

            s7_8x16b = _mm_add_epi32(s6_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s8_8x16b = _mm_srai_epi32(s7_8x16b,  FILTER_PREC);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s9_8x16b = _mm_packus_epi32(s8_8x16b, zero_8x16b);
            s9_8x16b = _mm_min_epi16(s9_8x16b, clip_tmp_8x16b);
            s9_8x16b = _mm_max_epi16(s9_8x16b, zero_8x16b);
            _mm_storel_epi64((__m128i *)(pu2_dst + (0 * dst_strd)), s9_8x16b);

            /* ROW 2 */
            s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
            s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
            s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_8_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(5 * src_strd)));

            /* load 8 pixel values */
            s2_9_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(6 * src_strd)));

            s3_4_16x8b = _mm_unpacklo_epi16(s2_8_16x8b, s2_9_16x8b);

            s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

            s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
            s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
            s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

            s27_8x16b = _mm_add_epi32(s26_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s28_8x16b = _mm_srai_epi32(s27_8x16b,  FILTER_PREC);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s29_8x16b = _mm_packus_epi32(s28_8x16b, zero_8x16b);

            s29_8x16b = _mm_min_epi16(s29_8x16b, clip_tmp_8x16b);
            s29_8x16b = _mm_max_epi16(s29_8x16b, zero_8x16b);

            _mm_storel_epi64((__m128i *)(pu2_dst + (2 * dst_strd)), s29_8x16b);
            /* ROW 1 */
            s4_0_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

            s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );

            s4_1_16x8b = _mm_unpacklo_epi16(s2_3_16x8b, s2_4_16x8b);

            s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );

            s4_2_16x8b = _mm_unpacklo_epi16(s2_5_16x8b, s2_6_16x8b);

            s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

            s4_3_16x8b = _mm_unpacklo_epi16(s2_7_16x8b, s2_8_16x8b);

            s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

            s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
            s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
            s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

            s17_8x16b = _mm_add_epi32(s16_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s18_8x16b = _mm_srai_epi32(s17_8x16b,  FILTER_PREC);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s19_8x16b = _mm_packus_epi32(s18_8x16b, zero_8x16b);

            s19_8x16b = _mm_min_epi16(s19_8x16b, clip_tmp_8x16b);

            s19_8x16b = _mm_max_epi16(s19_8x16b, zero_8x16b);
            _mm_storel_epi64((__m128i *)(pu2_dst + (1 * dst_strd)), s19_8x16b);
            /* ROW 3 */
            s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
            s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
            s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_10_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(7 * src_strd)));

            s4_4_16x8b = _mm_unpacklo_epi16(s2_9_16x8b, s2_10_16x8b);

            s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

            s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
            s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
            s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

            s37_8x16b = _mm_add_epi32(s36_8x16b, offset_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s38_8x16b = _mm_srai_epi32(s37_8x16b,  SHIFT_14_MINUS_BIT_DEPTH);

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s39_8x16b = _mm_packus_epi32(s38_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 2 */
            // Clipping
            s39_8x16b = _mm_min_epi16(s39_8x16b, clip_tmp_8x16b);

            s39_8x16b = _mm_max_epi16(s39_8x16b, zero_8x16b);

            _mm_storel_epi64((__m128i *)(pu2_dst + (3 * dst_strd)), s39_8x16b);

            pu2_src += (8 * src_strd);
            pu2_dst += (4 * dst_strd);

            for(row = 4; row < ht; row += 4)
            {

                PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);

                s3_0_16x8b = s3_2_16x8b;
                s3_1_16x8b = s3_3_16x8b;
                s3_2_16x8b = s3_4_16x8b;

                s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
                s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
                s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );

                /* load 16 pixel values from (cur_row + 4)th row */
                s2_0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));

                s3_3_16x8b = _mm_unpacklo_epi16(s2_10_16x8b, s2_0_16x8b);
                s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

                s4_0_16x8b = s4_2_16x8b;
                s4_1_16x8b = s4_3_16x8b;
                s4_2_16x8b = s4_4_16x8b;

                s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
                s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
                s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

                s7_8x16b = _mm_add_epi32(s6_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s8_8x16b = _mm_srai_epi32(s7_8x16b,  SHIFT_14_MINUS_BIT_DEPTH);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s9_8x16b = _mm_packus_epi32(s8_8x16b, zero_8x16b);

                s9_8x16b = _mm_min_epi16(s9_8x16b, clip_tmp_8x16b);

                s9_8x16b = _mm_max_epi16(s9_8x16b, zero_8x16b);

                _mm_storel_epi64((__m128i *)(pu2_dst + (0 * dst_strd)), s9_8x16b);

                /* row + 2 */
                s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
                s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
                s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

                /* load 16 pixel values from (cur_row + 5)th row */
                s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + src_strd));

                /* load 16 pixel values from (cur_row + 6)th row */
                s2_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + (2 * src_strd)));

                /* unpacking (cur_row + 5)th row and (cur_row + 6)th row */
                s3_4_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

                s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

                s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
                s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
                s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

                s27_8x16b = _mm_add_epi32(s26_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s28_8x16b = _mm_srai_epi32(s27_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s29_8x16b = _mm_packus_epi32(s28_8x16b, zero_8x16b);
                s29_8x16b = _mm_min_epi16(s29_8x16b, clip_tmp_8x16b);

                s29_8x16b = _mm_max_epi16(s29_8x16b, zero_8x16b);

                _mm_storel_epi64((__m128i *)(pu2_dst + (2 * dst_strd)), s29_8x16b);
                /* row + 1 */
                s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
                s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
                s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

                /* unpacking (cur_row + 4)th row and (cur_row + 5)th row */
                s4_3_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
                s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

                s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
                s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
                s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

                s17_8x16b = _mm_add_epi32(s16_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s18_8x16b = _mm_srai_epi32(s17_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s19_8x16b = _mm_packus_epi32(s18_8x16b, zero_8x16b);

                s19_8x16b = _mm_min_epi16(s19_8x16b, clip_tmp_8x16b);

                s19_8x16b = _mm_max_epi16(s19_8x16b, zero_8x16b);

                _mm_storel_epi64((__m128i *)(pu2_dst + (1 * dst_strd)), s19_8x16b);
                /* row + 3 */
                s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
                s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
                s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

                /* load 16 pixel values from (cur_row + 7)th row */
                s2_3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + (3 * src_strd)));

                /* unpacking (cur_row + 6)th row and (cur_row + 7)th row */
                s4_4_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

                s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

                s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
                s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
                s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

                s37_8x16b = _mm_add_epi32(s36_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s38_8x16b = _mm_srai_epi32(s37_8x16b,  SHIFT_14_MINUS_BIT_DEPTH);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s39_8x16b = _mm_packus_epi32(s38_8x16b, zero_8x16b);

                /* store 8 8-bit output values */
                /* Store the output pixels of (cur_row+3) */
                // Clipping
                s39_8x16b = _mm_min_epi16(s39_8x16b, clip_tmp_8x16b);

                s39_8x16b = _mm_max_epi16(s39_8x16b, zero_8x16b);

                _mm_storel_epi64((__m128i *)(pu2_dst + (3 * dst_strd)), s39_8x16b);

                s2_10_16x8b = s2_3_16x8b;

                pu2_src += 4 * src_strd; /* pointer update */
                pu2_dst += 4 * dst_strd; /* pointer update */
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     Interprediction luma filter for horizontal input for 16 bit output
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to  by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and  writes to the location pointed
*    by 'pu2_dst'  The output is downshifted by 6 and clipped to 8 bits
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_luma_horz_w16out_sse42(UWORD16 *pu2_src,
                                                 WORD16 *pi2_dst,
                                                 WORD32 src_strd,
                                                 WORD32 dst_strd,
                                                 WORD8 *pi1_coeff,
                                                 WORD32 ht,
                                                 WORD32 wd,
                                                 UWORD8 bit_depth)
{
    WORD32 row, col;

    /* all 128 bit registers are named with a suffix mxnb, where m is the */
    /* number of n bits packed in the register */
    __m128i zero_8x16b;
    __m128i src_temp1_16x8b, src_temp2_16x8b, src_temp3_16x8b, src_temp4_16x8b, src_temp5_16x8b, src_temp6_16x8b;
    __m128i src_temp7_16x8b, src_temp8_16x8b, src_temp9_16x8b, src_temp0_16x8b;
    __m128i src_temp11_16x8b, src_temp12_16x8b, src_temp13_16x8b, src_temp14_16x8b, src_temp15_16x8b, src_temp16_16x8b;
    __m128i res_temp1_8x16b, res_temp2_8x16b, res_temp3_8x16b, res_temp4_8x16b, res_temp5_8x16b, res_temp6_8x16b, res_temp7_8x16b, res_temp8_8x16b;
    __m128i res_temp9_8x16b, res_temp0_8x16b;
    __m128i res_temp11_8x16b, res_temp12_8x16b, res_temp13_8x16b, res_temp14_8x16b, res_temp15_8x16b, res_temp16_8x16b;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b, coeff4_5_8x16b, coeff6_7_8x16b;
    __m128i sign_reg;

    ASSERT(wd%4 == 0); /* checking assumption */

    PREFETCH((char const*)(pu2_src+(0 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(1 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(2 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(3 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(4 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(5 * src_strd)), _MM_HINT_T0);

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    src_temp1_16x8b = _mm_loadl_epi64((__m128i *)pi1_coeff);
    zero_8x16b = _mm_set1_epi32(0);

    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, src_temp1_16x8b);
    src_temp2_16x8b  = _mm_unpacklo_epi8(src_temp1_16x8b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    coeff4_5_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(2, 2, 2, 2));  /* pi1_coeff[4] */
    coeff6_7_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(3, 3, 3, 3));  /* pi1_coeff[4] */
    if(0 == (ht & 1)) /* ht multiple of 2 case */
    {
        if(0 == (wd & 7)) /* wd = multiple of 8 case */
        {
            for(row = 0; row < ht; row += 1)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                for(col = 0; col < wd; col += 8)
                {
                    /* load 8 pixel values from row 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));   /* row = 0 */

                    /* load 8 pixel values from row 1 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset)); /* row = 1 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp7_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */
                    res_temp7_8x16b = _mm_madd_epi16(src_temp7_16x8b, coeff0_1_8x16b);
                    /* row = 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp8_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);
                    res_temp8_8x16b = _mm_madd_epi16(src_temp8_16x8b, coeff2_3_8x16b); /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp9_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */
                    res_temp9_8x16b = _mm_madd_epi16(src_temp9_16x8b, coeff4_5_8x16b);

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b = _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp0_16x8b = _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */
                    res_temp0_8x16b = _mm_madd_epi16(src_temp0_16x8b, coeff6_7_8x16b);

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp7_8x16b, res_temp8_8x16b);
                    res_temp7_8x16b = _mm_add_epi32(res_temp9_8x16b, res_temp0_8x16b);
                    res_temp8_8x16b = _mm_add_epi32(res_temp6_8x16b, res_temp7_8x16b);

                    res_temp6_8x16b = _mm_srai_epi32(res_temp5_8x16b, (FILTER_PREC-14+bit_depth)); /* row = 0 */
                    res_temp7_8x16b = _mm_srai_epi32(res_temp8_8x16b, (FILTER_PREC-14+bit_depth));
                    res_temp5_8x16b = _mm_packs_epi32(res_temp6_8x16b, res_temp7_8x16b);       /* row = 0 */

                    /* to store the 1st 4 pixels res. */
                    _mm_storeu_si128((__m128i *)(pi2_dst + offset), res_temp5_8x16b);

                    offset += 8; /* To pointer updates */
                }
                pu2_src += 1*src_strd; /* pointer updates */
                pi2_dst += 1*dst_strd; /* pointer updates */
            }
        }
        else /* wd = multiple of 4 case */
        {
            for(row = 0; row < ht; row += 2)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                for(col = 0; col < wd; col += 4)
                {
                    /* load 16 pixel values from 12:-3 pos. relative to cur. pos. */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));         /* row = 0 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset)); /* row = 1 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));                 /* row = 0 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_srai_epi32(res_temp5_8x16b, (FILTER_PREC-14+bit_depth));          /* row = 0 */
                    res_temp5_8x16b = _mm_packs_epi32(res_temp6_8x16b, res_temp6_8x16b);       /* row = 0 */

                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+1));                 /* row =1 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row =1 */
                    src_temp13_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp11_8x16b = _mm_madd_epi16(src_temp13_16x8b, coeff0_1_8x16b);    /* row = 1 */
                    /* row =1 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+2));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+3));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp14_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp12_8x16b = _mm_madd_epi16(src_temp14_16x8b, coeff2_3_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+4));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+5));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp13_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff4_5_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+6));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+7));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp16_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp14_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff6_7_8x16b);    /* row = 1 */

                    res_temp15_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);
                    res_temp16_8x16b = _mm_add_epi32(res_temp13_8x16b, res_temp14_8x16b);
                    res_temp15_8x16b = _mm_add_epi32(res_temp15_8x16b, res_temp16_8x16b);

                    res_temp16_8x16b = _mm_srai_epi32(res_temp15_8x16b, (FILTER_PREC-14+bit_depth));           /* row = 1 */
                    res_temp15_8x16b = _mm_packs_epi32(res_temp16_8x16b, res_temp16_8x16b);       /* row = 1 */

                    /* to store the 1st 4 pixels res. */
                    _mm_storel_epi64((__m128i *)(pi2_dst + offset), res_temp5_8x16b);
                    _mm_storel_epi64((__m128i *)(pi2_dst + dst_strd + offset), res_temp15_8x16b);

                    offset += 4; /* To pointer updates */
                }
                pu2_src += 2*src_strd;  /* Pointer update */
                pi2_dst += 2*dst_strd;  /* Pointer update */
            }
        }
    }
    else /* odd ht */
    {
        if(0 == (wd & 7)) /* multiple of 8 case */
        {
            for(row = 0; row < ht; row++)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                for(col = 0; col < wd; col += 8)
                {
                    /* load 8 pixel values from row 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));   /* row = 0 */

                    /* load 8 pixel values from row 1 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset)); /* row = 1 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp7_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */
                    res_temp7_8x16b = _mm_madd_epi16(src_temp7_16x8b, coeff0_1_8x16b);
                    /* row = 0 */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp8_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);
                    res_temp8_8x16b = _mm_madd_epi16(src_temp8_16x8b, coeff2_3_8x16b); /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp9_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */
                    res_temp9_8x16b = _mm_madd_epi16(src_temp9_16x8b, coeff4_5_8x16b);

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b = _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    src_temp0_16x8b = _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */
                    res_temp0_8x16b = _mm_madd_epi16(src_temp0_16x8b, coeff6_7_8x16b);

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_add_epi32(res_temp7_8x16b, res_temp8_8x16b);
                    res_temp7_8x16b = _mm_add_epi32(res_temp9_8x16b, res_temp0_8x16b);
                    res_temp8_8x16b = _mm_add_epi32(res_temp6_8x16b, res_temp7_8x16b);

                    res_temp6_8x16b = _mm_srai_epi32(res_temp5_8x16b, (FILTER_PREC-14+bit_depth)); /* row = 0 */
                    res_temp7_8x16b = _mm_srai_epi32(res_temp8_8x16b, (FILTER_PREC-14+bit_depth));
                    res_temp5_8x16b = _mm_packs_epi32(res_temp6_8x16b, res_temp7_8x16b);       /* row = 0 */

                    _mm_storeu_si128((__m128i *)(pi2_dst + offset), res_temp5_8x16b);

                    offset += 8; /* To pointer updates */
                }
                pu2_src += src_strd;    /* pointer updates */
                pi2_dst += dst_strd;    /* pointer updates */
            }
        }
        else  /* wd = multiple of 4 case */
        {
            for(row = 0; row < (ht-1); row += 2)
            {
                int offset =0;

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                for(col = 0; col < wd; col += 4)
                {
                    /* load 16 pixel values from 12:-3 pos. relative to cur. pos. */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));         /* row = 0 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset)); /* row = 1 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));                 /* row = 0 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_srai_epi32(res_temp5_8x16b, (FILTER_PREC-14+bit_depth));          /* row = 0 */
                    res_temp5_8x16b = _mm_packs_epi32(res_temp6_8x16b, res_temp6_8x16b);       /* row = 0 */

                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+1));                 /* row =1 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row =1 */
                    src_temp13_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp11_8x16b = _mm_madd_epi16(src_temp13_16x8b, coeff0_1_8x16b);    /* row = 1 */
                    /* row =1 */
                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+2));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+3));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp14_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp12_8x16b = _mm_madd_epi16(src_temp14_16x8b, coeff2_3_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+4));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+5));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp13_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff4_5_8x16b);    /* row = 1 */

                    src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+6));                 /* row = 1 */
                    src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+src_strd - 3+ offset+7));                /* row = 1 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row =1 */
                    src_temp16_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b); /* row = 1 */
                    res_temp14_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff6_7_8x16b);    /* row = 1 */

                    res_temp15_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);
                    res_temp16_8x16b = _mm_add_epi32(res_temp13_8x16b, res_temp14_8x16b);
                    res_temp15_8x16b = _mm_add_epi32(res_temp15_8x16b, res_temp16_8x16b);

                    res_temp16_8x16b = _mm_srai_epi32(res_temp15_8x16b, (FILTER_PREC-14+bit_depth));           /* row = 1 */
                    res_temp15_8x16b = _mm_packs_epi32(res_temp16_8x16b, res_temp16_8x16b);       /* row = 1 */

                    /* to store the 1st 4 pixels res. */
                    _mm_storel_epi64((__m128i *)(pi2_dst + offset), res_temp5_8x16b);
                    _mm_storel_epi64((__m128i *)(pi2_dst + dst_strd + offset), res_temp15_8x16b);

                    offset += 4; /* To pointer updates */
                }
                pu2_src += 2*src_strd;  /* Pointer update */
                pi2_dst += 2*dst_strd;  /* Pointer update */
            }
            {
                /* last repeat at outside the loop */
                int offset =0;
                for(col = 0; col < wd; col += 4)
                {
                    /* load 16 pixel values from 12:-3 pos. relative to cur. pos. */
                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset));         /* row = 0 */

                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+1));                 /* row = 0 */
                    /* pix. |5:-2|4:-3| to do two dot-products at same time */              /* row = 0 */
                    src_temp3_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp1_8x16b = _mm_madd_epi16(src_temp3_16x8b, coeff0_1_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+2));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+3));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp4_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp2_8x16b = _mm_madd_epi16(src_temp4_16x8b, coeff2_3_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+4));                /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+5));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp3_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff4_5_8x16b);    /* row = 0 */

                    src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+6));                 /* row = 0 */
                    src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src - 3+ offset+7));                  /* row = 0 */
                    /* pix. |7:0|6:-1| to do two dot-products at same time */               /* row = 0 */
                    src_temp6_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b); /* row = 0 */
                    res_temp4_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff6_7_8x16b);    /* row = 0 */

                    res_temp5_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                    res_temp6_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);
                    res_temp5_8x16b = _mm_add_epi32(res_temp5_8x16b, res_temp6_8x16b);

                    res_temp6_8x16b = _mm_srai_epi32(res_temp5_8x16b, (FILTER_PREC-14+bit_depth));          /* row = 0 */
                    res_temp5_8x16b = _mm_packs_epi32(res_temp6_8x16b, res_temp6_8x16b);       /* row = 0 */

                    _mm_storel_epi64((__m128i *)(pi2_dst + offset), res_temp5_8x16b);

                    offset += 4; /* To pointer updates */
                }
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*      Interprediction luma filter for vertical 16bit output
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
*    the elements pointed by 'pu1_src' and  writes to the location pointed by
*    'pu1_dst'  No downshifting or clipping is done and the output is  used as
*    an input for weighted prediction
*
* @param[in] pu1_src
*  UWORD8 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_luma_vert_w16out_sse42(UWORD16 *pu2_src,
                                                 WORD16 *pi2_dst,
                                                 WORD32 src_strd,
                                                 WORD32 dst_strd,
                                                 WORD8 *pi1_coeff,
                                                 WORD32 ht,
                                                 WORD32 wd,
                                                 UWORD8 bit_depth)
{
    WORD32 row, col;
    UWORD16 *pu2_src_copy;
    WORD16 *pi2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b, coeff4_5_8x16b, coeff6_7_8x16b;
    __m128i s0_8x16b, s1_8x16b, s2_8x16b, s3_8x16b, s4_8x16b, s5_8x16b, s6_8x16b, s7_8x16b, s8_8x16b, s9_8x16b;
    __m128i s2_0_16x8b, s2_1_16x8b, s2_2_16x8b, s2_3_16x8b, s2_4_16x8b, s2_5_16x8b, s2_6_16x8b, s2_7_16x8b, s2_8_16x8b, s2_9_16x8b, s2_10_16x8b;
    __m128i s3_0_16x8b, s3_1_16x8b, s3_2_16x8b, s3_3_16x8b, s3_4_16x8b, s3_5_16x8b, s3_6_16x8b, s3_7_16x8b, s3_8_16x8b;
    __m128i s4_0_16x8b, s4_1_16x8b, s4_2_16x8b, s4_3_16x8b, s4_4_16x8b, s4_5_16x8b, s4_6_16x8b, s4_7_16x8b, s4_8_16x8b;
    __m128i s10_8x16b, s11_8x16b, s12_8x16b, s13_8x16b, s14_8x16b, s15_8x16b, s16_8x16b, s17_8x16b, s18_8x16b, s19_8x16b;
    __m128i s20_8x16b, s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b, s26_8x16b, s27_8x16b, s28_8x16b, s29_8x16b;
    __m128i s30_8x16b, s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b, s36_8x16b, s37_8x16b, s38_8x16b, s39_8x16b;

    __m128i zero_8x16b, offset_8x16b, mask_low_32b, mask_high_96b;
    __m128i sign_reg;
    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128(); /* for saturated clipping */
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    coeff4_5_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(2, 2, 2, 2));  /* pi1_coeff[4] */
    coeff6_7_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(3, 3, 3, 3));  /* pi1_coeff[4] */
    /* seting  values in register */
    offset_8x16b = _mm_set1_epi32((1<<(FILTER_PREC-1))); /* for offset addition */
    mask_low_32b = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000);
    mask_high_96b = _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF);

    /* outer for loop starts from here */
    if(wd %8 == 0)
    { /* wd = multiple of 8 case */

        pu2_src_copy = pu2_src;
        pi2_dst_copy = pi2_dst;

        for(col = 0; col < wd; col += 8)
        {
            pu2_src = pu2_src_copy + col;
            pi2_dst = pi2_dst_copy + col;

            PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);

            /* load 8 pixel values. */
            s2_0_16x8b  = _mm_loadu_si128((__m128i *)(pu2_src+(-3 * src_strd)));

            /* load 8 pixel values */
            s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-2 * src_strd)));

            s3_0_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
            s3_4_16x8b = _mm_unpackhi_epi16(s2_0_16x8b, s2_1_16x8b);

            s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
            s4_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff0_1_8x16b );
            /* load 8 pixel values */
            s2_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

            /* load 8 pixel values */
            s2_3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

            s3_1_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);
            s3_5_16x8b = _mm_unpackhi_epi16(s2_2_16x8b, s2_3_16x8b);

            s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
            s5_8x16b = _mm_madd_epi16(s3_5_16x8b, coeff2_3_8x16b );

            /* load 8 pixel values */
            s2_4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

            /* load 8 pixel values */
            s2_5_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

            s3_2_16x8b = _mm_unpacklo_epi16(s2_4_16x8b, s2_5_16x8b);
            s3_6_16x8b = _mm_unpackhi_epi16(s2_4_16x8b, s2_5_16x8b);

            s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );
            s6_8x16b = _mm_madd_epi16(s3_6_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_6_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

            /* load 8 pixel values */
            s2_7_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(4 * src_strd)));

            s3_3_16x8b = _mm_unpacklo_epi16(s2_6_16x8b, s2_7_16x8b);
            s3_7_16x8b = _mm_unpackhi_epi16(s2_6_16x8b, s2_7_16x8b);

            s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );
            s7_8x16b = _mm_madd_epi16(s3_7_16x8b, coeff6_7_8x16b );

            s0_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
            s2_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
            s4_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);
            s6_8x16b = _mm_add_epi32(s6_8x16b, s7_8x16b);
            s0_8x16b = _mm_add_epi32(s0_8x16b, s2_8x16b);
            s4_8x16b = _mm_add_epi32(s4_8x16b, s6_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s7_8x16b = _mm_srai_epi32(s0_8x16b,  (FILTER_PREC-14+bit_depth));
            s8_8x16b = _mm_srai_epi32(s4_8x16b,  (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s9_8x16b = _mm_packs_epi32(s7_8x16b, s8_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 0 */
            _mm_storeu_si128((__m128i *)(pi2_dst + (0 * dst_strd)), s9_8x16b);
            /* ROW 2 */
            s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
            s24_8x16b = _mm_madd_epi16(s3_5_16x8b, coeff0_1_8x16b );
            s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
            s25_8x16b = _mm_madd_epi16(s3_6_16x8b, coeff2_3_8x16b );
            s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );
            s26_8x16b = _mm_madd_epi16(s3_7_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_8_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(5 * src_strd)));

            /* load 8 pixel values */
            s2_9_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(6 * src_strd)));

            s3_4_16x8b = _mm_unpacklo_epi16(s2_8_16x8b, s2_9_16x8b);
            s3_8_16x8b = _mm_unpackhi_epi16(s2_8_16x8b, s2_9_16x8b);

            s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );
            s27_8x16b = _mm_madd_epi16(s3_8_16x8b, coeff6_7_8x16b );

            s20_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
            s22_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
            s24_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);
            s26_8x16b = _mm_add_epi32(s26_8x16b, s27_8x16b);
            s20_8x16b = _mm_add_epi32(s20_8x16b, s22_8x16b);
            s24_8x16b = _mm_add_epi32(s24_8x16b, s26_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s27_8x16b = _mm_srai_epi32(s20_8x16b,  (FILTER_PREC-14+bit_depth));
            s28_8x16b = _mm_srai_epi32(s24_8x16b,  (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s29_8x16b = _mm_packs_epi32(s27_8x16b, s28_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 2 */
            _mm_storeu_si128((__m128i *)(pi2_dst + (2 * dst_strd)), s29_8x16b);
            /* ROW 1 */
            s4_0_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);
            s4_4_16x8b = _mm_unpackhi_epi16(s2_1_16x8b, s2_2_16x8b);

            s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
            s14_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff0_1_8x16b );

            s4_1_16x8b = _mm_unpacklo_epi16(s2_3_16x8b, s2_4_16x8b);
            s4_5_16x8b = _mm_unpackhi_epi16(s2_3_16x8b, s2_4_16x8b);

            s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
            s15_8x16b = _mm_madd_epi16(s4_5_16x8b, coeff2_3_8x16b );

            s4_2_16x8b = _mm_unpacklo_epi16(s2_5_16x8b, s2_6_16x8b);
            s4_6_16x8b = _mm_unpackhi_epi16(s2_5_16x8b, s2_6_16x8b);

            s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );
            s16_8x16b = _mm_madd_epi16(s4_6_16x8b, coeff4_5_8x16b );

            s4_3_16x8b = _mm_unpacklo_epi16(s2_7_16x8b, s2_8_16x8b);
            s4_7_16x8b = _mm_unpackhi_epi16(s2_7_16x8b, s2_8_16x8b);

            s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );
            s17_8x16b = _mm_madd_epi16(s4_7_16x8b, coeff6_7_8x16b );

            s10_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
            s12_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
            s14_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);
            s16_8x16b = _mm_add_epi32(s16_8x16b, s17_8x16b);
            s10_8x16b = _mm_add_epi32(s10_8x16b, s12_8x16b);
            s14_8x16b = _mm_add_epi32(s14_8x16b, s16_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s17_8x16b = _mm_srai_epi32(s10_8x16b,  (FILTER_PREC-14+bit_depth));
            s18_8x16b = _mm_srai_epi32(s14_8x16b,  (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s19_8x16b = _mm_packs_epi32(s17_8x16b, s18_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 1 */
            _mm_storeu_si128((__m128i *)(pi2_dst + (1 * dst_strd)), s19_8x16b);

            /* ROW 3 */
            s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
            s34_8x16b = _mm_madd_epi16(s4_5_16x8b, coeff0_1_8x16b );
            s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
            s35_8x16b = _mm_madd_epi16(s4_6_16x8b, coeff2_3_8x16b );
            s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );
            s36_8x16b = _mm_madd_epi16(s4_7_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_10_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(7 * src_strd)));

            s4_4_16x8b = _mm_unpacklo_epi16(s2_9_16x8b, s2_10_16x8b);
            s4_8_16x8b = _mm_unpackhi_epi16(s2_9_16x8b, s2_10_16x8b);

            s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );
            s37_8x16b = _mm_madd_epi16(s4_8_16x8b, coeff6_7_8x16b );

            s30_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
            s32_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
            s34_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);
            s36_8x16b = _mm_add_epi32(s36_8x16b, s37_8x16b);
            s30_8x16b = _mm_add_epi32(s30_8x16b, s32_8x16b);
            s34_8x16b = _mm_add_epi32(s34_8x16b, s36_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s37_8x16b = _mm_srai_epi32(s30_8x16b,  (FILTER_PREC-14+bit_depth));
            s38_8x16b = _mm_srai_epi32(s34_8x16b,  (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s39_8x16b = _mm_packs_epi32(s37_8x16b, s38_8x16b);

            _mm_storeu_si128((__m128i *)(pi2_dst + (3 * dst_strd)), s39_8x16b);

            pu2_src += (8 * src_strd);
            pi2_dst += (4 * dst_strd);

            for(row = 4; row < ht; row += 4)
            {
                PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);

                s3_0_16x8b = s3_2_16x8b;
                s3_1_16x8b = s3_3_16x8b;
                s3_2_16x8b = s3_4_16x8b;
                s3_4_16x8b = s3_6_16x8b;
                s3_5_16x8b = s3_7_16x8b;
                s3_6_16x8b = s3_8_16x8b;

                s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
                s4_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff0_1_8x16b );
                s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
                s5_8x16b = _mm_madd_epi16(s3_5_16x8b, coeff2_3_8x16b );
                s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );
                s6_8x16b = _mm_madd_epi16(s3_6_16x8b, coeff4_5_8x16b );

                /* load 8 pixel values from (cur_row + 4)th row */
                s2_0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));

                s3_3_16x8b = _mm_unpacklo_epi16(s2_10_16x8b, s2_0_16x8b);
                s3_7_16x8b = _mm_unpackhi_epi16(s2_10_16x8b, s2_0_16x8b);
                s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );
                s7_8x16b = _mm_madd_epi16(s3_7_16x8b, coeff6_7_8x16b );

                s4_0_16x8b = s4_2_16x8b;
                s4_1_16x8b = s4_3_16x8b;
                s4_2_16x8b = s4_4_16x8b;
                s4_4_16x8b = s4_6_16x8b;
                s4_5_16x8b = s4_7_16x8b;
                s4_6_16x8b = s4_8_16x8b;

                s0_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
                s2_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
                s4_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);
                s6_8x16b = _mm_add_epi32(s6_8x16b, s7_8x16b);
                s0_8x16b = _mm_add_epi32(s0_8x16b, s2_8x16b);
                s4_8x16b = _mm_add_epi32(s4_8x16b, s6_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s7_8x16b = _mm_srai_epi32(s0_8x16b,  (FILTER_PREC-14+bit_depth));
                s8_8x16b = _mm_srai_epi32(s4_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s9_8x16b = _mm_packs_epi32(s7_8x16b, s8_8x16b);

                /* store 8 8-bit output values */
                /* Store the output pixels of row 0 */
                _mm_storeu_si128((__m128i *)(pi2_dst + (0 * dst_strd)), s9_8x16b);
                /* row + 2 */
                s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
                s24_8x16b = _mm_madd_epi16(s3_5_16x8b, coeff0_1_8x16b );
                s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
                s25_8x16b = _mm_madd_epi16(s3_6_16x8b, coeff2_3_8x16b );
                s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );
                s26_8x16b = _mm_madd_epi16(s3_7_16x8b, coeff4_5_8x16b );

                /* load 8 pixel values from (cur_row + 5)th row */
                s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + src_strd));

                /* load 8 pixel values from (cur_row + 6)th row */
                s2_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + (2 * src_strd)));

                /* unpacking (cur_row + 5)th row and (cur_row + 6)th row */
                s3_4_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);
                s3_8_16x8b = _mm_unpackhi_epi16(s2_1_16x8b, s2_2_16x8b);

                s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );
                s27_8x16b = _mm_madd_epi16(s3_8_16x8b, coeff6_7_8x16b );

                s20_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
                s22_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
                s24_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);
                s26_8x16b = _mm_add_epi32(s26_8x16b, s27_8x16b);
                s20_8x16b = _mm_add_epi32(s20_8x16b, s22_8x16b);
                s24_8x16b = _mm_add_epi32(s24_8x16b, s26_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s27_8x16b = _mm_srai_epi32(s20_8x16b,  (FILTER_PREC-14+bit_depth));
                s28_8x16b = _mm_srai_epi32(s24_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s29_8x16b = _mm_packs_epi32(s27_8x16b, s28_8x16b);

                /* store 8 8-bit output values */
                /* Store the output pixels of row 2 */
                _mm_storeu_si128((__m128i *)(pi2_dst + (2 * dst_strd)), s29_8x16b);

                /* row + 1 */
                s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
                s14_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff0_1_8x16b );
                s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
                s15_8x16b = _mm_madd_epi16(s4_5_16x8b, coeff2_3_8x16b );
                s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );
                s16_8x16b = _mm_madd_epi16(s4_6_16x8b, coeff4_5_8x16b );

                /* unpacking (cur_row + 4)th row and (cur_row + 5)th row */
                s4_3_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
                s4_7_16x8b = _mm_unpackhi_epi16(s2_0_16x8b, s2_1_16x8b);
                s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );
                s17_8x16b = _mm_madd_epi16(s4_7_16x8b, coeff6_7_8x16b );

                s10_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
                s12_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
                s14_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);
                s16_8x16b = _mm_add_epi32(s16_8x16b, s17_8x16b);
                s10_8x16b = _mm_add_epi32(s10_8x16b, s12_8x16b);
                s14_8x16b = _mm_add_epi32(s14_8x16b, s16_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s17_8x16b = _mm_srai_epi32(s10_8x16b,  (FILTER_PREC-14+bit_depth));
                s18_8x16b = _mm_srai_epi32(s14_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s19_8x16b = _mm_packs_epi32(s17_8x16b, s18_8x16b);

                /* store 8 8-bit output values */
                /* Store the output pixels of (cur_row + 1) */
                _mm_storeu_si128((__m128i *)(pi2_dst + (1 * dst_strd)), s19_8x16b);
                /* row + 3 */
                s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
                s34_8x16b = _mm_madd_epi16(s4_5_16x8b, coeff0_1_8x16b );
                s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
                s35_8x16b = _mm_madd_epi16(s4_6_16x8b, coeff2_3_8x16b );
                s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );
                s36_8x16b = _mm_madd_epi16(s4_7_16x8b, coeff4_5_8x16b );

                /* load 8 pixel values from (cur_row + 7)th row */
                s2_3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + (3 * src_strd)));

                /* unpacking (cur_row + 6)th row and (cur_row + 7)th row */
                s4_4_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);
                s4_8_16x8b = _mm_unpackhi_epi16(s2_2_16x8b, s2_3_16x8b);

                s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );
                s37_8x16b = _mm_madd_epi16(s4_8_16x8b, coeff6_7_8x16b );

                s30_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
                s32_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
                s34_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);
                s36_8x16b = _mm_add_epi32(s36_8x16b, s37_8x16b);
                s30_8x16b = _mm_add_epi32(s30_8x16b, s32_8x16b);
                s34_8x16b = _mm_add_epi32(s34_8x16b, s36_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s37_8x16b = _mm_srai_epi32(s30_8x16b,  (FILTER_PREC-14+bit_depth));
                s38_8x16b = _mm_srai_epi32(s34_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s39_8x16b = _mm_packs_epi32(s37_8x16b, s38_8x16b);

                /* store 8 8-bit output values */
                /* Store the output pixels of (cur_row+3) */

                _mm_storeu_si128((__m128i *)(pi2_dst + (3 * dst_strd)), s39_8x16b);

                s2_10_16x8b = s2_3_16x8b;

                pu2_src += 4 * src_strd; /* pointer update */
                pi2_dst += 4 * dst_strd; /* pointer update */
            }
        }
    }
    else
    {
        /* wd = multiple of 4 case */

        pu2_src_copy = pu2_src;
        pi2_dst_copy = pi2_dst;

        for(col = 0; col < wd; col += 4)
        {

            pu2_src = pu2_src_copy + col;
            pi2_dst = pi2_dst_copy + col;

            PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);

            /* load 8 pixel values */
            s2_0_16x8b  = _mm_loadu_si128((__m128i *)(pu2_src+(-3 * src_strd)));

            /* load 8 pixel values */
            s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-2 * src_strd)));

            s3_0_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);

            s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
            /* load 8 pixel values */
            s2_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

            /* load 8 pixel values */
            s2_3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

            s3_1_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

            s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
            /* load 8 pixel values */
            s2_4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

            /* load 8 pixel values */
            s2_5_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

            s3_2_16x8b = _mm_unpacklo_epi16(s2_4_16x8b, s2_5_16x8b);

            s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );
            /* load 8 pixel values */
            s2_6_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

            /* load 8 pixel values */
            s2_7_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(4 * src_strd)));

            s3_3_16x8b = _mm_unpacklo_epi16(s2_6_16x8b, s2_7_16x8b);

            s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

            s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
            s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
            s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s8_8x16b = _mm_srai_epi32(s6_8x16b,  (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s9_8x16b = _mm_packs_epi32(s8_8x16b, zero_8x16b);
            _mm_storel_epi64((__m128i *)(pi2_dst + (0 * dst_strd)), s9_8x16b);

            /* ROW 2 */
            s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
            s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
            s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_8_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(5 * src_strd)));

            /* load 8 pixel values */
            s2_9_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(6 * src_strd)));

            s3_4_16x8b = _mm_unpacklo_epi16(s2_8_16x8b, s2_9_16x8b);

            s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

            s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
            s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
            s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s28_8x16b = _mm_srai_epi32(s26_8x16b,  (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s29_8x16b = _mm_packs_epi32(s28_8x16b, zero_8x16b);
            _mm_storel_epi64((__m128i *)(pi2_dst + (2 * dst_strd)), s29_8x16b);

            /* ROW 1 */
            s4_0_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

            s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );

            s4_1_16x8b = _mm_unpacklo_epi16(s2_3_16x8b, s2_4_16x8b);

            s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );

            s4_2_16x8b = _mm_unpacklo_epi16(s2_5_16x8b, s2_6_16x8b);

            s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

            s4_3_16x8b = _mm_unpacklo_epi16(s2_7_16x8b, s2_8_16x8b);

            s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

            s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
            s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
            s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s18_8x16b = _mm_srai_epi32(s16_8x16b,  (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s19_8x16b = _mm_packs_epi32(s18_8x16b, zero_8x16b);
            _mm_storel_epi64((__m128i *)(pi2_dst + (1 * dst_strd)), s19_8x16b);

            /* ROW 3 */
            s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
            s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
            s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

            /* load 8 pixel values */
            s2_10_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+(7 * src_strd)));

            s4_4_16x8b = _mm_unpacklo_epi16(s2_9_16x8b, s2_10_16x8b);

            s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

            s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
            s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
            s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s38_8x16b = _mm_srai_epi32(s36_8x16b, (FILTER_PREC-14+bit_depth));

            /* i2_tmp = CLIP_U8(i2_tmp); */
            s39_8x16b = _mm_packs_epi32(s38_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 2 */

            _mm_storel_epi64((__m128i *)(pi2_dst + (3 * dst_strd)), s39_8x16b);

            pu2_src += (8 * src_strd);
            pi2_dst += (4 * dst_strd);

            for(row = 4; row < ht; row += 4)
            {
                PREFETCH((char const*)(pu2_src+(8 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(9 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(10 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(11 * src_strd)), _MM_HINT_T0);


                s3_0_16x8b = s3_2_16x8b;
                s3_1_16x8b = s3_3_16x8b;
                s3_2_16x8b = s3_4_16x8b;

                s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
                s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
                s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );

                /* load 16 pixel values from (cur_row + 4)th row */
                s2_0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));

                s3_3_16x8b = _mm_unpacklo_epi16(s2_10_16x8b, s2_0_16x8b);
                s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

                s4_0_16x8b = s4_2_16x8b;
                s4_1_16x8b = s4_3_16x8b;
                s4_2_16x8b = s4_4_16x8b;

                s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
                s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
                s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s8_8x16b = _mm_srai_epi32(s6_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s9_8x16b = _mm_packs_epi32(s8_8x16b, zero_8x16b);
                _mm_storel_epi64((__m128i *)(pi2_dst + (0 * dst_strd)), s9_8x16b);
                /* row + 2 */
                s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
                s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
                s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

                /* load 16 pixel values from (cur_row + 5)th row */
                s2_1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + src_strd));

                /* load 16 pixel values from (cur_row + 6)th row */
                s2_2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + (2 * src_strd)));

                /* unpacking (cur_row + 5)th row and (cur_row + 6)th row */
                s3_4_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

                s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

                s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
                s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
                s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s28_8x16b = _mm_srai_epi32(s26_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s29_8x16b = _mm_packs_epi32(s28_8x16b, zero_8x16b);
                _mm_storel_epi64((__m128i *)(pi2_dst + (2 * dst_strd)), s29_8x16b);

                /* row + 1 */
                s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
                s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
                s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

                /* unpacking (cur_row + 4)th row and (cur_row + 5)th row */
                s4_3_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
                s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

                s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
                s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
                s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s18_8x16b = _mm_srai_epi32(s16_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s19_8x16b = _mm_packs_epi32(s18_8x16b, zero_8x16b);
                _mm_storel_epi64((__m128i *)(pi2_dst + (1 * dst_strd)), s19_8x16b);

                /* row + 3 */
                s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
                s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
                s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

                /* load 16 pixel values from (cur_row + 7)th row */
                s2_3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src + (3 * src_strd)));

                /* unpacking (cur_row + 6)th row and (cur_row + 7)th row */
                s4_4_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

                s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

                s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
                s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
                s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s38_8x16b = _mm_srai_epi32(s36_8x16b,  (FILTER_PREC-14+bit_depth));

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s39_8x16b = _mm_packs_epi32(s38_8x16b, zero_8x16b);

                /* store 8 8-bit output values */
                /* Store the output pixels of (cur_row+3) */
                _mm_storel_epi64((__m128i *)(pi2_dst + (3 * dst_strd)), s39_8x16b);

                s2_10_16x8b = s2_3_16x8b;

                pu2_src += 4 * src_strd; /* pointer update */
                pi2_dst += 4 * dst_strd; /* pointer update */
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*
*        Luma vertical filter for 16bit input.
*
* @par Description:
*   Applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
*   the elements pointed by 'pu2_src' and  writes to the location pointed by
*   'pu2_dst'  Input is 16 bits  The filter output is downshifted by 12 and
*   clipped to lie  between 0 and 255
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_luma_vert_w16inp_sse42(WORD16 *pi2_src,
                                                 UWORD16 *pu2_dst,
                                                 WORD32 src_strd,
                                                 WORD32 dst_strd,
                                                 WORD8 *pi1_coeff,
                                                 WORD32 ht,
                                                 WORD32 wd,
                                                 UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD16 *pi2_src_copy;
    UWORD16 *pu2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b, coeff4_5_8x16b, coeff6_7_8x16b;
    __m128i s0_8x16b, s1_8x16b, s2_8x16b, s3_8x16b, s4_8x16b, s5_8x16b, s6_8x16b, s8_8x16b, s9_8x16b;
    __m128i s2_0_16x8b, s2_1_16x8b, s2_2_16x8b, s2_3_16x8b, s2_4_16x8b, s2_5_16x8b, s2_6_16x8b, s2_7_16x8b, s2_8_16x8b, s2_9_16x8b, s2_10_16x8b;
    __m128i s3_0_16x8b, s3_1_16x8b, s3_2_16x8b, s3_3_16x8b, s3_4_16x8b;
    __m128i s4_0_16x8b, s4_1_16x8b, s4_2_16x8b, s4_3_16x8b, s4_4_16x8b;
    __m128i s10_8x16b, s11_8x16b, s12_8x16b, s13_8x16b, s14_8x16b, s15_8x16b, s16_8x16b, s18_8x16b, s19_8x16b;
    __m128i s20_8x16b, s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b, s26_8x16b, s28_8x16b, s29_8x16b;
    __m128i s30_8x16b, s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b, s36_8x16b, s38_8x16b, s39_8x16b;
    __m128i clip_tmp_8x16b;
    __m128i zero_8x16b, offset_8x16b, mask_low_32b, mask_high_96b, sign_reg;

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128();
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    coeff4_5_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(2, 2, 2, 2));  /* pi1_coeff[4] */
    coeff6_7_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(3, 3, 3, 3));  /* pi1_coeff[4] */


    /* seting values in register */
    offset_8x16b = _mm_set1_epi32((1<<(FILTER_PREC-1))); /* for offset addition */
    mask_low_32b = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000);
    mask_high_96b = _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF);


    pi2_src_copy = pi2_src;
    pu2_dst_copy = pu2_dst;

    /* outer for loop starts from here */
    for(col = 0; col < wd; col += 4)
    {
        pi2_src = pi2_src_copy + col;
        pu2_dst = pu2_dst_copy + col;

        /* load 4 pixel values */
        s2_0_16x8b  = _mm_loadl_epi64((__m128i *)(pi2_src+(-3 * src_strd)));

        /* load 4 pixel values */
        s2_1_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(-2 * src_strd)));

        s3_0_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);

        s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );

        /* load 4 pixel values */
        s2_2_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(-1 * src_strd)));

        /* load 4 pixel values */
        s2_3_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(0 * src_strd)));

        s3_1_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

        s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );

        /* load 4 pixel values */
        s2_4_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(1 * src_strd)));

        /* load 4 pixel values */
        s2_5_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(2 * src_strd)));

        s3_2_16x8b = _mm_unpacklo_epi16(s2_4_16x8b, s2_5_16x8b);

        s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );

        /* load 4 pixel values */
        s2_6_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(3 * src_strd)));

        /* load 4 pixel values */
        s2_7_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(4 * src_strd)));

        s3_3_16x8b = _mm_unpacklo_epi16(s2_6_16x8b, s2_7_16x8b);

        s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

        s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
        s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
        s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s8_8x16b = _mm_srai_epi32(s6_8x16b,  (14-bit_depth));

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
        s9_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);

        /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s8_8x16b = _mm_srai_epi32(s9_8x16b,  FILTER_PREC);

        s8_8x16b = _mm_packus_epi32(s8_8x16b, zero_8x16b);

        // Clipping
        clip_tmp_8x16b = _mm_cmpeq_epi16(s8_8x16b, s8_8x16b);
        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
        s8_8x16b = _mm_min_epi16(s8_8x16b, clip_tmp_8x16b);

        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
        s8_8x16b = _mm_max_epi16(s8_8x16b, clip_tmp_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 0 */
        _mm_storel_epi64((__m128i *)(pu2_dst), s8_8x16b);

        /* ROW 2 */
        s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
        s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
        s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

        /* load 4 pixel values */
        s2_8_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(5 * src_strd)));

        /* load 4 pixel values */
        s2_9_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(6 * src_strd)));

        s3_4_16x8b = _mm_unpacklo_epi16(s2_8_16x8b, s2_9_16x8b);

        s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

        s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
        s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
        s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s28_8x16b = _mm_srai_epi32(s26_8x16b,  (14-bit_depth));

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
        s29_8x16b = _mm_add_epi32(s28_8x16b, offset_8x16b);

        /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s28_8x16b = _mm_srai_epi32(s29_8x16b,  FILTER_PREC);

        s28_8x16b = _mm_packus_epi32(s28_8x16b, zero_8x16b);

        // Clipping
        clip_tmp_8x16b = _mm_cmpeq_epi16(s28_8x16b, s28_8x16b);
        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
        s28_8x16b = _mm_min_epi16(s28_8x16b, clip_tmp_8x16b);

        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
        s28_8x16b = _mm_max_epi16(s28_8x16b, clip_tmp_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 2 */
        _mm_storel_epi64((__m128i *)(pu2_dst + (2 * dst_strd)), s28_8x16b);

        /* ROW 1 */
        s4_0_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

        s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );

        s4_1_16x8b = _mm_unpacklo_epi16(s2_3_16x8b, s2_4_16x8b);

        s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );

        s4_2_16x8b = _mm_unpacklo_epi16(s2_5_16x8b, s2_6_16x8b);

        s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

        s4_3_16x8b = _mm_unpacklo_epi16(s2_7_16x8b, s2_8_16x8b);

        s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

        s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
        s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
        s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s18_8x16b = _mm_srai_epi32(s16_8x16b,  (14-bit_depth));

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
        s19_8x16b = _mm_add_epi32(s18_8x16b, offset_8x16b);

        /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s18_8x16b = _mm_srai_epi32(s19_8x16b,  FILTER_PREC);

        s18_8x16b = _mm_packus_epi32(s18_8x16b, zero_8x16b);

        // Clipping
        clip_tmp_8x16b = _mm_cmpeq_epi16(s18_8x16b, s18_8x16b);
        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
        s18_8x16b = _mm_min_epi16(s18_8x16b, clip_tmp_8x16b);

        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
        s18_8x16b = _mm_max_epi16(s18_8x16b, clip_tmp_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 1 */
        _mm_storel_epi64((__m128i *)(pu2_dst + (dst_strd)), s18_8x16b);

        /* ROW 3 */
        s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
        s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
        s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

        /* load 4 pixel values */
        s2_10_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(7 * src_strd)));

        s4_4_16x8b = _mm_unpacklo_epi16(s2_9_16x8b, s2_10_16x8b);

        s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

        s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
        s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
        s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s38_8x16b = _mm_srai_epi32(s36_8x16b,  (14-bit_depth));

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
        s39_8x16b = _mm_add_epi32(s38_8x16b, offset_8x16b);

        /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
        s38_8x16b = _mm_srai_epi32(s39_8x16b,  FILTER_PREC);

        s38_8x16b = _mm_packus_epi32(s38_8x16b, zero_8x16b);

        // Clipping
        clip_tmp_8x16b = _mm_cmpeq_epi16(s38_8x16b, s38_8x16b);
        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
        s38_8x16b = _mm_min_epi16(s38_8x16b, clip_tmp_8x16b);

        clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
        s38_8x16b = _mm_max_epi16(s38_8x16b, clip_tmp_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 2 */
        _mm_storel_epi64((__m128i *)(pu2_dst + (3 * dst_strd)), s38_8x16b);

        pi2_src += (8 * src_strd);
        pu2_dst += (4 * dst_strd);

        for(row = 4; row < ht; row += 4)
        {
            s3_0_16x8b = s3_2_16x8b;
            s3_1_16x8b = s3_3_16x8b;
            s3_2_16x8b = s3_4_16x8b;

            s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
            s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
            s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );

            /* load 4 pixel values from (cur_row + 4)th row */
            s2_0_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src));

            s3_3_16x8b = _mm_unpacklo_epi16(s2_10_16x8b, s2_0_16x8b);
            s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

            s4_0_16x8b = s4_2_16x8b;
            s4_1_16x8b = s4_3_16x8b;
            s4_2_16x8b = s4_4_16x8b;

            s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
            s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
            s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s8_8x16b = _mm_srai_epi32(s6_8x16b,  (14-bit_depth));

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
            s9_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);

            /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s8_8x16b = _mm_srai_epi32(s9_8x16b,  FILTER_PREC);

            s8_8x16b = _mm_packus_epi32(s8_8x16b, zero_8x16b);

            // Clipping
            clip_tmp_8x16b = _mm_cmpeq_epi16(s8_8x16b, s8_8x16b);
            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
            s8_8x16b = _mm_min_epi16(s8_8x16b, clip_tmp_8x16b);

            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
            s8_8x16b = _mm_max_epi16(s8_8x16b, clip_tmp_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 4 */
            _mm_storel_epi64((__m128i *)(pu2_dst), s8_8x16b);

            /* row + 2 */
            s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
            s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
            s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

            /* load 4 pixel values from (cur_row + 5)th row */
            s2_1_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src + src_strd));

            /* load 4 pixel values from (cur_row + 6)th row */
            s2_2_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src + (2 * src_strd)));

            /* unpacking (cur_row + 5)th row and (cur_row + 6)th row */
            s3_4_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

            s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

            s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
            s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
            s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s28_8x16b = _mm_srai_epi32(s26_8x16b,  (14-bit_depth));

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
            s29_8x16b = _mm_add_epi32(s28_8x16b, offset_8x16b);

            /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s28_8x16b = _mm_srai_epi32(s29_8x16b,  FILTER_PREC);

            s28_8x16b = _mm_packs_epi32(s28_8x16b, zero_8x16b);

            // Clipping
            clip_tmp_8x16b = _mm_cmpeq_epi16(s28_8x16b, s28_8x16b);
            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
            s28_8x16b = _mm_min_epi16(s28_8x16b, clip_tmp_8x16b);

            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
            s28_8x16b = _mm_max_epi16(s28_8x16b, clip_tmp_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of (cur_row+2) */
            _mm_storel_epi64((__m128i *)(pu2_dst + (2 * dst_strd)), s28_8x16b);

            /* row + 1 */
            s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
            s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
            s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

            /* unpacking (cur_row + 4)th row and (cur_row + 5)th row */
            s4_3_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
            s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

            s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
            s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
            s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s18_8x16b = _mm_srai_epi32(s16_8x16b,  (14-bit_depth));

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
            s19_8x16b = _mm_add_epi32(s18_8x16b, offset_8x16b);

            /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s18_8x16b = _mm_srai_epi32(s19_8x16b,  FILTER_PREC);

            s18_8x16b = _mm_packs_epi32(s18_8x16b, zero_8x16b);

            // Clipping
            clip_tmp_8x16b = _mm_cmpeq_epi16(s18_8x16b, s18_8x16b);
            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
            s18_8x16b = _mm_min_epi16(s18_8x16b, clip_tmp_8x16b);

            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
            s18_8x16b = _mm_max_epi16(s18_8x16b, clip_tmp_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of (cur_row + 1) */
            _mm_storel_epi64((__m128i *)(pu2_dst + dst_strd), s18_8x16b);

            /* row + 3 */
            s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
            s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
            s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

            /* load 4 pixel values from (cur_row + 7)th row */
            s2_3_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src + (3 * src_strd)));

            /* unpacking (cur_row + 6)th row and (cur_row + 7)th row */
            s4_4_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

            s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

            s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
            s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
            s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s38_8x16b = _mm_srai_epi32(s36_8x16b,  (14-bit_depth));

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
            s39_8x16b = _mm_add_epi32(s38_8x16b, offset_8x16b);

            /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
            s38_8x16b = _mm_srai_epi32(s39_8x16b,  FILTER_PREC);

            s38_8x16b = _mm_packs_epi32(s38_8x16b, zero_8x16b);

            // Clipping
            clip_tmp_8x16b = _mm_cmpeq_epi16(s38_8x16b, s38_8x16b);
            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
            s38_8x16b = _mm_min_epi16(s38_8x16b, clip_tmp_8x16b);

            clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
            s38_8x16b = _mm_max_epi16(s38_8x16b, clip_tmp_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of (cur_row+3) */
            _mm_storel_epi64((__m128i *)(pu2_dst + (3 * dst_strd)), s38_8x16b);

            s2_10_16x8b = s2_3_16x8b;

            pi2_src += 4 * src_strd; /* pointer update */
            pu2_dst += 4 * dst_strd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*      Luma prediction filter for vertical 16bit input & output
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
*    the elements pointed by 'pu1_src' and  writes to the location pointed by
*    'pu1_dst'  Input is 16 bits  The filter output is downshifted by 6 and
*    8192 is  subtracted to store it as a 16 bit number  The output is used as
*    a input to weighted prediction
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_luma_vert_w16inp_w16out_sse42(WORD16 *pi2_src,
                                                        WORD16 *pi2_dst,
                                                        WORD32 src_strd,
                                                        WORD32 dst_strd,
                                                        WORD8 *pi1_coeff,
                                                        WORD32 ht,
                                                        WORD32 wd,
                                                        UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD16 *pi2_src_copy;
    WORD16 *pi2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b, coeff4_5_8x16b, coeff6_7_8x16b;
    __m128i s0_8x16b, s1_8x16b, s2_8x16b, s3_8x16b, s4_8x16b, s5_8x16b, s6_8x16b, s8_8x16b, s9_8x16b;
    __m128i s2_0_16x8b, s2_1_16x8b, s2_2_16x8b, s2_3_16x8b, s2_4_16x8b, s2_5_16x8b, s2_6_16x8b, s2_7_16x8b, s2_8_16x8b, s2_9_16x8b, s2_10_16x8b;
    __m128i s3_0_16x8b, s3_1_16x8b, s3_2_16x8b, s3_3_16x8b, s3_4_16x8b;
    __m128i s4_0_16x8b, s4_1_16x8b, s4_2_16x8b, s4_3_16x8b, s4_4_16x8b;
    __m128i s10_8x16b, s11_8x16b, s12_8x16b, s13_8x16b, s14_8x16b, s15_8x16b, s16_8x16b, s18_8x16b, s19_8x16b;
    __m128i s20_8x16b, s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b, s26_8x16b, s28_8x16b, s29_8x16b;
    __m128i s30_8x16b, s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b, s36_8x16b, s38_8x16b, s39_8x16b;

    __m128i zero_8x16b, offset_8x16b, sign_reg;

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128();
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    coeff4_5_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(2, 2, 2, 2));  /* pi1_coeff[4] */
    coeff6_7_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(3, 3, 3, 3));  /* pi1_coeff[4] */

    /* seting values in register */
    offset_8x16b = _mm_set1_epi32(OFFSET14); /* for offset addition */

    pi2_src_copy = pi2_src;
    pi2_dst_copy = pi2_dst;

    /* outer for loop starts from here */
    for(col = 0; col < wd; col += 4)
    {
        pi2_src = pi2_src_copy + col;
        pi2_dst = pi2_dst_copy + col;

        /* load 4 pixel values */
        s2_0_16x8b  = _mm_loadl_epi64((__m128i *)(pi2_src+(-3 * src_strd)));

        /* load 4 pixel values */
        s2_1_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(-2 * src_strd)));

        s3_0_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);

        s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );

        /* load 4 pixel values */
        s2_2_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(-1 * src_strd)));

        /* load 4 pixel values */
        s2_3_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(0 * src_strd)));

        s3_1_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

        s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );

        /* load 4 pixel values */
        s2_4_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(1 * src_strd)));

        /* load 4 pixel values */
        s2_5_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(2 * src_strd)));

        s3_2_16x8b = _mm_unpacklo_epi16(s2_4_16x8b, s2_5_16x8b);

        s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );

        /* load 4 pixel values */
        s2_6_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(3 * src_strd)));

        /* load 4 pixel values */
        s2_7_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(4 * src_strd)));

        s3_3_16x8b = _mm_unpacklo_epi16(s2_6_16x8b, s2_7_16x8b);

        s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

        s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
        s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
        s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
        s8_8x16b = _mm_srai_epi32(s6_8x16b,  FILTER_PREC);

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
        s9_8x16b = _mm_sub_epi32(s8_8x16b, offset_8x16b);

        s8_8x16b = _mm_packs_epi32(s9_8x16b, zero_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 0 */
        _mm_storel_epi64((__m128i *)(pi2_dst), s8_8x16b);

        /* ROW 2 */
        s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
        s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
        s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

        /* load 4 pixel values */
        s2_8_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(5 * src_strd)));

        /* load 4 pixel values */
        s2_9_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(6 * src_strd)));

        s3_4_16x8b = _mm_unpacklo_epi16(s2_8_16x8b, s2_9_16x8b);

        s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

        s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
        s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
        s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
        s28_8x16b = _mm_srai_epi32(s26_8x16b,  FILTER_PREC);

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
        s29_8x16b = _mm_sub_epi32(s28_8x16b, offset_8x16b);

        s28_8x16b = _mm_packs_epi32(s29_8x16b, zero_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 2 */
        _mm_storel_epi64((__m128i *)(pi2_dst + (2 * dst_strd)), s28_8x16b);

        /* ROW 1 */
        s4_0_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

        s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );

        s4_1_16x8b = _mm_unpacklo_epi16(s2_3_16x8b, s2_4_16x8b);

        s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );

        s4_2_16x8b = _mm_unpacklo_epi16(s2_5_16x8b, s2_6_16x8b);

        s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

        s4_3_16x8b = _mm_unpacklo_epi16(s2_7_16x8b, s2_8_16x8b);

        s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

        s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
        s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
        s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
        s18_8x16b = _mm_srai_epi32(s16_8x16b,  FILTER_PREC);

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
        s19_8x16b = _mm_sub_epi32(s18_8x16b, offset_8x16b);

        s18_8x16b = _mm_packs_epi32(s19_8x16b, zero_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 1 */
        _mm_storel_epi64((__m128i *)(pi2_dst + (dst_strd)), s18_8x16b);

        /* ROW 3 */
        s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
        s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
        s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

        /* load 4 pixel values */
        s2_10_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src+(7 * src_strd)));

        s4_4_16x8b = _mm_unpacklo_epi16(s2_9_16x8b, s2_10_16x8b);

        s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

        s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
        s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
        s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

        /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
        s38_8x16b = _mm_srai_epi32(s36_8x16b,  FILTER_PREC);

        /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
        s39_8x16b = _mm_sub_epi32(s38_8x16b, offset_8x16b);

        s38_8x16b = _mm_packs_epi32(s39_8x16b, zero_8x16b);

        /* store 8 8-bit output values */
        /* Store the output pixels of row 2 */
        _mm_storel_epi64((__m128i *)(pi2_dst + (3 * dst_strd)), s38_8x16b);

        pi2_src += (8 * src_strd);
        pi2_dst += (4 * dst_strd);

        for(row = 4; row < ht; row += 4)
        {
            s3_0_16x8b = s3_2_16x8b;
            s3_1_16x8b = s3_3_16x8b;
            s3_2_16x8b = s3_4_16x8b;

            s0_8x16b = _mm_madd_epi16(s3_0_16x8b, coeff0_1_8x16b );
            s1_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff2_3_8x16b );
            s2_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff4_5_8x16b );

            /* load 4 pixel values from (cur_row + 4)th row */
            s2_0_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src));

            s3_3_16x8b = _mm_unpacklo_epi16(s2_10_16x8b, s2_0_16x8b);
            s3_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff6_7_8x16b );

            s4_0_16x8b = s4_2_16x8b;
            s4_1_16x8b = s4_3_16x8b;
            s4_2_16x8b = s4_4_16x8b;

            s4_8x16b = _mm_add_epi32(s0_8x16b, s1_8x16b);
            s5_8x16b = _mm_add_epi32(s2_8x16b, s3_8x16b);
            s6_8x16b = _mm_add_epi32(s4_8x16b, s5_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
            s8_8x16b = _mm_srai_epi32(s6_8x16b,  FILTER_PREC);

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
            s9_8x16b = _mm_sub_epi32(s8_8x16b, offset_8x16b);

            s8_8x16b = _mm_packs_epi32(s9_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of row 4 */
            _mm_storel_epi64((__m128i *)(pi2_dst), s8_8x16b);

            /* row + 2 */
            s20_8x16b = _mm_madd_epi16(s3_1_16x8b, coeff0_1_8x16b );
            s21_8x16b = _mm_madd_epi16(s3_2_16x8b, coeff2_3_8x16b );
            s22_8x16b = _mm_madd_epi16(s3_3_16x8b, coeff4_5_8x16b );

            /* load 4 pixel values from (cur_row + 5)th row */
            s2_1_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src + src_strd));

            /* load 4 pixel values from (cur_row + 6)th row */
            s2_2_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src + (2 * src_strd)));

            /* unpacking (cur_row + 5)th row and (cur_row + 6)th row */
            s3_4_16x8b = _mm_unpacklo_epi16(s2_1_16x8b, s2_2_16x8b);

            s23_8x16b = _mm_madd_epi16(s3_4_16x8b, coeff6_7_8x16b );

            s24_8x16b = _mm_add_epi32(s20_8x16b, s21_8x16b);
            s25_8x16b = _mm_add_epi32(s22_8x16b, s23_8x16b);
            s26_8x16b = _mm_add_epi32(s24_8x16b, s25_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
            s28_8x16b = _mm_srai_epi32(s26_8x16b,  FILTER_PREC);

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
            s29_8x16b = _mm_sub_epi32(s28_8x16b, offset_8x16b);

            s28_8x16b = _mm_packs_epi32(s29_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of (cur_row+2) */
            _mm_storel_epi64((__m128i *)(pi2_dst + (2 * dst_strd)), s28_8x16b);

            /* row + 1 */
            s10_8x16b = _mm_madd_epi16(s4_0_16x8b, coeff0_1_8x16b );
            s11_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff2_3_8x16b );
            s12_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff4_5_8x16b );

            /* unpacking (cur_row + 4)th row and (cur_row + 5)th row */
            s4_3_16x8b = _mm_unpacklo_epi16(s2_0_16x8b, s2_1_16x8b);
            s13_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff6_7_8x16b );

            s14_8x16b = _mm_add_epi32(s10_8x16b, s11_8x16b);
            s15_8x16b = _mm_add_epi32(s12_8x16b, s13_8x16b);
            s16_8x16b = _mm_add_epi32(s14_8x16b, s15_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
            s18_8x16b = _mm_srai_epi32(s16_8x16b,  FILTER_PREC);

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
            s19_8x16b = _mm_sub_epi32(s18_8x16b, offset_8x16b);

            s18_8x16b = _mm_packs_epi32(s19_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of (cur_row + 1) */
            _mm_storel_epi64((__m128i *)(pi2_dst + dst_strd), s18_8x16b);

            /* row + 3 */
            s30_8x16b = _mm_madd_epi16(s4_1_16x8b, coeff0_1_8x16b );
            s31_8x16b = _mm_madd_epi16(s4_2_16x8b, coeff2_3_8x16b );
            s32_8x16b = _mm_madd_epi16(s4_3_16x8b, coeff4_5_8x16b );

            /* load 4 pixel values from (cur_row + 7)th row */
            s2_3_16x8b = _mm_loadl_epi64((__m128i *)(pi2_src + (3 * src_strd)));

            /* unpacking (cur_row + 6)th row and (cur_row + 7)th row */
            s4_4_16x8b = _mm_unpacklo_epi16(s2_2_16x8b, s2_3_16x8b);

            s33_8x16b = _mm_madd_epi16(s4_4_16x8b, coeff6_7_8x16b );

            s34_8x16b = _mm_add_epi32(s30_8x16b, s31_8x16b);
            s35_8x16b = _mm_add_epi32(s32_8x16b, s33_8x16b);
            s36_8x16b = _mm_add_epi32(s34_8x16b, s35_8x16b);

            /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
            s38_8x16b = _mm_srai_epi32(s36_8x16b,  FILTER_PREC);

            /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) - OFFSET_14) */
            s39_8x16b = _mm_sub_epi32(s38_8x16b, offset_8x16b);

            s38_8x16b = _mm_packs_epi32(s39_8x16b, zero_8x16b);

            /* store 8 8-bit output values */
            /* Store the output pixels of (cur_row+3) */
            _mm_storel_epi64((__m128i *)(pi2_dst + (3 * dst_strd)), s38_8x16b);

            s2_10_16x8b = s2_3_16x8b;

            pi2_src += 4 * src_strd; /* pointer update */
            pi2_dst += 4 * dst_strd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*       Interprediction luma filter for copy 16bit output
*
* @par Description:
*    Copies the array of width 'wd' and height 'ht' from the  location pointed
*    by 'src' to the location pointed by 'dst' The output is upshifted by 6
*    bits and is used as input for vertical filtering or weighted prediction
*
* @param[in] pu1_src
*  UWORD8 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_luma_copy_w16out_sse42(UWORD16 *pu2_src,
                                                 WORD16 *pi2_dst,
                                                 WORD32 src_strd,
                                                 WORD32 dst_strd,
                                                 WORD8 *pi1_coeff,
                                                 WORD32 ht,
                                                 WORD32 wd,
                                                 UWORD8 bit_depth)
{
    WORD32 row, col;
    __m128i  src0_16x8b, src1_16x8b, src2_16x8b, src3_16x8b;

    ASSERT(wd%4 == 0); /* checking assumption */
    ASSERT(ht%4 == 0); /* checking assumption */

    if(0 == (wd & 7)) /* multiple of 8 case */
    {
        for(row = 0; row < ht; row += 4)
        {
            for(col = 0; col < wd; col += 8)
            {
                /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                src0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));            /* row =0 */
                src1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                src2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                src3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                src0_16x8b = _mm_slli_epi16(src0_16x8b,  (14 - bit_depth)); /* (pu1_src[col] << SHIFT_14_MINUS_BIT_DEPTH */
                src1_16x8b = _mm_slli_epi16(src1_16x8b,  (14 - bit_depth));
                src2_16x8b = _mm_slli_epi16(src2_16x8b,  (14 - bit_depth));
                src3_16x8b = _mm_slli_epi16(src3_16x8b,  (14 - bit_depth));

                /* storing 16 8-bit output values */
                _mm_storeu_si128((__m128i *)(pi2_dst), src0_16x8b);               /* row =0 */
                _mm_storeu_si128((__m128i *)(pi2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                _mm_storeu_si128((__m128i *)(pi2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                _mm_storeu_si128((__m128i *)(pi2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                pu2_src += 8; /* pointer update */
                pi2_dst += 8; /* pointer update */
            } /* inner for loop ends here(8-output values in single iteration) */

            pu2_src += 4*src_strd - wd; /* pointer update */
            pi2_dst += 4*dst_strd - wd; /* pointer update */
        }
    }
    else /* wd = multiple of 4 case */
    {
        for(row = 0; row < ht; row += 4)
        {
            for(col = 0; col < wd; col += 4)
            {
                /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                src0_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src));            /* row =0 */
                src1_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                src2_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                src3_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                src0_16x8b = _mm_slli_epi16(src0_16x8b,  (14 - bit_depth)); /* (pu1_src[col] << SHIFT_14_MINUS_BIT_DEPTH */
                src1_16x8b = _mm_slli_epi16(src1_16x8b,  (14 - bit_depth));
                src2_16x8b = _mm_slli_epi16(src2_16x8b,  (14 - bit_depth));
                src3_16x8b = _mm_slli_epi16(src3_16x8b,  (14 - bit_depth));

                /* storing 16 8-bit output values */
                _mm_storel_epi64((__m128i *)(pi2_dst), src0_16x8b);               /* row =0 */
                _mm_storel_epi64((__m128i *)(pi2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                _mm_storel_epi64((__m128i *)(pi2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                _mm_storel_epi64((__m128i *)(pi2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                pu2_src += 4; /* pointer update */
                pi2_dst += 4; /* pointer update */
            } /* inner for loop ends here(4-output values in single iteration) */

            pu2_src += 4*src_strd - wd; /* pointer update */
            pi2_dst += 4*dst_strd - wd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     Chroma interprediction filter for horizontal input
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to  by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and  writes to the location pointed
*    by 'pu2_dst'  The output is downshifted by 6 and clipped to 8 bits
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_horz_sse42(UWORD16 *pu2_src,
                                            UWORD16 *pu2_dst,
                                            WORD32 src_strd,
                                            WORD32 dst_strd,
                                            WORD8 *pi1_coeff,
                                            WORD32 ht,
                                            WORD32 wd,
                                            UWORD8 bit_depth)
{
    WORD32 row, col;

    __m128i coeff0_1_8x16b, coeff2_3_8x16b;
    __m128i src_temp1_16x8b, src_temp2_16x8b, src_temp3_16x8b, src_temp4_16x8b, src_temp5_16x8b, src_temp6_16x8b, src_temp7_16x8b, src_temp8_16x8b;
    __m128i src_temp11_16x8b, src_temp12_16x8b, src_temp13_16x8b, src_temp14_16x8b, src_temp15_16x8b, src_temp16_16x8b, src_temp17_16x8b, src_temp18_16x8b;
    __m128i res_temp1_8x16b, res_temp2_8x16b, res_temp3_8x16b, res_temp4_8x16b, res_temp6_8x16b;
    __m128i res_temp11_8x16b, res_temp12_8x16b, res_temp13_8x16b, res_temp14_8x16b, res_temp15_8x16b, res_temp16_8x16b;
    __m128i zero_8x16b, sign_reg, clip_tmp_8x16b, offset_8x16b;
    PREFETCH((char const*)(pu2_src+(0 * src_strd)), _MM_HINT_T0)
        PREFETCH((char const*)(pu2_src+(1 * src_strd)), _MM_HINT_T0)
        PREFETCH((char const*)(pu2_src+(2 * src_strd)), _MM_HINT_T0)
        PREFETCH((char const*)(pu2_src+(3 * src_strd)), _MM_HINT_T0)
        PREFETCH((char const*)(pu2_src+(4 * src_strd)), _MM_HINT_T0)
        PREFETCH((char const*)(pu2_src+(5 * src_strd)), _MM_HINT_T0)

        ASSERT(wd%2 == 0); /* checking assumption */

    /* loading four 8-bit coefficients */
    src_temp1_16x8b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    offset_8x16b = _mm_set1_epi32((1<<(FILTER_PREC-1))); /* for offset addition */

    zero_8x16b = _mm_setzero_si128();
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, src_temp1_16x8b);
    src_temp2_16x8b  = _mm_unpacklo_epi8(src_temp1_16x8b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    /* outer for loop starts from here */
    if(wd %2==0 && wd%4 !=0)
    {
        for(row = 0; row < ht; row += 2)
        {
            int offset =0;

            PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

            for(col = 0; col < 2*wd; col += 4)
            {
                /* load 16 pixel values from row 0 */
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset)); /* pu1_src[col + (i-1) * 2] */

                /* load 16 pixel values from row 1 */
                src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset)); /* pu1_src[col + (i-1) * 2] */

                /* Derive the source pixels for processing the 2nd pixel */
                src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+2));

                src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b);

                /* Derive the source pixels for processing the 3rd pixel */
                src_temp3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+4));

                /* Derive the source pixels for processing the 4th pixel */
                src_temp4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+6));

                src_temp6_16x8b= _mm_unpacklo_epi16(src_temp3_16x8b, src_temp4_16x8b);

                /* Derive the source pixels for processing the 2nd pixel */
                src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+2));

                src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b);

                /* Derive the source pixels for processing the 3rd pixel */
                src_temp13_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+4));
                /* Derive the source pixels for processing the 4th pixel */
                src_temp14_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+6));

                src_temp16_16x8b= _mm_unpacklo_epi16(src_temp13_16x8b, src_temp14_16x8b);

                res_temp1_8x16b  = _mm_madd_epi16(src_temp5_16x8b, coeff0_1_8x16b);
                res_temp2_8x16b  = _mm_madd_epi16(src_temp6_16x8b, coeff2_3_8x16b);
                res_temp11_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff0_1_8x16b);
                res_temp12_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp3_8x16b  = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp13_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);

                res_temp3_8x16b = _mm_add_epi32(res_temp3_8x16b, offset_8x16b);            /* row = 0 */
                res_temp4_8x16b = _mm_srai_epi32(res_temp3_8x16b, FILTER_PREC); /* row = 0 */
                res_temp14_8x16b = _mm_add_epi32(res_temp13_8x16b, offset_8x16b);            /* row = 0 */
                res_temp15_8x16b = _mm_srai_epi32(res_temp14_8x16b, FILTER_PREC); /* row = 0 */
                res_temp13_8x16b = _mm_packus_epi32(res_temp4_8x16b, res_temp15_8x16b);       /* row = 0 */

                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(res_temp13_8x16b, res_temp13_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                res_temp13_8x16b = _mm_min_epi16(res_temp13_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                res_temp13_8x16b = _mm_max_epi16(res_temp13_8x16b, clip_tmp_8x16b);

                /* store 4 16-bit values */
                _mm_storel_epi64((__m128i *)(pu2_dst + offset), res_temp13_8x16b); /* pu1_dst[col] = i2_tmp_u */

                res_temp13_8x16b = _mm_srli_si128(res_temp13_8x16b, 8);
                /* store 4 16-bit values */
                _mm_storel_epi64((__m128i *)(pu2_dst + dst_strd + offset), res_temp13_8x16b); /* pu1_dst[col] = i2_tmp_u */

                offset += 4; /* To pointer update */

            } /* inner loop ends here(8- output values in single iteration) */

            pu2_src += 2 * src_strd; /* pointer update */
            pu2_dst += 2 * dst_strd; /* pointer update */
        }
    }
    else
    {
        for(row = 0; row < ht; row += 2)
        {
            int offset =0;

            PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

            for(col = 0; col < 2*wd; col += 8)
            {
                /* load 16 pixel values from row 0 */
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset)); /* pu1_src[col + (i-1) * 2] */

                /* load 16 pixel values from row 1 */
                src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset)); /* pu1_src[col + (i-1) * 2] */

                /* Derive the source pixels for processing the 2nd pixel */
                src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+2));

                src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b);
                src_temp7_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);

                /* Derive the source pixels for processing the 3rd pixel */
                src_temp3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+4));

                /* Derive the source pixels for processing the 4th pixel */
                src_temp4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+6));

                src_temp6_16x8b= _mm_unpacklo_epi16(src_temp3_16x8b, src_temp4_16x8b);
                src_temp8_16x8b= _mm_unpackhi_epi16(src_temp3_16x8b, src_temp4_16x8b);

                res_temp1_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff0_1_8x16b);
                res_temp2_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff2_3_8x16b);
                res_temp3_8x16b = _mm_madd_epi16(src_temp7_16x8b, coeff0_1_8x16b);
                res_temp4_8x16b = _mm_madd_epi16(src_temp8_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp1_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp3_8x16b = _mm_add_epi32(res_temp3_8x16b, res_temp4_8x16b);

                res_temp2_8x16b = _mm_add_epi32(res_temp1_8x16b, offset_8x16b);
                res_temp4_8x16b = _mm_add_epi32(res_temp3_8x16b, offset_8x16b);            /* row = 0 */
                res_temp2_8x16b = _mm_srai_epi32(res_temp2_8x16b, FILTER_PREC); /* row = 0 */
                res_temp4_8x16b = _mm_srai_epi32(res_temp4_8x16b, FILTER_PREC); /* row = 0 */
                res_temp6_8x16b = _mm_packus_epi32(res_temp2_8x16b, res_temp4_8x16b);       /* row = 0 */

                /* store 4 16-bit values */

                /* Derive the source pixels for processing the 2nd pixel of row 1 */
                src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+2));

                src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b);
                src_temp17_16x8b= _mm_unpackhi_epi16(src_temp11_16x8b, src_temp12_16x8b);

                /* Derive the source pixels for processing the 3rd pixel of row 1 */
                src_temp13_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+4));

                /* Derive the source pixels for processing the 4th pixel of row 1 */
                src_temp14_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+6));

                src_temp16_16x8b= _mm_unpacklo_epi16(src_temp13_16x8b, src_temp14_16x8b);
                src_temp18_16x8b= _mm_unpackhi_epi16(src_temp13_16x8b, src_temp14_16x8b);

                res_temp11_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff0_1_8x16b);
                res_temp12_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff2_3_8x16b);
                res_temp13_8x16b = _mm_madd_epi16(src_temp17_16x8b, coeff0_1_8x16b);
                res_temp14_8x16b = _mm_madd_epi16(src_temp18_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp11_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);
                res_temp13_8x16b = _mm_add_epi32(res_temp13_8x16b, res_temp14_8x16b);

                res_temp12_8x16b = _mm_add_epi32(res_temp11_8x16b, offset_8x16b);            /* row = 0 */
                res_temp14_8x16b = _mm_add_epi32(res_temp13_8x16b, offset_8x16b);            /* row = 0 */
                res_temp12_8x16b = _mm_srai_epi32(res_temp12_8x16b, FILTER_PREC); /* row = 0 */
                res_temp14_8x16b = _mm_srai_epi32(res_temp14_8x16b, FILTER_PREC); /* row = 0 */
                res_temp16_8x16b = _mm_packus_epi32(res_temp12_8x16b, res_temp14_8x16b);       /* row = 0 */

                /* store 4 16-bit values */
                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(res_temp6_8x16b, res_temp6_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                res_temp6_8x16b = _mm_min_epi16(res_temp6_8x16b, clip_tmp_8x16b);
                res_temp16_8x16b = _mm_min_epi16(res_temp16_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                res_temp6_8x16b = _mm_max_epi16(res_temp6_8x16b, clip_tmp_8x16b);
                res_temp16_8x16b = _mm_max_epi16(res_temp16_8x16b, clip_tmp_8x16b);

                _mm_storeu_si128((__m128i *)(pu2_dst + offset), res_temp6_8x16b); /* pu1_dst[col] = i2_tmp_u */
                _mm_storeu_si128((__m128i *)(pu2_dst + dst_strd + offset), res_temp16_8x16b); /* pu1_dst[col] = i2_tmp_u */

                offset += 8; /* To pointer update */
            } /* inner loop ends here(8- output values in single iteration) */

            pu2_src += 2 * src_strd; /* pointer update */
            pu2_dst += 2 * dst_strd; /* pointer update */
        }
    }
}
/**
*******************************************************************************
*
* @brief
*     Chroma interprediction filter for vertical input
*
* @par Description:
*    Applies a vertcal filter with coefficients pointed to  by 'pi1_coeff' to
*    the elements pointed by 'pu1_src' and  writes to the location pointed by
*    'pu1_dst'  The output is downshifted by 6 and clipped to 8 bits
*
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_vert_sse42(UWORD16 *pu2_src,
                                            UWORD16 *pu2_dst,
                                            WORD32 src_strd,
                                            WORD32 dst_strd,
                                            WORD8 *pi1_coeff,
                                            WORD32 ht,
                                            WORD32 wd,
                                            UWORD8 bit_depth)
{
    WORD32 row, col;
    UWORD16 *pu2_src_copy;
    UWORD16 *pu2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b;
    __m128i s4_8x16b, s5_8x16b, s6_8x16b, s7_8x16b, s8_8x16b;
    __m128i s11_8x16b, s12_8x16b, s15_8x16b, s16_8x16b;
    __m128i zero_8x16b, offset_8x16b, mask_low_32b, mask_high_96b;
    __m128i s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b;
    __m128i s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b;
    __m128i clip_tmp_8x16b, sign_reg;
    PREFETCH((char const*)(pu2_src+(0 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(1 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(2 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(3 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(4 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(5 * src_strd)), _MM_HINT_T0);

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128();
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */
    /* seting  values in register */

    offset_8x16b = _mm_set1_epi32((1<<(FILTER_PREC-1))); /* for offset addition */
    mask_low_32b = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000);
    mask_high_96b = _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF);

    /* outer for loop starts from here */
    if(wd % 8 == 0)
    {
        /* wd = multiple of 8 case */
        pu2_src_copy = pu2_src;
        pu2_dst_copy = pu2_dst;

        for(col = 0; col < 2 * wd; col += 8)
        {
            pu2_src = pu2_src_copy + col;
            pu2_dst = pu2_dst_copy + col;

            for(row = 0; row < ht; row += 2)
            {
                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                /* load 8 pixel values from -751:-768 pos. relative to cur. pos. */
                s21_8x16b  = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

                /* load 8 pixel values from -495:-512 pos. relative to cur. pos. */
                s22_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

                /* load 8 pixel values from -239:-256 pos. relative to cur. pos. */
                s23_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

                /* load 8 pixel values from 15:0 pos. relative to cur. pos. */
                s24_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s21_8x16b, s22_8x16b);

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s23_8x16b, s24_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b);

                s5_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);

                s31_8x16b = _mm_add_epi32(s35_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
                s6_8x16b = _mm_srai_epi32(s5_8x16b,  FILTER_PREC);

                s32_8x16b = _mm_srai_epi32(s31_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s7_8x16b = _mm_packus_epi32(s6_8x16b, s32_8x16b);

                s25_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s22_8x16b, s23_8x16b);

                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s24_8x16b, s25_8x16b);

                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s5_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);

                s31_8x16b = _mm_add_epi32(s35_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
                s6_8x16b = _mm_srai_epi32(s5_8x16b,  FILTER_PREC);

                s32_8x16b = _mm_srai_epi32(s31_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s8_8x16b = _mm_packus_epi32(s6_8x16b, s32_8x16b);

                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(s7_8x16b, s7_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                s7_8x16b = _mm_min_epi16(s7_8x16b, clip_tmp_8x16b);
                s8_8x16b = _mm_min_epi16(s8_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                s7_8x16b = _mm_max_epi16(s7_8x16b, clip_tmp_8x16b);
                s8_8x16b = _mm_max_epi16(s8_8x16b, clip_tmp_8x16b);

                /* store 8 8-bit output values */
                _mm_storeu_si128((__m128i *)(pu2_dst), s7_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst + dst_strd), s8_8x16b);

                pu2_src += 2 * src_strd;
                pu2_dst += 2 * dst_strd;
            } /* inner for loop ends here(8-output values in single iteration) */
        }
    }
    else if(wd % 4 == 0)
    {
        /* wd = multiple of 8 case */
        for(row = 0; row < ht; row += 2)
        {
            pu2_src_copy = pu2_src;
            pu2_dst_copy = pu2_dst;
            for(col = 0; col < 2 * wd; col += 8)
            {
                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                /* load 8 pixel values from -751:-768 pos. relative to cur. pos. */
                s21_8x16b  = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

                /* load 8 pixel values from -495:-512 pos. relative to cur. pos. */
                s22_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);
                s31_8x16b = _mm_unpackhi_epi16(s21_8x16b, s22_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );
                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                /* load 8 pixel values from -239:-256 pos. relative to cur. pos. */
                s23_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

                /* load 8 pixel values from 15:0 pos. relative to cur. pos. */
                s24_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);
                s33_8x16b = _mm_unpackhi_epi16(s23_8x16b, s24_8x16b);

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );
                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */
                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b);

                s5_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);
                s31_8x16b = _mm_add_epi32(s35_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
                s6_8x16b = _mm_srai_epi32(s5_8x16b,  FILTER_PREC);
                s32_8x16b = _mm_srai_epi32(s31_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s7_8x16b = _mm_packus_epi32(s6_8x16b, s32_8x16b);

                s25_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);
                s31_8x16b = _mm_unpackhi_epi16(s22_8x16b, s23_8x16b);

                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );
                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);
                s33_8x16b = _mm_unpackhi_epi16(s24_8x16b, s25_8x16b);

                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );
                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */
                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b);

                s5_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);
                s31_8x16b = _mm_add_epi32(s35_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
                s6_8x16b = _mm_srai_epi32(s5_8x16b,  FILTER_PREC);
                s33_8x16b = _mm_srai_epi32(s31_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s8_8x16b = _mm_packus_epi32(s6_8x16b, s33_8x16b);

                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(s7_8x16b, s7_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                s7_8x16b = _mm_min_epi16(s7_8x16b, clip_tmp_8x16b);
                s8_8x16b = _mm_min_epi16(s8_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                s7_8x16b = _mm_max_epi16(s7_8x16b, clip_tmp_8x16b);
                s8_8x16b = _mm_max_epi16(s8_8x16b, clip_tmp_8x16b);

                /* store 8 8-bit output values */
                /* pu1_dst[col] = (UWORD8)i2_tmp; */

                _mm_storeu_si128((__m128i *)(pu2_dst), s7_8x16b);
                _mm_storeu_si128((__m128i *)(pu2_dst + dst_strd), s8_8x16b);

                pu2_src += 8;    /* To pointer update */
                pu2_dst += 8;
            } /* inner for loop ends here(8-output values in single iteration) */

            pu2_src = pu2_src_copy + 2 * src_strd; /* pointer update */
            pu2_dst = pu2_dst_copy + 2 * dst_strd; /* pointer update */
        }
    }
    else
    {
        /* wd = multiple of 4 case */
        for(row = 0; row < ht; row += 2)
        {
            pu2_src_copy = pu2_src;
            pu2_dst_copy = pu2_dst;
            for(col = 0; col < 2 * wd; col += 4)
            {
                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                /* load 8 pixel values from -751:-768 pos. relative to cur. pos. */
                s21_8x16b  = _mm_loadl_epi64((__m128i *)(pu2_src+(-1 * src_strd)));

                /* load 8 pixel values from -495:-512 pos. relative to cur. pos. */
                s22_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(0 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                /* load 8 pixel values from -239:-256 pos. relative to cur. pos. */
                s23_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(1 * src_strd)));

                /* load 8 pixel values from 15:0 pos. relative to cur. pos. */
                s24_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(2 * src_strd)));

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s5_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
                s6_8x16b = _mm_srai_epi32(s5_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s7_8x16b = _mm_packus_epi32(s6_8x16b, zero_8x16b);

                /* store 8 8-bit output values */
                /* pu1_dst[col] = (UWORD8)i2_tmp; */

                s25_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);
                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);
                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s5_8x16b = _mm_add_epi32(s8_8x16b, offset_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> FILTER_PREC */
                s6_8x16b = _mm_srai_epi32(s5_8x16b,  FILTER_PREC);

                /* i2_tmp = CLIP_U8(i2_tmp); */
                s8_8x16b = _mm_packus_epi32(s6_8x16b, zero_8x16b);

                /* store 8 8-bit output values */
                /* pu1_dst[col] = (UWORD8)i2_tmp; */
                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(s7_8x16b, s7_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                s7_8x16b = _mm_min_epi16(s7_8x16b, clip_tmp_8x16b);
                s8_8x16b = _mm_min_epi16(s8_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                s7_8x16b = _mm_max_epi16(s7_8x16b, clip_tmp_8x16b);
                s8_8x16b = _mm_max_epi16(s8_8x16b, clip_tmp_8x16b);

                _mm_storel_epi64((__m128i *)(pu2_dst), s7_8x16b);
                _mm_storel_epi64((__m128i *)(pu2_dst + dst_strd), s8_8x16b);

                pu2_src += 4;   /* To pointer update */
                pu2_dst += 4;
            } /* inner for loop ends here(8-output values in single iteration) */

            pu2_src = pu2_src_copy + 2 * src_strd; /* pointer update */
            pu2_dst = pu2_dst_copy + 2 * dst_strd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*       chroma interprediction filter to store horizontal 16bit ouput
*
* @par Description:
*    Applies a horizontal filter with coefficients pointed to  by 'pi1_coeff'
*    to the elements pointed by 'pu2_src' and  writes to the location pointed
*    by 'pu1_dst'  No downshifting or clipping is done and the output is  used
*    as an input for vertical filtering or weighted  prediction
*
* @param[in] pu2_src
*  UWORD8 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/
void ihevc_hbd_inter_pred_chroma_horz_w16out_sse42(UWORD16 *pu2_src,
                                                   WORD16 *pi2_dst,
                                                   WORD32 src_strd,
                                                   WORD32 dst_strd,
                                                   WORD8 *pi1_coeff,
                                                   WORD32 ht,
                                                   WORD32 wd,
                                                   UWORD8 bit_depth)
{
    WORD32 row, col;

    __m128i coeff0_1_8x16b, coeff2_3_8x16b, all_zero;
    __m128i src_temp1_16x8b, src_temp2_16x8b, src_temp3_16x8b, src_temp4_16x8b, src_temp5_16x8b, src_temp6_16x8b, src_temp7_16x8b, src_temp8_16x8b;
    __m128i src_temp11_16x8b, src_temp12_16x8b, src_temp13_16x8b, src_temp14_16x8b, src_temp15_16x8b, src_temp16_16x8b, src_temp17_16x8b, src_temp18_16x8b;
    __m128i res_temp1_8x16b, res_temp2_8x16b, res_temp3_8x16b;
    __m128i res_temp11_8x16b, res_temp12_8x16b, res_temp13_8x16b;
    __m128i sign_reg;
    PREFETCH((char const*)(pu2_src+(0 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(1 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(2 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(3 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(4 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(5 * src_strd)), _MM_HINT_T0);

    ASSERT(wd%2 == 0); /* checking assumption */

    /* loading four 8-bit coefficients and convert 8-bit into 16-bit */
    src_temp1_16x8b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    all_zero = _mm_setzero_si128();

    sign_reg =  _mm_cmpgt_epi8(all_zero, src_temp1_16x8b);
    src_temp2_16x8b  = _mm_unpacklo_epi8(src_temp1_16x8b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(src_temp2_16x8b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */
    /* outer for loop starts from here */
    if(wd %2==0 && wd%4 !=0)
    {
        int offset =0;
        for(row = ht; row >= 2; row -= 2)
        {
            offset =0;
            PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

            for(col = 0; col < 2*wd; col += 4)
            {
                /* load 16 pixel values of row 0 */
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset)); /* pu1_src[col + (i-1) * 2] */

                /* load 16 pixel values of row 1 */
                src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset)); /* pu1_src[col + (i-1) * 2] */

                /* Derive the source pixels for processing the 2nd pixel of row 0 */
                src_temp2_16x8b = _mm_srli_si128(src_temp1_16x8b, 4);

                src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b);

                /* Derive the source pixels for processing the 3rd pixel of row 0 */
                src_temp3_16x8b = _mm_srli_si128(src_temp1_16x8b, 8);

                /* Derive the source pixels for processing the 4th pixel of row 0 */
                src_temp4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+6));

                src_temp6_16x8b= _mm_unpacklo_epi16(src_temp3_16x8b, src_temp4_16x8b);

                /* Derive the source pixels for processing the 2nd pixel of row 1 */
                src_temp12_16x8b = _mm_srli_si128(src_temp11_16x8b, 4);

                src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b);

                /* Derive the source pixels for processing the 3rd pixel of row 1 */
                src_temp13_16x8b = _mm_srli_si128(src_temp11_16x8b, 8);

                /* Derive the source pixels for processing the 4th pixel of row 1 */
                src_temp14_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+6));

                src_temp16_16x8b= _mm_unpacklo_epi16(src_temp13_16x8b, src_temp14_16x8b);

                res_temp1_8x16b  = _mm_madd_epi16(src_temp5_16x8b, coeff0_1_8x16b);
                res_temp2_8x16b  = _mm_madd_epi16(src_temp6_16x8b, coeff2_3_8x16b);
                res_temp11_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff0_1_8x16b);
                res_temp12_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp3_8x16b  = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp13_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);

                res_temp3_8x16b = _mm_srai_epi32(res_temp3_8x16b, (FILTER_PREC - (14-bit_depth)));
                res_temp13_8x16b = _mm_srai_epi32(res_temp13_8x16b, (FILTER_PREC - (14-bit_depth)));
                res_temp13_8x16b = _mm_packs_epi32(res_temp3_8x16b, res_temp13_8x16b);

                res_temp3_8x16b = _mm_srli_si128(res_temp13_8x16b, 8);

                /* store 4 16-bit values */
                _mm_storel_epi64((__m128i *)(pi2_dst + offset), res_temp13_8x16b); /* pi2_dst[col] = i2_tmp_u */

                /* store 4 16-bit values */
                _mm_storel_epi64((__m128i *)(pi2_dst + dst_strd + offset), res_temp3_8x16b); /* pi2_dst[col] = i2_tmp_u */

                offset += 4; /* To pointer update */
            } /* inner loop ends here(8- output values in single iteration) */

            pu2_src += 2 * src_strd; /* pointer update */
            pi2_dst += 2 * dst_strd; /* pointer update */
        }

        /* Epilogue to handle ht= odd case */
        if(row)
        {
            offset = 0;
            for(col = 0; col < 2*wd; col += 4)
            {
                /* load 16 pixel values of row 0 */
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset)); /* pu1_src[col + (i-1) * 2] */

                /* Derive the source pixels for processing the 2nd pixel of row 0 */
                src_temp2_16x8b = _mm_srli_si128(src_temp1_16x8b, 4);

                src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b);

                /* Derive the source pixels for processing the 3rd pixel of row 0 */
                src_temp3_16x8b = _mm_srli_si128(src_temp1_16x8b, 8);

                /* Derive the source pixels for processing the 4th pixel of row 0 */
                src_temp4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+6));

                src_temp6_16x8b= _mm_unpacklo_epi16(src_temp3_16x8b, src_temp4_16x8b);

                res_temp1_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff0_1_8x16b);
                res_temp2_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp3_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp3_8x16b = _mm_srai_epi32(res_temp3_8x16b, (FILTER_PREC - (14-bit_depth)));
                res_temp13_8x16b = _mm_packs_epi32(res_temp3_8x16b, res_temp3_8x16b);

                /* store 4 16-bit values */
                _mm_storel_epi64((__m128i *)(pi2_dst + offset), res_temp13_8x16b); /* pi2_dst[col] = i2_tmp_u */

                offset += 4; /* To pointer update */
            }
        }
    }
    else
    {
        int offset =0;

        for(row = ht; row >= 2; row -= 2)
        {
            offset =0;
            PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
            PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

            for(col = 0; col < 2*wd; col += 8)
            {
                /* load 16 pixel values of row 0 */
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset)); /* pu1_src[col + (i-1) * 2] */

                /* load 16 pixel values of row 1 */
                src_temp11_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset)); /* pu1_src[col + (i-1) * 2] */

                /* Derive the source pixels for processing the 2nd pixel of row 0 */
                src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+2));

                src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b);
                src_temp7_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);

                /* Derive the source pixels for processing the 3rd pixel of row 0 */
                src_temp3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+4));

                /* Derive the source pixels for processing the 4th pixel of row 0 */
                src_temp4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+6));

                src_temp6_16x8b= _mm_unpacklo_epi16(src_temp3_16x8b, src_temp4_16x8b);
                src_temp8_16x8b= _mm_unpackhi_epi16(src_temp3_16x8b, src_temp4_16x8b);

                res_temp1_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff0_1_8x16b);
                res_temp2_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff2_3_8x16b);
                res_temp11_8x16b = _mm_madd_epi16(src_temp7_16x8b, coeff0_1_8x16b);
                res_temp12_8x16b = _mm_madd_epi16(src_temp8_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp1_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp3_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);

                res_temp1_8x16b = _mm_srai_epi32(res_temp1_8x16b, (FILTER_PREC - (14-bit_depth))); /* row = 0 */
                res_temp3_8x16b = _mm_srai_epi32(res_temp3_8x16b, (FILTER_PREC - (14-bit_depth))); /* row = 0 */
                res_temp3_8x16b = _mm_packs_epi32(res_temp1_8x16b, res_temp3_8x16b);       /* row = 0 */
                /* store 8 16-bit values */
                _mm_storeu_si128((__m128i *)(pi2_dst + offset), res_temp3_8x16b); /* pi2_dst[col] = i2_tmp_u */

                /* Derive the source pixels for processing the 2nd pixel of row 1 */
                src_temp12_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+2));

                src_temp15_16x8b= _mm_unpacklo_epi16(src_temp11_16x8b, src_temp12_16x8b);
                src_temp17_16x8b= _mm_unpackhi_epi16(src_temp11_16x8b, src_temp12_16x8b);

                /* Derive the source pixels for processing the 3rd pixel of row 1 */
                src_temp13_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+4));

                /* Derive the source pixels for processing the 4th pixel of row 1 */
                src_temp14_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + src_strd + offset+6));

                src_temp16_16x8b= _mm_unpacklo_epi16(src_temp13_16x8b, src_temp14_16x8b);
                src_temp18_16x8b= _mm_unpackhi_epi16(src_temp13_16x8b, src_temp14_16x8b);

                res_temp11_8x16b = _mm_madd_epi16(src_temp15_16x8b, coeff0_1_8x16b);
                res_temp12_8x16b = _mm_madd_epi16(src_temp16_16x8b, coeff2_3_8x16b);
                res_temp1_8x16b = _mm_madd_epi16(src_temp17_16x8b, coeff0_1_8x16b);
                res_temp2_8x16b = _mm_madd_epi16(src_temp18_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp11_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);
                res_temp13_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp11_8x16b = _mm_srai_epi32(res_temp11_8x16b, (FILTER_PREC - (14-bit_depth)));
                res_temp13_8x16b = _mm_srai_epi32(res_temp13_8x16b, (FILTER_PREC - (14-bit_depth)));
                res_temp13_8x16b = _mm_packs_epi32(res_temp11_8x16b, res_temp13_8x16b);

                /* store 8 16-bit values */
                _mm_storeu_si128((__m128i *)(pi2_dst + dst_strd + offset), res_temp13_8x16b); /* pi2_dst[col] = i2_tmp_u */

                offset += 8; /* To pointer update */

            } /* inner loop ends here(8- output values in single iteration) */

            pu2_src += 2 * src_strd; /* pointer update */
            pi2_dst += 2 * dst_strd; /* pointer update */
        }

        /* Epilogue to take care of odd ht */
        if(row)
        {
            offset = 0;
            for(col = 0; col < 2*wd; col += 8)
            {

                /* load 16 pixel values of row 0 */
                src_temp1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset)); /* pu1_src[col + (i-1) * 2] */

                /* Derive the source pixels for processing the 2nd pixel of row 0 */
                src_temp2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+2));

                src_temp5_16x8b= _mm_unpacklo_epi16(src_temp1_16x8b, src_temp2_16x8b);
                src_temp7_16x8b= _mm_unpackhi_epi16(src_temp1_16x8b, src_temp2_16x8b);

                /* Derive the source pixels for processing the 3rd pixel of row 0 */
                src_temp3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+4));

                /* Derive the source pixels for processing the 4th pixel of row 0 */
                src_temp4_16x8b = _mm_loadu_si128((__m128i *)(pu2_src -2 + offset+6));

                src_temp6_16x8b= _mm_unpacklo_epi16(src_temp3_16x8b, src_temp4_16x8b);
                src_temp8_16x8b= _mm_unpackhi_epi16(src_temp3_16x8b, src_temp4_16x8b);

                res_temp1_8x16b = _mm_madd_epi16(src_temp5_16x8b, coeff0_1_8x16b);
                res_temp2_8x16b = _mm_madd_epi16(src_temp6_16x8b, coeff2_3_8x16b);
                res_temp11_8x16b = _mm_madd_epi16(src_temp7_16x8b, coeff0_1_8x16b);
                res_temp12_8x16b = _mm_madd_epi16(src_temp8_16x8b, coeff2_3_8x16b);

                /* tmp += pi1_coeff[i] * pi2_src[col + (i-1) * 2] */
                res_temp1_8x16b = _mm_add_epi32(res_temp1_8x16b, res_temp2_8x16b);
                res_temp3_8x16b = _mm_add_epi32(res_temp11_8x16b, res_temp12_8x16b);
                res_temp1_8x16b = _mm_srai_epi32(res_temp1_8x16b, (FILTER_PREC - (14-bit_depth)));
                res_temp3_8x16b = _mm_srai_epi32(res_temp3_8x16b, (FILTER_PREC - (14-bit_depth)));
                res_temp3_8x16b = _mm_packs_epi32(res_temp1_8x16b, res_temp3_8x16b);
                /* store 8 16-bit values */
                _mm_storeu_si128((__m128i *)(pi2_dst + offset), res_temp3_8x16b); /* pi2_dst[col] = i2_tmp_u */

                offset += 8; /* To pointer update */
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     Interprediction chroma filter to store vertical 16bit ouput
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
*    the elements pointed by 'pu2_src' and  writes to the location pointed by
*    'pu2_dst'  No downshifting or clipping is done and the output is  used as
*    an input for weighted prediction
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_vert_w16out_sse42(UWORD16 *pu2_src,
                                                   WORD16 *pi2_dst,
                                                   WORD32 src_strd,
                                                   WORD32 dst_strd,
                                                   WORD8 *pi1_coeff,
                                                   WORD32 ht,
                                                   WORD32 wd,
                                                   UWORD8 bit_depth)
{
    WORD32 row, col;
    UWORD16 *pu2_src_copy;
    WORD16 *pi2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b;
    __m128i s4_8x16b, s5_8x16b, s6_8x16b, s8_8x16b;
    __m128i s11_8x16b, s12_8x16b, s15_8x16b, s16_8x16b;
    __m128i zero_8x16b;
    __m128i s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b;
    __m128i s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b;
    __m128i sign_reg;

    PREFETCH((char const*)(pu2_src+(0 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(1 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(2 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(3 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(4 * src_strd)), _MM_HINT_T0);
    PREFETCH((char const*)(pu2_src+(5 * src_strd)), _MM_HINT_T0);

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128();
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    /* outer for loop starts from here */
    if(wd % 8 == 0)
    {
        /* wd = multiple of 8 case */

        pu2_src_copy = pu2_src;
        pi2_dst_copy = pi2_dst;

        for(col = 0; col < 2 * wd; col += 8)
        {
            pu2_src = pu2_src_copy + col;
            pi2_dst = pi2_dst_copy + col;

            for(row = 0; row < ht; row += 2)
            {
                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                /* load 16 pixel values */
                s21_8x16b  = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

                /* load 16 pixel values */
                s22_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

                /* load 16 pixel values */
                s23_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

                /* load 16 pixel values */
                s24_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s21_8x16b, s22_8x16b);

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s23_8x16b, s24_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b);

                s8_8x16b = _mm_srai_epi32(s8_8x16b,  (FILTER_PREC - (14-bit_depth)));

                s35_8x16b = _mm_srai_epi32(s35_8x16b,  (FILTER_PREC - (14-bit_depth)));

                s8_8x16b = _mm_packs_epi32(s8_8x16b, s35_8x16b);

                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storeu_si128((__m128i *)(pi2_dst), s8_8x16b);

                s25_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s22_8x16b, s23_8x16b);

                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s24_8x16b, s25_8x16b);

                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s8_8x16b = _mm_srai_epi32(s8_8x16b,  (FILTER_PREC - (14-bit_depth)));

                s35_8x16b = _mm_srai_epi32(s35_8x16b,  (FILTER_PREC - (14-bit_depth)));

                s8_8x16b = _mm_packs_epi32(s8_8x16b, s35_8x16b);
                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storeu_si128((__m128i *)(pi2_dst + dst_strd), s8_8x16b);

                pu2_src += 2 * src_strd;
                pi2_dst += 2 * dst_strd;
            } /* inner for loop ends here(8-output values in single iteration) */
        }
    }
    else if(wd % 4 == 0)
    {
        /* wd = multiple of 8 case */
        for(row = 0; row < ht; row += 2)
        {
            pu2_src_copy = pu2_src;
            pi2_dst_copy = pi2_dst;

            for(col = 0; col < 2 * wd; col += 8)
            {

                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                /* load 8 pixel values */
                s21_8x16b  = _mm_loadu_si128((__m128i *)(pu2_src+(-1 * src_strd)));

                /* load 8 pixel values */
                s22_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(0 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);
                s31_8x16b = _mm_unpackhi_epi16(s21_8x16b, s22_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );
                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                /* load 8 pixel values */
                s23_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(1 * src_strd)));

                /* load 8 pixel values */
                s24_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(2 * src_strd)));

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);
                s33_8x16b = _mm_unpackhi_epi16(s23_8x16b, s24_8x16b);

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );
                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */
                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b);

                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (FILTER_PREC - (14-bit_depth)));
                s32_8x16b = _mm_srai_epi32(s35_8x16b,  (FILTER_PREC - (14-bit_depth)));
                s8_8x16b = _mm_packs_epi32(s6_8x16b, s32_8x16b);

                _mm_storeu_si128((__m128i *)(pi2_dst), s8_8x16b);

                s25_8x16b = _mm_loadu_si128((__m128i *)(pu2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);
                s31_8x16b = _mm_unpackhi_epi16(s22_8x16b, s23_8x16b);
                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );
                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);
                s33_8x16b = _mm_unpackhi_epi16(s24_8x16b, s25_8x16b);

                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );
                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */
                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (FILTER_PREC - (14-bit_depth)));
                s33_8x16b = _mm_srai_epi32(s35_8x16b,  (FILTER_PREC - (14-bit_depth)));

                s8_8x16b = _mm_packs_epi32(s6_8x16b, s33_8x16b);

                _mm_storeu_si128((__m128i *)(pi2_dst + dst_strd), s8_8x16b);

                pu2_src += 8;    /* To pointer update */
                pi2_dst += 8;
            } /* inner for loop ends here(8-output values in single iteration) */

            pu2_src = pu2_src_copy + 2 * src_strd; /* pointer update */
            pi2_dst = pi2_dst_copy + 2 * dst_strd; /* pointer update */
        }
    }
    else
    {
        /* wd = multiple of 4 case */
        for(row = 0; row < ht; row += 2)
        {
            pu2_src_copy = pu2_src;
            pi2_dst_copy = pi2_dst;
            for(col = 0; col < 2 * wd; col += 4)
            {
                PREFETCH((char const*)(pu2_src+(6 * src_strd)), _MM_HINT_T0);
                PREFETCH((char const*)(pu2_src+(7 * src_strd)), _MM_HINT_T0);

                /* load 8 pixel values */
                s21_8x16b  = _mm_loadl_epi64((__m128i *)(pu2_src+(-1 * src_strd)));

                /* load 8 pixel values */
                s22_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(0 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                /* load 8 pixel values */
                s23_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(1 * src_strd)));

                /* load 8 pixel values */
                s24_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(2 * src_strd)));

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (FILTER_PREC - (14-bit_depth)));

                s6_8x16b = _mm_packs_epi32(s6_8x16b, zero_8x16b);
                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storel_epi64((__m128i *)(pi2_dst), s6_8x16b);

                s25_8x16b = _mm_loadl_epi64((__m128i *)(pu2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);
                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);
                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (FILTER_PREC - (14-bit_depth)));

                s6_8x16b = _mm_packs_epi32(s6_8x16b, zero_8x16b);
                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storel_epi64((__m128i *)(pi2_dst + dst_strd), s6_8x16b);

                pu2_src += 4;   /* To pointer update */
                pi2_dst += 4;
            } /* inner for loop ends here(8-output values in single iteration) */

            pu2_src = pu2_src_copy + 2 * src_strd; /* pointer update */
            pi2_dst = pi2_dst_copy + 2 * dst_strd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*     chroma interprediction filter for vertical 16bit input
*
* @par Description:
*    Applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
*    the elements pointed by 'pu2_src' and  writes to the location pointed by
*    'pu2_dst'  Input is 16 bits  The filter output is downshifted by 12 and
*    clipped to lie  between 0 and 255
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pu2_dst
*  UWORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_vert_w16inp_sse42(WORD16 *pi2_src,
                                                   UWORD16 *pu2_dst,
                                                   WORD32 src_strd,
                                                   WORD32 dst_strd,
                                                   WORD8 *pi1_coeff,
                                                   WORD32 ht,
                                                   WORD32 wd,
                                                   UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD16 *pi2_src_copy;
    UWORD16 *pu2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b;
    __m128i s4_8x16b, s5_8x16b, s6_8x16b, s7_8x16b, s8_8x16b, s9_8x16b;
    __m128i s11_8x16b, s12_8x16b, s15_8x16b, s16_8x16b;
    __m128i zero_8x16b, offset_8x16b, mask_low_32b, mask_high_96b, sign_reg;
    __m128i s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b;
    __m128i s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b;
    __m128i clip_tmp_8x16b;

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128();
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    /* seting  values in register */
    offset_8x16b = _mm_set1_epi32((1<<(FILTER_PREC-1))); /* for offset addition */
    mask_low_32b = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000);
    mask_high_96b = _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF);

    /* outer for loop starts from here */
    if(wd % 4 == 0)
    {
        /* wd = multiple of 8 case */
        pi2_src_copy = pi2_src;
        pu2_dst_copy = pu2_dst;

        for(col = 0; col < 2 * wd; col += 8)
        {
            pi2_src = pi2_src_copy + col;
            pu2_dst = pu2_dst_copy + col;

            for(row = 0; row < ht; row += 2)
            {
                /* load 16 pixel values */
                s21_8x16b  = _mm_loadu_si128((__m128i *)(pi2_src+(-1 * src_strd)));

                /* load 16 pixel values */
                s22_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(0 * src_strd)));

                /* load 16 pixel values */
                s23_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(1 * src_strd)));

                /* load 16 pixel values */
                s24_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(2 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s21_8x16b, s22_8x16b);

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s23_8x16b, s24_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (14-bit_depth));

                s32_8x16b = _mm_srai_epi32(s35_8x16b,  (14-bit_depth));

                /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
                s7_8x16b = _mm_add_epi32(s6_8x16b, offset_8x16b);

                /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s8_8x16b = _mm_srai_epi32(s7_8x16b,  FILTER_PREC);

                /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
                s33_8x16b = _mm_add_epi32(s32_8x16b, offset_8x16b);

                /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s34_8x16b = _mm_srai_epi32(s33_8x16b,  FILTER_PREC);

                s35_8x16b = _mm_packus_epi32(s8_8x16b, s34_8x16b);

                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(s35_8x16b, s35_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                s7_8x16b = _mm_min_epi16(s35_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                s7_8x16b = _mm_max_epi16(s7_8x16b, clip_tmp_8x16b);
                /* store 8 8-bit output values */
                /* pu1_dst[col] = (UWORD8)i2_tmp; */
                _mm_storeu_si128((__m128i *)(pu2_dst), s7_8x16b);

                s25_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s22_8x16b, s23_8x16b);

                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s24_8x16b, s25_8x16b);

                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (14-bit_depth));

                s32_8x16b = _mm_srai_epi32(s35_8x16b,  (14-bit_depth));

                /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
                s7_8x16b = _mm_add_epi32(s6_8x16b, offset_8x16b);

                /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s8_8x16b = _mm_srai_epi32(s7_8x16b,  FILTER_PREC);

                /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
                s33_8x16b = _mm_add_epi32(s32_8x16b, offset_8x16b);

                /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s34_8x16b = _mm_srai_epi32(s33_8x16b,  FILTER_PREC);

                s35_8x16b = _mm_packus_epi32(s8_8x16b, s34_8x16b);

                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(s35_8x16b, s35_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                s7_8x16b = _mm_min_epi16(s35_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                s7_8x16b = _mm_max_epi16(s7_8x16b, clip_tmp_8x16b);

                /* store 8 8-bit output values */
                /* pu1_dst[col] = (UWORD8)i2_tmp; */
                _mm_storeu_si128((__m128i *)(pu2_dst + dst_strd), s7_8x16b);

                pi2_src += 2 * src_strd;
                pu2_dst += 2 * dst_strd;
            } /* inner for loop ends here(8-output values in single iteration) */
        }
    }
    else
    {
        /* wd = multiple of 4 case */
        for(row = 0; row < ht; row += 2)
        {
            pi2_src_copy = pi2_src;
            pu2_dst_copy = pu2_dst;
            for(col = 0; col < 2 * wd; col += 4)
            {
                /* load 8 pixel values */
                s21_8x16b  = _mm_loadl_epi64((__m128i *)(pi2_src+(-1 * src_strd)));

                /* load 8 pixel values */
                s22_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(0 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                /* load 8 pixel values */
                s23_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(1 * src_strd)));

                /* load 8 pixel values */
                s24_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(2 * src_strd)));

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (14-bit_depth));

                /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
                s7_8x16b = _mm_add_epi32(s6_8x16b, offset_8x16b);

                /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s8_8x16b = _mm_srai_epi32(s7_8x16b,  FILTER_PREC);

                s9_8x16b = _mm_packus_epi32(s8_8x16b, zero_8x16b);

                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(s9_8x16b, s9_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                s9_8x16b = _mm_min_epi16(s9_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                s9_8x16b = _mm_max_epi16(s9_8x16b, clip_tmp_8x16b);

                /* store 8 8-bit output values */
                /* pu1_dst[col] = (UWORD8)i2_tmp; */
                _mm_storel_epi64((__m128i *)(pu2_dst), s9_8x16b);

                s25_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);
                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);
                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  (14-bit_depth));

                /* (tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) */
                s7_8x16b = _mm_add_epi32(s6_8x16b, offset_8x16b);

                /* tmp = ((tmp >> SHIFT_14_MINUS_BIT_DEPTH) + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s8_8x16b = _mm_srai_epi32(s7_8x16b,  FILTER_PREC);

                s9_8x16b = _mm_packus_epi32(s8_8x16b, zero_8x16b);

                // Clipping
                clip_tmp_8x16b = _mm_cmpeq_epi16(s9_8x16b, s9_8x16b);
                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, 16-bit_depth);
                s9_8x16b = _mm_min_epi16(s9_8x16b, clip_tmp_8x16b);

                clip_tmp_8x16b = _mm_srli_epi16(clip_tmp_8x16b, bit_depth);
                s9_8x16b = _mm_max_epi16(s9_8x16b, clip_tmp_8x16b);

                /* store 8 8-bit output values */
                /* pu1_dst[col] = (UWORD8)i2_tmp; */
                _mm_storel_epi64((__m128i *)(pu2_dst + dst_strd), s9_8x16b);

                pi2_src += 4;   /* To pointer update */
                pu2_dst += 4;
            } /* inner for loop ends here(8-output values in single iteration) */

            pi2_src = pi2_src_copy + 2 * src_strd; /* pointer update */
            pu2_dst = pu2_dst_copy + 2 * dst_strd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*
*      Chroma interprediction filter for 16bit vertical input and output.
*
* @par Description:
*       Applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
*       the elements pointed by 'pu1_src' and  writes to the location pointed by
*       'pu1_dst'  Input is 16 bits  The filter output is downshifted by 6 and
*       8192 is  subtracted to store it as a 16 bit number  The output is used as
*       a input to weighted prediction
*
* @param[in] pi2_src
*  WORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_vert_w16inp_w16out_sse42(WORD16 *pi2_src,
                                                          WORD16 *pi2_dst,
                                                          WORD32 src_strd,
                                                          WORD32 dst_strd,
                                                          WORD8 *pi1_coeff,
                                                          WORD32 ht,
                                                          WORD32 wd,
                                                          UWORD8 bit_depth)
{
    WORD32 row, col;
    WORD16 *pi2_src_copy;
    WORD16 *pi2_dst_copy;
    __m128i coeff0_1_8x16b, coeff2_3_8x16b;
    __m128i s4_8x16b, s5_8x16b, s6_8x16b, s7_8x16b, s8_8x16b, s9_8x16b;
    __m128i s11_8x16b, s12_8x16b, s15_8x16b, s16_8x16b;
    __m128i zero_8x16b, offset_8x16b, mask_low_32b, mask_high_96b, sign_reg;
    __m128i s21_8x16b, s22_8x16b, s23_8x16b, s24_8x16b, s25_8x16b;
    __m128i s31_8x16b, s32_8x16b, s33_8x16b, s34_8x16b, s35_8x16b;

    /* load 8 8-bit coefficients and convert 8-bit into 16-bit */
    s4_8x16b = _mm_loadl_epi64((__m128i *)pi1_coeff);

    zero_8x16b = _mm_setzero_si128();
    sign_reg =  _mm_cmpgt_epi8(zero_8x16b, s4_8x16b);
    s5_8x16b  = _mm_unpacklo_epi8(s4_8x16b, sign_reg);

    coeff0_1_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(0, 0, 0, 0));  /* pi1_coeff[4] */
    coeff2_3_8x16b = _mm_shuffle_epi32(s5_8x16b, _MM_SHUFFLE(1, 1, 1, 1));  /* pi1_coeff[4] */

    /* seting  values in register */
    offset_8x16b = _mm_set1_epi32(OFFSET_14_MINUS_BIT_DEPTH); /* for offset addition */
    mask_low_32b = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000);
    mask_high_96b = _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF);

    /* outer for loop starts from here */
    if(wd % 4 == 0)
    {
        /* wd = multiple of 8 case */

        pi2_src_copy = pi2_src;
        pi2_dst_copy = pi2_dst;

        for(col = 0; col < 2 * wd; col += 8)
        {
            pi2_src = pi2_src_copy + col;
            pi2_dst = pi2_dst_copy + col;

            for(row = 0; row < ht; row += 2)
            {
                /* load 16 pixel values */
                s21_8x16b  = _mm_loadu_si128((__m128i *)(pi2_src+(-1 * src_strd)));

                /* load 16 pixel values */
                s22_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(0 * src_strd)));

                /* load 16 pixel values */
                s23_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(1 * src_strd)));

                /* load 16 pixel values */
                s24_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(2 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s21_8x16b, s22_8x16b);

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s23_8x16b, s24_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b);

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  FILTER_PREC);

                s32_8x16b = _mm_srai_epi32(s35_8x16b,  FILTER_PREC);

                s9_8x16b = _mm_packs_epi32(s6_8x16b, zero_8x16b);

                s35_8x16b = _mm_packs_epi32(s32_8x16b, zero_8x16b);

                s7_8x16b = _mm_unpacklo_epi64(s9_8x16b, s35_8x16b);
                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storeu_si128((__m128i *)(pi2_dst), s7_8x16b);

                s25_8x16b = _mm_loadu_si128((__m128i *)(pi2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);

                s31_8x16b = _mm_unpackhi_epi16(s22_8x16b, s23_8x16b);

                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s32_8x16b = _mm_madd_epi16(s31_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);

                s33_8x16b = _mm_unpackhi_epi16(s24_8x16b, s25_8x16b);

                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s34_8x16b = _mm_madd_epi16(s33_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                s35_8x16b = _mm_add_epi32(s32_8x16b, s34_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  FILTER_PREC);

                s32_8x16b = _mm_srai_epi32(s35_8x16b,  FILTER_PREC);

                s9_8x16b = _mm_packs_epi32(s6_8x16b, zero_8x16b);

                s35_8x16b = _mm_packs_epi32(s32_8x16b, zero_8x16b);

                s7_8x16b = _mm_unpacklo_epi64(s9_8x16b, s35_8x16b);
                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storeu_si128((__m128i *)(pi2_dst + dst_strd), s7_8x16b);

                pi2_src += 2 * src_strd;
                pi2_dst += 2 * dst_strd;
            } /* inner for loop ends here(8-output values in single iteration) */
        }
    }
    else
    {
        /* wd = multiple of 4 case */
        for(row = 0; row < ht; row += 2)
        {
            pi2_src_copy = pi2_src;
            pi2_dst_copy = pi2_dst;
            for(col = 0; col < 2 * wd; col += 4)
            {
                /* load 4 pixel values */
                s21_8x16b  = _mm_loadl_epi64((__m128i *)(pi2_src+(-1 * src_strd)));

                /* load 4 pixel values */
                s22_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(0 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s21_8x16b, s22_8x16b);

                s11_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                /* load 4 pixel values */
                s23_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(1 * src_strd)));

                /* load 4 pixel values */
                s24_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(2 * src_strd)));

                s6_8x16b = _mm_unpacklo_epi16(s23_8x16b, s24_8x16b);

                s12_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s11_8x16b, s12_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  FILTER_PREC);

                s9_8x16b = _mm_packs_epi32(s6_8x16b, zero_8x16b);

                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storel_epi64((__m128i *)(pi2_dst), s9_8x16b);

                s25_8x16b = _mm_loadl_epi64((__m128i *)(pi2_src+(3 * src_strd)));

                s5_8x16b = _mm_unpacklo_epi16(s22_8x16b, s23_8x16b);
                s15_8x16b = _mm_madd_epi16(s5_8x16b, coeff0_1_8x16b );

                s6_8x16b = _mm_unpacklo_epi16(s24_8x16b, s25_8x16b);
                s16_8x16b = _mm_madd_epi16(s6_8x16b, coeff2_3_8x16b );

                s8_8x16b = _mm_add_epi32(s15_8x16b, s16_8x16b); /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) */

                /* (i2_tmp + OFFSET_14_MINUS_BIT_DEPTH) >> SHIFT_14_MINUS_BIT_DEPTH */
                s6_8x16b = _mm_srai_epi32(s8_8x16b,  FILTER_PREC);

                s9_8x16b = _mm_packs_epi32(s6_8x16b, zero_8x16b);

                /* store 8 8-bit output values */
                /* pi2_dst[col] = (UWORD8)i2_tmp; */
                _mm_storel_epi64((__m128i *)(pi2_dst + dst_strd), s9_8x16b);

                pi2_src += 4;   /* To pointer update */
                pi2_dst += 4;
            } /* inner for loop ends here(8-output values in single iteration) */

            pi2_src = pi2_src_copy + 2 * src_strd; /* pointer update */
            pi2_dst = pi2_dst_copy + 2 * dst_strd; /* pointer update */
        }
    }
}

/**
*******************************************************************************
*
* @brief
*      Chroma interprediction filter for copy
*
* @par Description:
*    Copies the array of width 'wd' and height 'ht' from the  location pointed
*    by 'src' to the location pointed by 'dst'
*
* @param[in] pu1_src
*  UWORD8 pointer to the source
*
* @param[out] pu1_dst
*  UWORD8 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_copy_sse42(UWORD16 *pu2_src,
                                            UWORD16 *pu2_dst,
                                            WORD32 src_strd,
                                            WORD32 dst_strd,
                                            WORD8 *pi1_coeff,
                                            WORD32 ht,
                                            WORD32 wd,
                                            UWORD8 bit_depth)
{
    WORD32 row, col, wdx2;
    __m128i  src0_16x8b, src1_16x8b, src2_16x8b, src3_16x8b;

    ASSERT(wd%2 == 0); /* checking assumption */
    ASSERT(ht%2 == 0); /* checking assumption */

    wdx2 = wd*2;

    if(0 == (ht & 3)) /* ht multiple of 4 */
    {
        if(0 == (wdx2 & 7)) /* wdx2 multiple of 16 case */
        {
            for(row = 0; row < ht; row += 4)
            {
                for(col = 0; col < wdx2; col += 8)
                {
                    /* load 16 pixel values from 15:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                    src2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                    src3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                    /* storing 16 8-bit output values */
                    _mm_storeu_si128((__m128i *)(pu2_dst), src0_16x8b);               /* row =0 */
                    _mm_storeu_si128((__m128i *)(pu2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                    _mm_storeu_si128((__m128i *)(pu2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                    _mm_storeu_si128((__m128i *)(pu2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                    pu2_src += 8; /* pointer update */
                    pu2_dst += 8; /* pointer update */
                } /* inner for loop ends here(16-output values in single iteration) */

                pu2_src += 4*src_strd - wdx2; /* pointer update */
                pu2_dst += 4*dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 = multiple of 4 case */
        {
            for(row = 0; row < ht; row += 4)
            {
                for(col = 0; col < wdx2; col += 4)
                {
                    /* load 16 pixel values from 15:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                    src2_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                    src3_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                    _mm_storel_epi64((__m128i *)(pu2_dst), src0_16x8b);               /* row =0 */
                    _mm_storel_epi64((__m128i *)(pu2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                    _mm_storel_epi64((__m128i *)(pu2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                    _mm_storel_epi64((__m128i *)(pu2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                    pu2_src += 4; /* pointer update */
                    pu2_dst += 4; /* pointer update */
                } /* inner for loop ends here(4- output values in single iteration) */

                pu2_src += 4*src_strd - wdx2; /* pointer update */
                pu2_dst += 4*dst_strd - wdx2; /* pointer update */
            }
        }
    }
    else /* ht multiple of 2 */
    {
        if(0 == (wdx2 & 7)) /* wdx2 multiple of 16 case */
        {
            for(row = 0; row < ht; row += 2)
            {
                for(col = 0; col < wdx2; col += 8)
                {
                    /* load 16 pixel values from 15:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+1*src_strd)); /* row =1 */

                    /* storing 16 8-bit output values */
                    _mm_storeu_si128((__m128i *)(pu2_dst), src0_16x8b);               /* row =0 */
                    _mm_storeu_si128((__m128i *)(pu2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */

                    pu2_src += 8; /* pointer update */
                    pu2_dst += 8; /* pointer update */
                } /* inner for loop ends here(16-output values in single iteration) */

                pu2_src += 2*src_strd - wdx2; /* pointer update */
                pu2_dst += 2*dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 = multiple of 4 case */
        {
            for(row = 0; row < ht; row += 2)
            {
                for(col = 0; col < wdx2; col += 4)
                {
                    /* load 16 pixel values from 15:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+1*src_strd)); /* row =1 */

                    _mm_storel_epi64((__m128i *)(pu2_dst), src0_16x8b);               /* row =0 */
                    _mm_storel_epi64((__m128i *)(pu2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */

                    pu2_src += 4; /* pointer update */
                    pu2_dst += 4; /* pointer update */
                } /* inner for loop ends here(4- output values in single iteration) */

                pu2_src += 2*src_strd - wdx2; /* pointer update */
                pu2_dst += 2*dst_strd - wdx2; /* pointer update */
            }
        }
    }
}

/**
*******************************************************************************
*
* @brief
*       chroma interprediction filter for copying 16bit output
*
* @par Description:
*    Copies the array of width 'wd' and height 'ht' from the  location pointed
*    by 'src' to the location pointed by 'dst' The output is upshifted by 6
*    bits and is used as input for vertical filtering or weighted prediction
*
* @param[in] pu2_src
*  UWORD16 pointer to the source
*
* @param[out] pi2_dst
*  WORD16 pointer to the destination
*
* @param[in] src_strd
*  integer source stride
*
* @param[in] dst_strd
*  integer destination stride
*
* @param[in] pi1_coeff
*  WORD8 pointer to the filter coefficients
*
* @param[in] ht
*  integer height of the array
*
* @param[in] wd
*  integer width of the array
*
* @returns
*
* @remarks
*  None
*
*******************************************************************************
*/

void ihevc_hbd_inter_pred_chroma_copy_w16out_sse42(UWORD16 *pu2_src,
                                                   WORD16 *pi2_dst,
                                                   WORD32 src_strd,
                                                   WORD32 dst_strd,
                                                   WORD8 *pi1_coeff,
                                                   WORD32 ht,
                                                   WORD32 wd,
                                                   UWORD8 bit_depth)
{
    WORD32 row, col, wdx2;
    __m128i  src0_16x8b, src1_16x8b, src2_16x8b, src3_16x8b;

    ASSERT(wd%2 == 0); /* checking assumption */
    ASSERT(ht%2 == 0); /* checking assumption */

    wdx2 = wd*2;

    if(0 == (ht & 3)) /* multiple of 4 case */
    {
        if(0 == (wdx2 & 7)) /* multiple of 8 case */
        {
            for(row = 0; row < ht; row += 4)
            {
                for(col = 0; col < wdx2; col += 8)
                {
                    /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                    src2_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                    src3_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                    src0_16x8b = _mm_slli_epi16(src0_16x8b,  (14 - bit_depth)); /* (pu1_src[col] << SHIFT_14_MINUS_BIT_DEPTH */
                    src1_16x8b = _mm_slli_epi16(src1_16x8b,  (14 - bit_depth));
                    src2_16x8b = _mm_slli_epi16(src2_16x8b,  (14 - bit_depth));
                    src3_16x8b = _mm_slli_epi16(src3_16x8b,  (14 - bit_depth));

                    /* storing 16 8-bit output values */
                    _mm_storeu_si128((__m128i *)(pi2_dst), src0_16x8b);               /* row =0 */
                    _mm_storeu_si128((__m128i *)(pi2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                    _mm_storeu_si128((__m128i *)(pi2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                    _mm_storeu_si128((__m128i *)(pi2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                    pu2_src += 8; /* pointer update */
                    pi2_dst += 8; /* pointer update */
                } /* inner for loop ends here(8-output values in single iteration) */

                pu2_src += 4*src_strd - wdx2; /* pointer update */
                pi2_dst += 4*dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 = multiple of 4 case */
        {
            for(row = 0; row < ht; row += 4)
            {
                for(col = 0; col < wdx2; col += 4)
                {
                    /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+1*src_strd)); /* row =1 */
                    src2_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+2*src_strd)); /* row =2 */
                    src3_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+3*src_strd)); /* row =3 */

                    src0_16x8b = _mm_slli_epi16(src0_16x8b,  (14 - bit_depth)); /* (pu1_src[col] << SHIFT_14_MINUS_BIT_DEPTH */
                    src1_16x8b = _mm_slli_epi16(src1_16x8b,  (14 - bit_depth));
                    src2_16x8b = _mm_slli_epi16(src2_16x8b,  (14 - bit_depth));
                    src3_16x8b = _mm_slli_epi16(src3_16x8b,  (14 - bit_depth));

                    /* storing 16 8-bit output values */
                    _mm_storel_epi64((__m128i *)(pi2_dst), src0_16x8b);               /* row =0 */
                    _mm_storel_epi64((__m128i *)(pi2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */
                    _mm_storel_epi64((__m128i *)(pi2_dst + 2*dst_strd), src2_16x8b);  /* row =2 */
                    _mm_storel_epi64((__m128i *)(pi2_dst + 3*dst_strd), src3_16x8b);  /* row =3 */

                    pu2_src += 4; /* pointer update */
                    pi2_dst += 4; /* pointer update */
                } /* inner for loop ends here(4-output values in single iteration) */

                pu2_src += 4*src_strd - wdx2; /* pointer update */
                pi2_dst += 4*dst_strd - wdx2; /* pointer update */
            }
        }
    }
    else  /* ht multiple of 2 case */
    {
        if(0 == (wdx2 & 7)) /* multiple of 8 case */
        {
            for(row = 0; row < ht; row += 2)
            {
                for(col = 0; col < wdx2; col += 8)
                {
                    /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadu_si128((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadu_si128((__m128i *)(pu2_src+1*src_strd)); /* row =1 */

                    src0_16x8b = _mm_slli_epi16(src0_16x8b,  (14 - bit_depth)); /* (pu1_src[col] << SHIFT_14_MINUS_BIT_DEPTH */
                    src1_16x8b = _mm_slli_epi16(src1_16x8b,  (14 - bit_depth));

                    /* storing 16 8-bit output values */
                    _mm_storeu_si128((__m128i *)(pi2_dst), src0_16x8b);               /* row =0 */
                    _mm_storeu_si128((__m128i *)(pi2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */

                    pu2_src += 8; /* pointer update */
                    pi2_dst += 8; /* pointer update */
                } /* inner for loop ends here(8-output values in single iteration) */

                pu2_src += 2*src_strd - wdx2; /* pointer update */
                pi2_dst += 2*dst_strd - wdx2; /* pointer update */
            }
        }
        else /* wdx2 = multiple of 4 case */
        {
            for(row = 0; row < ht; row += 2)
            {
                for(col = 0; col < wdx2; col += 4)
                {
                    /* load 8 pixel values from 7:0 pos. relative to cur. pos. */
                    src0_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src));            /* row =0 */
                    src1_16x8b = _mm_loadl_epi64((__m128i *)(pu2_src+1*src_strd)); /* row =1 */

                    src0_16x8b = _mm_slli_epi16(src0_16x8b,  (14 - bit_depth)); /* (pu1_src[col] << SHIFT_14_MINUS_BIT_DEPTH */
                    src1_16x8b = _mm_slli_epi16(src1_16x8b,  (14 - bit_depth));

                    /* storing 16 8-bit output values */
                    _mm_storel_epi64((__m128i *)(pi2_dst), src0_16x8b);               /* row =0 */
                    _mm_storel_epi64((__m128i *)(pi2_dst + 1*dst_strd), src1_16x8b);  /* row =1 */

                    pu2_src += 4; /* pointer update */
                    pi2_dst += 4; /* pointer update */
                } /* inner for loop ends here(4-output values in single iteration) */

                pu2_src += 2*src_strd - wdx2; /* pointer update */
                pi2_dst += 2*dst_strd - wdx2; /* pointer update */
            }
        }
    }
}
