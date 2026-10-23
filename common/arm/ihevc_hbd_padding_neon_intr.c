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
 *  ihevc_hbd_padding_neon_intr.c
 *
 * @brief
 *  Contains ARM NEON intrinsic function definitions for High Bit Depth (HBD)
 *  padding operations in HEVC.
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_hbd_pad_left_luma_neonintr()
 *  - ihevc_hbd_pad_right_luma_neonintr()
 *  - ihevc_hbd_pad_top_neonintr()
 *  - ihevc_hbd_pad_bottom_neonintr()
 *  - ihevc_hbd_pad_left_chroma_neonintr()
 *  - ihevc_hbd_pad_right_chroma_neonintr()
 *
 * @remarks
 *  Optimized with ARM NEON intrinsics (ARMv8 / AArch64 / AArch32 NEON).
 *  Exact bit-exact replacement for C implementations in ihevc_hbd_padding.c.
 *
 *******************************************************************************
 */

#include <arm_neon.h>
#include <assert.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_padding.h"
#include "ihevc_debug.h"

 /**
 *******************************************************************************
 *
 * @brief
 *   Padding (luma block) at the left of a 2D array using ARM NEON intrinsics.
 *
 * @par Description:
 *   The left column of a 2D array (val = *pu2_src) is replicated pad_size times
 *   to the left for ht rows.
 *
 * @param[in,out] pu2_src
 *   UWORD16 pointer to the source (col 0)
 *
 * @param[in] src_strd
 *   Source buffer stride in UWORD16 units
 *
 * @param[in] ht
 *   Height of the array to pad
 *
 * @param[in] pad_size
 *   Number of UWORD16 samples to pad to the left
 *
 * @returns
 *   Void
 *
 *******************************************************************************
 */
void ihevc_hbd_pad_left_luma_neonintr(UWORD16* pu2_src,
    WORD32 src_strd,
    WORD32 ht,
    WORD32 pad_size)
{
    WORD32 row;

    ASSERT(pad_size == 80);
    ASSERT((ht % 4) == 0);
    UNUSED(pad_size);

    /* Process 4 rows per iteration */
    for (row = 0; row < ht; row += 4)
    {
        uint16x8_t v0 = vdupq_n_u16(pu2_src[0]);
        uint16x8_t v1 = vdupq_n_u16(pu2_src[src_strd]);
        uint16x8_t v2 = vdupq_n_u16(pu2_src[2 * src_strd]);
        uint16x8_t v3 = vdupq_n_u16(pu2_src[3 * src_strd]);

        UWORD16* dst0 = pu2_src - 80;
        UWORD16* dst1 = dst0 + src_strd;
        UWORD16* dst2 = dst1 + src_strd;
        UWORD16* dst3 = dst2 + src_strd;

        /* Row 0 */
        vst1q_u16(dst0 + 0, v0);
        vst1q_u16(dst0 + 8, v0);
        vst1q_u16(dst0 + 16, v0);
        vst1q_u16(dst0 + 24, v0);
        vst1q_u16(dst0 + 32, v0);
        vst1q_u16(dst0 + 40, v0);
        vst1q_u16(dst0 + 48, v0);
        vst1q_u16(dst0 + 56, v0);
        vst1q_u16(dst0 + 64, v0);
        vst1q_u16(dst0 + 72, v0);

        /* Row 1 */
        vst1q_u16(dst1 + 0, v1);
        vst1q_u16(dst1 + 8, v1);
        vst1q_u16(dst1 + 16, v1);
        vst1q_u16(dst1 + 24, v1);
        vst1q_u16(dst1 + 32, v1);
        vst1q_u16(dst1 + 40, v1);
        vst1q_u16(dst1 + 48, v1);
        vst1q_u16(dst1 + 56, v1);
        vst1q_u16(dst1 + 64, v1);
        vst1q_u16(dst1 + 72, v1);

        /* Row 2 */
        vst1q_u16(dst2 + 0, v2);
        vst1q_u16(dst2 + 8, v2);
        vst1q_u16(dst2 + 16, v2);
        vst1q_u16(dst2 + 24, v2);
        vst1q_u16(dst2 + 32, v2);
        vst1q_u16(dst2 + 40, v2);
        vst1q_u16(dst2 + 48, v2);
        vst1q_u16(dst2 + 56, v2);
        vst1q_u16(dst2 + 64, v2);
        vst1q_u16(dst2 + 72, v2);

        /* Row 3 */
        vst1q_u16(dst3 + 0, v3);
        vst1q_u16(dst3 + 8, v3);
        vst1q_u16(dst3 + 16, v3);
        vst1q_u16(dst3 + 24, v3);
        vst1q_u16(dst3 + 32, v3);
        vst1q_u16(dst3 + 40, v3);
        vst1q_u16(dst3 + 48, v3);
        vst1q_u16(dst3 + 56, v3);
        vst1q_u16(dst3 + 64, v3);
        vst1q_u16(dst3 + 72, v3);

        pu2_src += 4 * src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding (luma block) at the right of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The rightmost column of a 2D array (val = *(pu2_src - 1)) is replicated
*   pad_size times to the right for ht rows.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the start of the right padding area (col wd)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad to the right
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_right_luma_neonintr(UWORD16* pu2_src,
    WORD32 src_strd,
    WORD32 ht,
    WORD32 pad_size)
{
    WORD32 row;

    ASSERT(pad_size == 80);
    ASSERT((ht % 4) == 0);
    UNUSED(pad_size);

    /* Process 4 rows per iteration */
    for (row = 0; row < ht; row += 4)
    {
        uint16x8_t v0 = vdupq_n_u16(*(pu2_src - 1));
        uint16x8_t v1 = vdupq_n_u16(*(pu2_src + src_strd - 1));
        uint16x8_t v2 = vdupq_n_u16(*(pu2_src + 2 * src_strd - 1));
        uint16x8_t v3 = vdupq_n_u16(*(pu2_src + 3 * src_strd - 1));

        UWORD16* dst0 = pu2_src;
        UWORD16* dst1 = dst0 + src_strd;
        UWORD16* dst2 = dst1 + src_strd;
        UWORD16* dst3 = dst2 + src_strd;

        /* Row 0 */
        vst1q_u16(dst0 + 0, v0);
        vst1q_u16(dst0 + 8, v0);
        vst1q_u16(dst0 + 16, v0);
        vst1q_u16(dst0 + 24, v0);
        vst1q_u16(dst0 + 32, v0);
        vst1q_u16(dst0 + 40, v0);
        vst1q_u16(dst0 + 48, v0);
        vst1q_u16(dst0 + 56, v0);
        vst1q_u16(dst0 + 64, v0);
        vst1q_u16(dst0 + 72, v0);

        /* Row 1 */
        vst1q_u16(dst1 + 0, v1);
        vst1q_u16(dst1 + 8, v1);
        vst1q_u16(dst1 + 16, v1);
        vst1q_u16(dst1 + 24, v1);
        vst1q_u16(dst1 + 32, v1);
        vst1q_u16(dst1 + 40, v1);
        vst1q_u16(dst1 + 48, v1);
        vst1q_u16(dst1 + 56, v1);
        vst1q_u16(dst1 + 64, v1);
        vst1q_u16(dst1 + 72, v1);

        /* Row 2 */
        vst1q_u16(dst2 + 0, v2);
        vst1q_u16(dst2 + 8, v2);
        vst1q_u16(dst2 + 16, v2);
        vst1q_u16(dst2 + 24, v2);
        vst1q_u16(dst2 + 32, v2);
        vst1q_u16(dst2 + 40, v2);
        vst1q_u16(dst2 + 48, v2);
        vst1q_u16(dst2 + 56, v2);
        vst1q_u16(dst2 + 64, v2);
        vst1q_u16(dst2 + 72, v2);

        /* Row 3 */
        vst1q_u16(dst3 + 0, v3);
        vst1q_u16(dst3 + 8, v3);
        vst1q_u16(dst3 + 16, v3);
        vst1q_u16(dst3 + 24, v3);
        vst1q_u16(dst3 + 32, v3);
        vst1q_u16(dst3 + 40, v3);
        vst1q_u16(dst3 + 48, v3);
        vst1q_u16(dst3 + 56, v3);
        vst1q_u16(dst3 + 64, v3);
        vst1q_u16(dst3 + 72, v3);

        pu2_src += 4 * src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding at the top of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The top row (row 0) is replicated pad_size times upwards.
*   For common HEVC widths (64, 32, 16, 8, 4), the source row is preloaded
*   ONCE into NEON registers prior to the loop, completely eliminating all
*   memory loads and cache-miss overhead inside the vertical replication loop.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the top valid row (row 0)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] wd
*   Width of the row to replicate in UWORD16 units
*
* @param[in] pad_size
*   Number of rows to pad above
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_top_neonintr(UWORD16* pu2_src,
    WORD32 src_strd,
    WORD32 wd,
    WORD32 pad_size)
{
    WORD32 row;
    UWORD16* dst = pu2_src - pad_size * src_strd;

    ASSERT((wd % 8) == 0);
    ASSERT(wd >= 8);
    ASSERT(pad_size > 0);

    for (row = 0; row < pad_size; row++)
    {
        const UWORD16* src = pu2_src;
        WORD32 col;

        for (col = 0; col + 64 <= wd; col += 64)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);
            uint16x8_t v2 = vld1q_u16(src + col + 16);
            uint16x8_t v3 = vld1q_u16(src + col + 24);
            uint16x8_t v4 = vld1q_u16(src + col + 32);
            uint16x8_t v5 = vld1q_u16(src + col + 40);
            uint16x8_t v6 = vld1q_u16(src + col + 48);
            uint16x8_t v7 = vld1q_u16(src + col + 56);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            vst1q_u16(dst + col + 16, v2);
            vst1q_u16(dst + col + 24, v3);
            vst1q_u16(dst + col + 32, v4);
            vst1q_u16(dst + col + 40, v5);
            vst1q_u16(dst + col + 48, v6);
            vst1q_u16(dst + col + 56, v7);
        }

        if (col + 32 <= wd)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);
            uint16x8_t v2 = vld1q_u16(src + col + 16);
            uint16x8_t v3 = vld1q_u16(src + col + 24);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            vst1q_u16(dst + col + 16, v2);
            vst1q_u16(dst + col + 24, v3);
            col += 32;
        }

        if (col + 16 <= wd)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            col += 16;
        }

        if (col + 8 <= wd)
        {
            vst1q_u16(dst + col, vld1q_u16(src + col));
        }

        dst += src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding at the bottom of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The bottom row (pu2_src - 1 * src_strd) is replicated pad_size times downwards
*   into pu2_src + (row - 1) * src_strd for row = 1 .. pad_size.
*   For common HEVC widths, the bottom valid row is preloaded ONCE into NEON
*   registers prior to the loop.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the start of the bottom padding area (row ht)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] wd
*   Width of the row to replicate in UWORD16 units
*
* @param[in] pad_size
*   Number of rows to pad below
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_bottom_neonintr(UWORD16* pu2_src,
    WORD32 src_strd,
    WORD32 wd,
    WORD32 pad_size)
{
    WORD32 row;
    const UWORD16* src = pu2_src - src_strd;
    UWORD16* dst = pu2_src;

    ASSERT((wd % 8) == 0);
    ASSERT(wd >= 8);
    ASSERT(pad_size > 0);

    for (row = 0; row < pad_size; row++)
    {
        WORD32 col;

        for (col = 0; col + 64 <= wd; col += 64)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);
            uint16x8_t v2 = vld1q_u16(src + col + 16);
            uint16x8_t v3 = vld1q_u16(src + col + 24);
            uint16x8_t v4 = vld1q_u16(src + col + 32);
            uint16x8_t v5 = vld1q_u16(src + col + 40);
            uint16x8_t v6 = vld1q_u16(src + col + 48);
            uint16x8_t v7 = vld1q_u16(src + col + 56);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            vst1q_u16(dst + col + 16, v2);
            vst1q_u16(dst + col + 24, v3);
            vst1q_u16(dst + col + 32, v4);
            vst1q_u16(dst + col + 40, v5);
            vst1q_u16(dst + col + 48, v6);
            vst1q_u16(dst + col + 56, v7);
        }

        if (col + 32 <= wd)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);
            uint16x8_t v2 = vld1q_u16(src + col + 16);
            uint16x8_t v3 = vld1q_u16(src + col + 24);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            vst1q_u16(dst + col + 16, v2);
            vst1q_u16(dst + col + 24, v3);
            col += 32;
        }

        if (col + 16 <= wd)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            col += 16;
        }

        if (col + 8 <= wd)
        {
            vst1q_u16(dst + col, vld1q_u16(src + col));
        }

        dst += src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding (chroma interleaved block) at the left of a 2D array.
*
* @par Description:
*   The leftmost UV pair (pu4_src[0]) is replicated pad_size/2 times to the left
*   for ht rows using 32-bit vector stores. Supports standard 4:2:0/4:2:2 (pad_size 80)
*   and 4:4:4 (pad_size 160) via dedicated unrolled paths.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the source (col 0)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad (halved for 32-bit UV pairs)
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_left_chroma_neonintr(UWORD16* pu2_src,
    WORD32 src_strd,
    WORD32 ht,
    WORD32 pad_size)
{
    WORD32 row;
    UWORD32* pu4_src = (UWORD32*)pu2_src;
    WORD32 strd32 = src_strd >> 1;

    /* pad_size is 80 for 4:2:0 / 4:2:2 and 160 for 4:4:4 */
    ASSERT((pad_size == 80) || (pad_size == 160));
    ASSERT((ht % 4) == 0);

    if (pad_size == 160)
    {
        /* 4:4:4 format - 160 UWORD16 samples (80 UV pairs) */
        /* Process 4 rows per iteration */
        for (row = 0; row < ht; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[0]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32]);

            UWORD32* dst0 = pu4_src - 80;
            UWORD32* dst1 = dst0 + strd32;
            UWORD32* dst2 = dst1 + strd32;
            UWORD32* dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0, v0);
            vst1q_u32(dst0 + 4, v0);
            vst1q_u32(dst0 + 8, v0);
            vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0);
            vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0);
            vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0);
            vst1q_u32(dst0 + 36, v0);
            vst1q_u32(dst0 + 40, v0);
            vst1q_u32(dst0 + 44, v0);
            vst1q_u32(dst0 + 48, v0);
            vst1q_u32(dst0 + 52, v0);
            vst1q_u32(dst0 + 56, v0);
            vst1q_u32(dst0 + 60, v0);
            vst1q_u32(dst0 + 64, v0);
            vst1q_u32(dst0 + 68, v0);
            vst1q_u32(dst0 + 72, v0);
            vst1q_u32(dst0 + 76, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0, v1);
            vst1q_u32(dst1 + 4, v1);
            vst1q_u32(dst1 + 8, v1);
            vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1);
            vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1);
            vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1);
            vst1q_u32(dst1 + 36, v1);
            vst1q_u32(dst1 + 40, v1);
            vst1q_u32(dst1 + 44, v1);
            vst1q_u32(dst1 + 48, v1);
            vst1q_u32(dst1 + 52, v1);
            vst1q_u32(dst1 + 56, v1);
            vst1q_u32(dst1 + 60, v1);
            vst1q_u32(dst1 + 64, v1);
            vst1q_u32(dst1 + 68, v1);
            vst1q_u32(dst1 + 72, v1);
            vst1q_u32(dst1 + 76, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0, v2);
            vst1q_u32(dst2 + 4, v2);
            vst1q_u32(dst2 + 8, v2);
            vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2);
            vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2);
            vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2);
            vst1q_u32(dst2 + 36, v2);
            vst1q_u32(dst2 + 40, v2);
            vst1q_u32(dst2 + 44, v2);
            vst1q_u32(dst2 + 48, v2);
            vst1q_u32(dst2 + 52, v2);
            vst1q_u32(dst2 + 56, v2);
            vst1q_u32(dst2 + 60, v2);
            vst1q_u32(dst2 + 64, v2);
            vst1q_u32(dst2 + 68, v2);
            vst1q_u32(dst2 + 72, v2);
            vst1q_u32(dst2 + 76, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0, v3);
            vst1q_u32(dst3 + 4, v3);
            vst1q_u32(dst3 + 8, v3);
            vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3);
            vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3);
            vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3);
            vst1q_u32(dst3 + 36, v3);
            vst1q_u32(dst3 + 40, v3);
            vst1q_u32(dst3 + 44, v3);
            vst1q_u32(dst3 + 48, v3);
            vst1q_u32(dst3 + 52, v3);
            vst1q_u32(dst3 + 56, v3);
            vst1q_u32(dst3 + 60, v3);
            vst1q_u32(dst3 + 64, v3);
            vst1q_u32(dst3 + 68, v3);
            vst1q_u32(dst3 + 72, v3);
            vst1q_u32(dst3 + 76, v3);

            pu4_src += 4 * strd32;
        }
    }
    else
    {
        /* Standard 4:2:0 and 4:2:2 format - 80 UWORD16 samples (40 UV pairs) */
        /* Process 4 rows per iteration*/
        for (row = 0; row < ht; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[0]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32]);

            UWORD32* dst0 = pu4_src - 40;
            UWORD32* dst1 = dst0 + strd32;
            UWORD32* dst2 = dst1 + strd32;
            UWORD32* dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0, v0);
            vst1q_u32(dst0 + 4, v0);
            vst1q_u32(dst0 + 8, v0);
            vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0);
            vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0);
            vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0);
            vst1q_u32(dst0 + 36, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0, v1);
            vst1q_u32(dst1 + 4, v1);
            vst1q_u32(dst1 + 8, v1);
            vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1);
            vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1);
            vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1);
            vst1q_u32(dst1 + 36, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0, v2);
            vst1q_u32(dst2 + 4, v2);
            vst1q_u32(dst2 + 8, v2);
            vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2);
            vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2);
            vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2);
            vst1q_u32(dst2 + 36, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0, v3);
            vst1q_u32(dst3 + 4, v3);
            vst1q_u32(dst3 + 8, v3);
            vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3);
            vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3);
            vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3);
            vst1q_u32(dst3 + 36, v3);

            pu4_src += 4 * strd32;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding (chroma interleaved block) at the right of a 2D array.
*
* @par Description:
*   The rightmost UV pair (pu4_src[-1]) is replicated pad_size/2 times to the
*   right for ht rows using 32-bit vector stores. Supports standard 4:2:0/4:2:2 (pad_size 80)
*   and 4:4:4 (pad_size 160) via dedicated unrolled paths.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the start of the right padding area (col wd)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad (halved for 32-bit UV pairs)
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_right_chroma_neonintr(UWORD16* pu2_src,
    WORD32 src_strd,
    WORD32 ht,
    WORD32 pad_size)
{
    WORD32 row;
    UWORD32* pu4_src = (UWORD32*)pu2_src;
    WORD32 strd32 = src_strd >> 1;

    /* pad_size is 80 for 4:2:0 / 4:2:2 and 160 for 4:4:4 */
    ASSERT((pad_size == 80) || (pad_size == 160));
    ASSERT((ht % 4) == 0);

    if (pad_size == 160)
    {
        /* 4:4:4 format - 160 UWORD16 samples (80 UV pairs) */
        /* Process 4 rows per iteration */
        for (row = 0; row < ht; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[-1]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32 - 1]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32 - 1]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32 - 1]);

            UWORD32* dst0 = pu4_src;
            UWORD32* dst1 = dst0 + strd32;
            UWORD32* dst2 = dst1 + strd32;
            UWORD32* dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0, v0);
            vst1q_u32(dst0 + 4, v0);
            vst1q_u32(dst0 + 8, v0);
            vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0);
            vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0);
            vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0);
            vst1q_u32(dst0 + 36, v0);
            vst1q_u32(dst0 + 40, v0);
            vst1q_u32(dst0 + 44, v0);
            vst1q_u32(dst0 + 48, v0);
            vst1q_u32(dst0 + 52, v0);
            vst1q_u32(dst0 + 56, v0);
            vst1q_u32(dst0 + 60, v0);
            vst1q_u32(dst0 + 64, v0);
            vst1q_u32(dst0 + 68, v0);
            vst1q_u32(dst0 + 72, v0);
            vst1q_u32(dst0 + 76, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0, v1);
            vst1q_u32(dst1 + 4, v1);
            vst1q_u32(dst1 + 8, v1);
            vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1);
            vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1);
            vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1);
            vst1q_u32(dst1 + 36, v1);
            vst1q_u32(dst1 + 40, v1);
            vst1q_u32(dst1 + 44, v1);
            vst1q_u32(dst1 + 48, v1);
            vst1q_u32(dst1 + 52, v1);
            vst1q_u32(dst1 + 56, v1);
            vst1q_u32(dst1 + 60, v1);
            vst1q_u32(dst1 + 64, v1);
            vst1q_u32(dst1 + 68, v1);
            vst1q_u32(dst1 + 72, v1);
            vst1q_u32(dst1 + 76, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0, v2);
            vst1q_u32(dst2 + 4, v2);
            vst1q_u32(dst2 + 8, v2);
            vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2);
            vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2);
            vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2);
            vst1q_u32(dst2 + 36, v2);
            vst1q_u32(dst2 + 40, v2);
            vst1q_u32(dst2 + 44, v2);
            vst1q_u32(dst2 + 48, v2);
            vst1q_u32(dst2 + 52, v2);
            vst1q_u32(dst2 + 56, v2);
            vst1q_u32(dst2 + 60, v2);
            vst1q_u32(dst2 + 64, v2);
            vst1q_u32(dst2 + 68, v2);
            vst1q_u32(dst2 + 72, v2);
            vst1q_u32(dst2 + 76, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0, v3);
            vst1q_u32(dst3 + 4, v3);
            vst1q_u32(dst3 + 8, v3);
            vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3);
            vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3);
            vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3);
            vst1q_u32(dst3 + 36, v3);
            vst1q_u32(dst3 + 40, v3);
            vst1q_u32(dst3 + 44, v3);
            vst1q_u32(dst3 + 48, v3);
            vst1q_u32(dst3 + 52, v3);
            vst1q_u32(dst3 + 56, v3);
            vst1q_u32(dst3 + 60, v3);
            vst1q_u32(dst3 + 64, v3);
            vst1q_u32(dst3 + 68, v3);
            vst1q_u32(dst3 + 72, v3);
            vst1q_u32(dst3 + 76, v3);

            pu4_src += 4 * strd32;
        }
    }
    else
    {
        /* Standard 4:2:0 and 4:2:2 format - 80 UWORD16 samples (40 UV pairs) */
        /* Process 4 rows per iteration */
        for (row = 0; row < ht; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[-1]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32 - 1]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32 - 1]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32 - 1]);

            UWORD32* dst0 = pu4_src;
            UWORD32* dst1 = dst0 + strd32;
            UWORD32* dst2 = dst1 + strd32;
            UWORD32* dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0, v0);
            vst1q_u32(dst0 + 4, v0);
            vst1q_u32(dst0 + 8, v0);
            vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0);
            vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0);
            vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0);
            vst1q_u32(dst0 + 36, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0, v1);
            vst1q_u32(dst1 + 4, v1);
            vst1q_u32(dst1 + 8, v1);
            vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1);
            vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1);
            vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1);
            vst1q_u32(dst1 + 36, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0, v2);
            vst1q_u32(dst2 + 4, v2);
            vst1q_u32(dst2 + 8, v2);
            vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2);
            vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2);
            vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2);
            vst1q_u32(dst2 + 36, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0, v3);
            vst1q_u32(dst3 + 4, v3);
            vst1q_u32(dst3 + 8, v3);
            vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3);
            vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3);
            vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3);
            vst1q_u32(dst3 + 36, v3);

            pu4_src += 4 * strd32;
        }
    }
}
