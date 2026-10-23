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
*  ihevc_padding_neon_intr.c
*
* @brief
*  Contains ARM NEON intrinsic function definitions for 8-bit padding
*  operations in HEVC.
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_pad_top_neonintr()
*  - ihevc_pad_bottom_neonintr()
*
* @remarks
*  Optimized with ARM NEON intrinsics (ARMv8 / AArch64 / AArch32 NEON).
*  Exact bit-exact replacement for C implementations in ihevc_padding.c.
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
*   Padding at the top of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The top row (row 0) is replicated pad_size times upwards.
*
* @param[in,out] pu1_src
*   UWORD8 pointer to the top valid row (row 0)
*
* @param[in] src_strd
*   Source buffer stride in UWORD8 units
*
* @param[in] wd
*   Width of the row to replicate in UWORD8 units
*
* @param[in] pad_size
*   Number of rows to pad above
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_pad_top_neonintr(UWORD8 *pu1_src,
                            WORD32 src_strd,
                            WORD32 wd,
                            WORD32 pad_size)
{
    WORD32 row;
    UWORD8 *dst = pu1_src - pad_size * src_strd;

    for(row = 0; row < pad_size; row++)
    {
        const UWORD8 *src = pu1_src;
        WORD32 col;

        for(col = 0; col + 128 <= wd; col += 128)
        {
            uint8x16_t v0 = vld1q_u8(src + col + 0);
            uint8x16_t v1 = vld1q_u8(src + col + 16);
            uint8x16_t v2 = vld1q_u8(src + col + 32);
            uint8x16_t v3 = vld1q_u8(src + col + 48);
            uint8x16_t v4 = vld1q_u8(src + col + 64);
            uint8x16_t v5 = vld1q_u8(src + col + 80);
            uint8x16_t v6 = vld1q_u8(src + col + 96);
            uint8x16_t v7 = vld1q_u8(src + col + 112);

            vst1q_u8(dst + col + 0, v0);
            vst1q_u8(dst + col + 16, v1);
            vst1q_u8(dst + col + 32, v2);
            vst1q_u8(dst + col + 48, v3);
            vst1q_u8(dst + col + 64, v4);
            vst1q_u8(dst + col + 80, v5);
            vst1q_u8(dst + col + 96, v6);
            vst1q_u8(dst + col + 112, v7);
        }

        if(col + 64 <= wd)
        {
            uint8x16_t v0 = vld1q_u8(src + col + 0);
            uint8x16_t v1 = vld1q_u8(src + col + 16);
            uint8x16_t v2 = vld1q_u8(src + col + 32);
            uint8x16_t v3 = vld1q_u8(src + col + 48);

            vst1q_u8(dst + col + 0, v0);
            vst1q_u8(dst + col + 16, v1);
            vst1q_u8(dst + col + 32, v2);
            vst1q_u8(dst + col + 48, v3);
            col += 64;
        }

        if(col + 32 <= wd)
        {
            uint8x16_t v0 = vld1q_u8(src + col + 0);
            uint8x16_t v1 = vld1q_u8(src + col + 16);

            vst1q_u8(dst + col + 0, v0);
            vst1q_u8(dst + col + 16, v1);
            col += 32;
        }

        if(col + 16 <= wd)
        {
            vst1q_u8(dst + col, vld1q_u8(src + col));
            col += 16;
        }

        if(col + 8 <= wd)
        {
            vst1_u8(dst + col, vld1_u8(src + col));
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
*   The bottom row (pu1_src - 1 * src_strd) is replicated pad_size times downwards
*   into pu1_src + (row - 1) * src_strd for row = 1 .. pad_size.
*
* @param[in,out] pu1_src
*   UWORD8 pointer to the start of the bottom padding area (row ht)
*
* @param[in] src_strd
*   Source buffer stride in UWORD8 units
*
* @param[in] wd
*   Width of the row to replicate in UWORD8 units
*
* @param[in] pad_size
*   Number of rows to pad below
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_pad_bottom_neonintr(UWORD8 *pu1_src,
                               WORD32 src_strd,
                               WORD32 wd,
                               WORD32 pad_size)
{
    WORD32 row;
    const UWORD8 *src = pu1_src - src_strd;
    UWORD8 *dst = pu1_src;

    for(row = 0; row < pad_size; row++)
    {
        WORD32 col;

        for(col = 0; col + 128 <= wd; col += 128)
        {
            uint8x16_t v0 = vld1q_u8(src + col + 0);
            uint8x16_t v1 = vld1q_u8(src + col + 16);
            uint8x16_t v2 = vld1q_u8(src + col + 32);
            uint8x16_t v3 = vld1q_u8(src + col + 48);
            uint8x16_t v4 = vld1q_u8(src + col + 64);
            uint8x16_t v5 = vld1q_u8(src + col + 80);
            uint8x16_t v6 = vld1q_u8(src + col + 96);
            uint8x16_t v7 = vld1q_u8(src + col + 112);

            vst1q_u8(dst + col + 0, v0);
            vst1q_u8(dst + col + 16, v1);
            vst1q_u8(dst + col + 32, v2);
            vst1q_u8(dst + col + 48, v3);
            vst1q_u8(dst + col + 64, v4);
            vst1q_u8(dst + col + 80, v5);
            vst1q_u8(dst + col + 96, v6);
            vst1q_u8(dst + col + 112, v7);
        }

        if(col + 64 <= wd)
        {
            uint8x16_t v0 = vld1q_u8(src + col + 0);
            uint8x16_t v1 = vld1q_u8(src + col + 16);
            uint8x16_t v2 = vld1q_u8(src + col + 32);
            uint8x16_t v3 = vld1q_u8(src + col + 48);

            vst1q_u8(dst + col + 0, v0);
            vst1q_u8(dst + col + 16, v1);
            vst1q_u8(dst + col + 32, v2);
            vst1q_u8(dst + col + 48, v3);
            col += 64;
        }

        if(col + 32 <= wd)
        {
            uint8x16_t v0 = vld1q_u8(src + col + 0);
            uint8x16_t v1 = vld1q_u8(src + col + 16);

            vst1q_u8(dst + col + 0, v0);
            vst1q_u8(dst + col + 16, v1);
            col += 32;
        }

        if(col + 16 <= wd)
        {
            vst1q_u8(dst + col, vld1q_u8(src + col));
            col += 16;
        }

        if(col + 8 <= wd)
        {
            vst1_u8(dst + col, vld1_u8(src + col));
        }

        dst += src_strd;
    }
}
