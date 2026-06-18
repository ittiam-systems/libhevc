@/*****************************************************************************
@*
@* Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore
@*
@* Licensed under the Apache License, Version 2.0 (the "License");
@* you may not use this file except in compliance with the License.
@* You may obtain a copy of the License at:
@*
@* http://www.apache.org/licenses/LICENSE-2.0
@*
@* Unless required by applicable law or agreed to in writing, software
@* distributed under the License is distributed on an "AS IS" BASIS,
@* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@* See the License for the specific language governing permissions and
@* limitations under the License.
@*
@*****************************************************************************/
@/**
@ *******************************************************************************
@ * ,:file
@ *  ihevc_mem_fns_neon.s
@ *
@ * ,:brief
@ *  Contains function definitions for memory manipulation
@ *
@ * ,:author
@ *  Naveen SR
@ *
@ * ,:par List of Functions:
@ *  - ihevc_memset_16bit_mul_8()
@ *
@ * ,:remarks
@ *  None
@ *
@ *******************************************************************************
@*/

@void ihevc_memset_16bit_mul_8(UWORD16 *pu2_dst,
@                                   UWORD16 value,
@                                   UWORD8 num_words)
@**************Variables Vs Registers*************************
@   r0 => *pu2_dst
@   r1 => value
@   r2 => num_words

.text
.p2align 2



    .global ihevc_memset_16bit_mul_8_a9q
.type ihevc_memset_16bit_mul_8_a9q, %function

ihevc_memset_16bit_mul_8_a9q:

@ Assumptions: num_words is either 8, 16 or 32

    @ Memset 8 words
    VDUP.16     d0,r1
LOOP_MEMSET_16BIT_MUL_8:
    VST1.16     d0,[r0]!
    VST1.16     d0,[r0]!

    SUBS        r2,r2,#8
    BNE         LOOP_MEMSET_16BIT_MUL_8

    BX          LR




@void ihevc_memset_16bit(UWORD16 *pu2_dst,
@                       UWORD16 value,
@                       UWORD8 num_words)
@**************Variables Vs Registers*************************
@   r0 => *pu2_dst
@   r1 => value
@   r2 => num_words



    .global ihevc_memset_16bit_a9q
.type ihevc_memset_16bit_a9q, %function

ihevc_memset_16bit_a9q:
    SUBS        r2,#8
    BLT         ARM_MEMSET_16BIT
    VDUP.16     d0,r1
LOOP_NEON_MEMSET_16BIT:
    @ Memset 8 words
    VST1.16     d0,[r0]!
    VST1.16     d0,[r0]!

    SUBS        r2,#8
    BGE         LOOP_NEON_MEMSET_16BIT
    CMP         r2,#-8
    BXEQ        LR

ARM_MEMSET_16BIT:
    ADD         r2,#8

LOOP_ARM_MEMSET_16BIT:
    STRH        r1,[r0],#2
    SUBS        r2,#1
    BNE         LOOP_ARM_MEMSET_16BIT
    BX          LR




    .section .note.GNU-stack,"",%progbits

