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
*  ihevcd_trace.c
*
* @brief
*  Contains trace related functions
*
* @author
*  Ittiam
*
* @par List of Functions:
*   - ihevcd_trace_init()
*   - ihevcd_trace_deinit()
*
* @remarks
*  None
*
*******************************************************************************
*/
/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#ifdef TRACE
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "ihevc_typedefs.h"
#include "iv.h"
#include "ivd.h"
#include "ihevcd_cxa.h"

#include "ihevc_defs.h"
#include "ihevc_debug.h"
#include "ihevc_structs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_cabac_tables.h"

#include "ihevcd_defs.h"
#include "ihevcd_error.h"
#include "ihevcd_function_selector.h"
#include "ihevcd_structs.h"
#include "ihevcd_trace.h"



/*****************************************************************************/
/* Declare globals                                                           */
/*****************************************************************************/
/**
 * Trace context
 */
trace_t g_trace;
/**
 * Trace file name
 */
CHAR ac_trace_fname[] = "trace.txt";



/**
*******************************************************************************
*
* @brief
*  Function used for initialization of trace parameters
*
* @par Description:
*  Initialize trace structure elements
*
* @param[in] pc_fname
*  File name for trace dumps
*
* @returns  none
*
* @remarks
*  Uses global hence not thread safe
*
*******************************************************************************
*/

void ihevcd_trace_init(CHAR *pc_fname)
{
    trace_t *ps_trace = &g_trace;

    if(pc_fname == NULL)
        pc_fname = ac_trace_fname;

    ps_trace->fp = fopen(pc_fname, "w");

    if(NULL == ps_trace->fp)
    {
        exit(-1);
    }
    return;
}
/**
*******************************************************************************
*
* @brief
*  Function used for deinitialization of trace parameters
*
* @par Description:
*  Initialize trace structure elements
*
* @param[in] ps_trace
*  Pointer to trace context
*
* @returns  none
*
* @remarks
*  Uses global hence not thread safe
*
*******************************************************************************
*/
void ihevcd_trace_deinit(trace_t *ps_trace)
{
    if(NULL != ps_trace->fp)
    {
        fclose(ps_trace->fp);
    }
    return;
}

#endif /* TRACE */
