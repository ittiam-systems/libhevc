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
#pragma once

// clang-format off
#include "ihevc_typedefs.h"
extern "C" {
#include "ihevcd_function_selector.h"
#include "iv.h"
}
// clang-format on

typedef func_selector_t ihevcd_func_selector_t;

const ihevcd_func_selector_t *get_dec_ref_func_ptr();
const ihevcd_func_selector_t *get_dec_tst_func_ptr(IV_ARCH_T arch);
