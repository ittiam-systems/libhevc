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
#include "dec_func_selector.h"

const ihevcd_func_selector_t ref = []() {
  ihevcd_func_selector_t ret = {};
  ihevcd_init_function_ptr_generic(&ret);
  return ret;
}();

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
const ihevcd_func_selector_t test_ssse3 = []() {
  ihevcd_func_selector_t ret = ref;
  ihevcd_init_function_ptr_ssse3(&ret);
  return ret;
}();

const ihevcd_func_selector_t test_sse42 = []() {
  ihevcd_func_selector_t ret = ref;
  ihevcd_init_function_ptr_sse42(&ret);
  return ret;
}();

#ifndef DISABLE_AVX2
const ihevcd_func_selector_t test_avx2 = []() {
  ihevcd_func_selector_t ret = ref;
  ihevcd_init_function_ptr_avx2(&ret);
  return ret;
}();
#endif

#elif defined(__aarch64__)
const ihevcd_func_selector_t test_arm64 = []() {
  ihevcd_func_selector_t ret = ref;
#ifndef DARWIN
  ihevcd_init_function_ptr_av8(&ret);
#endif
  return ret;
}();
#elif defined(__arm__)
const ihevcd_func_selector_t test_arm32 = []() {
  ihevcd_func_selector_t ret = ref;
#ifndef DARWIN
  ihevcd_init_function_ptr_a9q(&ret);
#endif
  return ret;
}();
#endif

const ihevcd_func_selector_t *get_dec_ref_func_ptr() { return &ref; }

const ihevcd_func_selector_t *get_dec_tst_func_ptr(IV_ARCH_T arch) {
  switch (arch) {
  case ARCH_NA:
    return &ref;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  case ARCH_X86_SSSE3:
    return &test_ssse3;
  case ARCH_X86_SSE42:
    return &test_sse42;
#ifndef DISABLE_AVX2
  case ARCH_X86_AVX2:
    return &test_avx2;
#endif
#elif defined(__aarch64__)
  case ARCH_ARMV8_GENERIC:
    return &test_arm64;
#elif defined(__arm__)
  case ARCH_ARM_A9Q:
    return &test_arm32;
#endif
  default:
    return nullptr;
  }
}
