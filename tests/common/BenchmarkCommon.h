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

#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "iv.h"
// clang-format on

#include "TestCommon.h"

// Returns human-readable architecture name ("C", "SSSE3", "SSE42", "ARMV8", etc.)
inline std::string GetArchName(IV_ARCH_T arch) { return get_arch_str(arch); }

// Returns the list of architectures to benchmark, starting with ARCH_NA
// (generic C), followed by all supported target architectures returned by
// getTstArch().
const std::vector<IV_ARCH_T>& GetBenchmarkArchitectures();

// Common representative PU block sizes for Luma benchmarks
const std::vector<std::pair<int, int>>& GetBenchmarkLumaPUSizes();

// Common representative PU block sizes for Chroma benchmarks
const std::vector<std::pair<int, int>>& GetBenchmarkChromaPUSizes();

// Implemented by each benchmark executable to register its benchmark suite
void RegisterAllBenchmarks();
