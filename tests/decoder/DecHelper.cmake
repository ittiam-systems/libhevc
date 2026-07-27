#******************************************************************************
#
# Copyright (C) 2026 Ittiam Systems Pvt Ltd, Bangalore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************

# Define the decoder test helper library
add_library(libhevc_dec_helper STATIC
  ${HEVC_ROOT}/tests/decoder/DecHelper.cpp
)

# Export include directories
target_include_directories(libhevc_dec_helper PUBLIC
  ${HEVC_ROOT}/tests/decoder
  ${HEVC_ROOT}/tests/common
  ${HEVC_ROOT}/common
  ${HEVC_ROOT}/decoder
)

# Link against the native decoder, common test helper, and GTest/GMock
target_link_libraries(libhevc_dec_helper PUBLIC
  libhevcdec
  libhevc_test_common
  GTest::gtest
  GTest::gmock
)

# Define the decoder regression test executable
libhevc_add_executable(
  hevc_dec_tests
  libhevc_dec_helper
  SOURCES ${HEVC_ROOT}/tests/decoder/DecTests.cpp
  LIBS GTest::gtest_main
)
target_compile_definitions(hevc_dec_tests PRIVATE "PROJECT_DIR=\"${HEVC_ROOT}\"")
gtest_discover_tests(
  hevc_dec_tests
  ${HEVC_GTEST_DISCOVERY_MODE}
  PROPERTIES ENVIRONMENT "HEVC_TEST_DIR=${hevc_test_data_SOURCE_DIR}"
)


