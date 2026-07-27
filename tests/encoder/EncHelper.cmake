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

# Define the encoder test helper library
add_library(libhevc_enc_helper STATIC
  ${HEVC_ROOT}/tests/encoder/EncHelper.cpp
)

# Export include directories
target_include_directories(libhevc_enc_helper PUBLIC
  ${HEVC_ROOT}/tests/encoder
  ${HEVC_ROOT}/tests/common
  ${HEVC_ROOT}/common
  ${HEVC_ROOT}/encoder
)

# Link against native encoder and common test helper library
target_link_libraries(libhevc_enc_helper PUBLIC
  libhevcenc
  libhevc_test_common
)

# Define the encoder regression test executable
libhevc_add_executable(
  hevc_enc_tests
  libhevc_enc_helper
  SOURCES ${HEVC_ROOT}/tests/encoder/EncTests.cpp
  LIBS libhevc_dec_helper GTest::gtest_main
)
target_compile_definitions(hevc_enc_tests PRIVATE "PROJECT_DIR=\"${HEVC_ROOT}\"")
gtest_discover_tests(
  hevc_enc_tests
  ${HEVC_GTEST_DISCOVERY_MODE}
  PROPERTIES ENVIRONMENT "HEVC_TEST_DIR=${hevc_test_data_SOURCE_DIR}"
)


