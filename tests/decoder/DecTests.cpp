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

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "DecHelper.h"
#include "TestCommon.h"

namespace libhevc {
namespace test {

struct DecodeStreamConfig {
  std::string inputFile;
  Format format;
  std::string refMd5File;
};

void PrintTo(const DecodeStreamConfig& config, ::std::ostream* os) {
  *os << config.inputFile;
}

// Helper to resolve absolute file path for test assets
static std::string getFullPath(const std::string& inputFile) {
  if (const char* envPath = std::getenv("HEVC_TEST_DIR")) {
    return std::string(envPath) + "/" + inputFile;
  }
  return inputFile;
}

class DecTestFixture
    : public ::testing::TestWithParam<std::tuple<DecodeStreamConfig, size_t>> {
 protected:
  DecTestFixture() = default;
  ~DecTestFixture() override = default;
};

TEST_P(DecTestFixture, DecodeVerify) {
  auto [config, cores] = GetParam();

  std::string inputPath = getFullPath(config.inputFile);
  std::string refMd5Path = getFullPath(config.refMd5File);

  auto builder = DecHelper::Builder()
                     .setInputFilePath(inputPath)
                     .setRefMd5Path(refMd5Path)
                     .setFormat(config.format)
                     .setCores(cores);

  std::unique_ptr<DecHelper> helper = builder.build();
  ASSERT_NE(helper, nullptr)
      << "Failed to build DecHelper for input: " << inputPath;
  EXPECT_TRUE(helper->decodeFile())
      << "Decoding failed for input: " << inputPath;
}

static const std::vector<DecodeStreamConfig> kDecodeStreams = {
    {"bbb_176x144_yuv400.hevc", Format::yuv400p, "bbb_176x144_yuv400_md5.txt"},
    {"bbb_176x144_yuv420.hevc", Format::yuv420p, "bbb_176x144_yuv420_md5.txt"},
    {"bbb_176x144_yuv422.hevc", Format::yuv422p, "bbb_176x144_yuv422_md5.txt"},
    {"bbb_176x144_yuv444.hevc", Format::yuv444p, "bbb_176x144_yuv444_md5.txt"}};

INSTANTIATE_TEST_SUITE_P(DecoderRegression, DecTestFixture,
                         ::testing::Combine(::testing::ValuesIn(kDecodeStreams),
                                            ::testing::Values(1, 2, 3, 4)));

}  // namespace test
}  // namespace libhevc
