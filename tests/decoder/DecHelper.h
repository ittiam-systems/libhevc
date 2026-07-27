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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "BitsBuf.h"
#include "BitsFile.h"
#include "RawBuf.h"
#include "RawFile.h"
#include "TestCommon.h"

class DecHelper {
 public:
  class Builder {
   public:
    Builder() = default;
    ~Builder() = default;

    Builder& setFormat(libhevc::test::Format format) {
      mFormat = format;
      return *this;
    }
    Builder& setCores(size_t cores) {
      mCores = cores;
      return *this;
    }
    Builder& setArch(libhevc::test::Arch arch) {
      mArch = arch;
      return *this;
    }
    Builder& setInputFilePath(const std::string& path) {
      mInputFilePath = path;
      return *this;
    }
    Builder& setOutputFilePath(const std::string& path) {
      mOutputFilePath = path;
      return *this;
    }
    Builder& setRefMd5Path(const std::string& path) {
      mRefMd5Path = path;
      return *this;
    }

    // Validates parameters and returns an initialized DecHelper instance
    std::unique_ptr<DecHelper> build();

   private:
    libhevc::test::Format mFormat = libhevc::test::Format::yuv420p;
    size_t mCores = 1;
    libhevc::test::Arch mArch = libhevc::test::Arch::defaultArch;
    std::optional<std::string> mInputFilePath;
    std::optional<std::string> mOutputFilePath;
    std::optional<std::string> mRefMd5Path;
    friend class DecHelper;
  };

  ~DecHelper() { deinitDecoder(); }

  // Executes the entire decoding process using the configured input/output
  // files. Uses inline GTest assertions to validate frame MD5 checksums.
  bool decodeFile();

  // Low-level chunk-based frame decoding
  bool decodeHeader(size_t& bytesConsumed);
  bool decodeFrame(size_t& bytesConsumed, bool& frameReady, size_t& frameIndex,
                   bool noInput = false);
  bool flushDecoder(size_t& frameIndex);
  void deinitDecoder();

  // Computes plane-by-plane MD5 checksum of a raw frame (ignoring stride
  // padding row-by-row)
  std::string computeFrameMd5(const RawBuf& buf);

  size_t width() const { return mWidth; }
  size_t height() const { return mHeight; }
  bool isHeaderDecoded() const { return mHeaderDecoded; }

 private:
  DecHelper() = default;  // Private constructor, instantiated only via Builder
  bool initDecoder();
  void processDecodedFrame(RawBuf& buf, size_t& frameIndex);

  void* mCodec = nullptr;  // Opaque pointer to internal iv_obj_t
  bool mHeaderDecoded = false;
  bool mInFlushMode = false;
  size_t mWidth = 0;
  size_t mHeight = 0;
  libhevc::test::Format mFormat = libhevc::test::Format::yuv420p;
  size_t mCores = 1;
  libhevc::test::Arch mArch = libhevc::test::Arch::defaultArch;

  std::optional<std::string> mInputFilePath;
  std::optional<std::string> mOutputFilePath;
  std::optional<std::string> mRefMd5Path;
  std::vector<std::string> mRefMd5s;

  BitsFile mBitsFile;
  RawFile mOutFile;

  BitsBuf mInputBuf;
  RawBuf mOutputBuf;
};
