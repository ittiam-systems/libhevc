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
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "BitsBuf.h"
#include "BitsFile.h"
#include "RawBuf.h"
#include "RawFile.h"
#include "TestEnums.h"

class EncHelper {
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
    Builder& setResolution(size_t width, size_t height) {
      mWidth = width;
      mHeight = height;
      return *this;
    }
    Builder& setBitrate(size_t bitrate) {
      mBitrate = bitrate;
      return *this;
    }
    Builder& setFrameRate(size_t num, size_t denom) {
      mFrameRateNum = num;
      mFrameRateDenom = denom;
      return *this;
    }
    Builder& setRcMode(size_t rcMode) {
      mRcMode = rcMode;
      return *this;
    }  // Rate control mode: VBR, CQP, or CBR
    Builder& setFrameQp(size_t qp) {
      mFrameQp = qp;
      return *this;
    }

    // Coding Tool & GOP Parameters
    Builder& setQualityPreset(size_t preset) {
      mQualityPreset = preset;
      return *this;
    }  // Quality or speed preset
    Builder& setGopParameters(size_t maxClosed, size_t minClosed,
                              size_t maxCraOpen) {
      mMaxClosedGop = maxClosed;
      mMinClosedGop = minClosed;
      mMaxCraOpenGop = maxCraOpen;
      return *this;
    }
    Builder& setProfileAndTier(size_t profile, size_t tier) {
      mProfile = profile;  // Profile: Main
      mTier = tier;        // Tier: Main or High
      return *this;
    }

    // File I/O Parameters
    Builder& setInputFilePath(const std::string& path) {
      mInputFilePath = path;
      return *this;
    }
    Builder& setOutFilePath(const std::string& path) {
      mOutFilePath = path;
      return *this;
    }
    Builder& setOutputReconMd5Path(const std::string& path) {
      mReconMd5FilePath = path;
      return *this;
    }

    // Validates parameters and returns an initialized EncHelper instance
    std::unique_ptr<EncHelper> build();

   private:
    libhevc::test::Format mFormat = libhevc::test::Format::yuv420p;
    size_t mCores = 1;
    libhevc::test::Arch mArch = libhevc::test::Arch::defaultArch;
    size_t mWidth = 0;
    size_t mHeight = 0;
    size_t mBitrate = 1000000;
    size_t mFrameRateNum = 30;
    size_t mFrameRateDenom = 1;
    size_t mRcMode = 2;
    size_t mFrameQp = 32;
    size_t mQualityPreset = 3;
    size_t mMaxClosedGop = 30;
    size_t mMinClosedGop = 1;
    size_t mMaxCraOpenGop = 30;
    size_t mProfile = 1;
    size_t mTier = 0;

    std::optional<std::string> mInputFilePath;
    std::optional<std::string> mOutFilePath;
    std::optional<std::string> mReconMd5FilePath;
    friend class EncHelper;
  };

  ~EncHelper() { deinitEncoder(); }

  // Executes the entire encoding process using the configured input/output
  // files. Computes plane-by-plane MD5 checksums of reconstruction frames
  // and writes them to the specified output file.
  bool encodeFile();

  // Low-level frame encoding
  bool encodeFrame(bool& bytesProduced, bool noInput = false);
  bool flushEncoder();
  void deinitEncoder();

  // Thread-safe callbacks for reconstruction frames
  void addReconMd5(const std::string& md5);
  std::string computeReconFrameMd5(void* pv_curr_out);

 private:
  EncHelper() = default;  // Private constructor, instantiated only via Builder
  bool initEncoder();

  void* mCodec = nullptr;  // Opaque pointer to internal ihevce_codec_t
  libhevc::test::Format mFormat = libhevc::test::Format::yuv420p;
  size_t mCores = 1;
  libhevc::test::Arch mArch = libhevc::test::Arch::defaultArch;
  size_t mWidth = 0;
  size_t mHeight = 0;
  size_t mBitrate = 1000000;
  size_t mFrameRateNum = 30;
  size_t mFrameRateDenom = 1;
  size_t mRcMode = 2;
  size_t mFrameQp = 32;
  size_t mQualityPreset = 3;
  size_t mMaxClosedGop = 30;
  size_t mMinClosedGop = 1;
  size_t mMaxCraOpenGop = 30;
  size_t mProfile = 1;
  size_t mTier = 0;

  std::optional<std::string> mInputFilePath;
  std::optional<std::string> mOutFilePath;
  std::optional<std::string> mReconMd5FilePath;

  std::mutex mReconMutex;
  std::vector<std::string> mReconMd5s;

  RawFile mRawFile;
  BitsFile mBitsFile;

  RawBuf mInputBuf;
  BitsBuf mOutputBuf;
};
