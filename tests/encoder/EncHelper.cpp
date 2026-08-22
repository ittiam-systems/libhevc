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

#include "EncHelper.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "BitsFile.h"
#include "Md5Wrapper.h"
#include "RawFile.h"

// Include internal libhevc encoder headers here inside the .cpp file to prevent
// public header pollution
// clang-format off
extern "C" {
#include "ihevc_typedefs.h"
#include "itt_video_api.h"  // Defines IV_ARCH_T and its ARCH_* values
#include "ihevce_api.h"
#include "ihevce_plugin.h"
}
// clang-format on

static IV_ARCH_T getDefaultArch() { return ARCH_NA; }

// Map architecture to native enum using local mapper helper
static IV_ARCH_T getNativeArch(libhevc::test::Arch arch) {
  switch (arch) {
    case libhevc::test::Arch::defaultArch:
      return getDefaultArch();
    case libhevc::test::Arch::generic:
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
      return ARCH_NA;
#else  // defined(__arm__) || defined(_M_ARM) || defined(__aarch64__) ||
       // defined(_M_ARM64)
      return ARCH_ARM_NONEON;
#endif
    case libhevc::test::Arch::armv7:
      return ARCH_ARM_A9Q;
    case libhevc::test::Arch::armv8:
      return ARCH_ARM_V8_NEON;
    case libhevc::test::Arch::x86_sse42:
      return ARCH_X86_SSE4;
    case libhevc::test::Arch::x86_avx2:
      return ARCH_X86_AVX2;
  }
  return ARCH_NA;
}

#ifndef DISABLE_MD5
// Re-declare the internal C-struct layout of iv_enc_recon_data_buffs_t
// to avoid including private, fragile internal encoder headers.
typedef struct {
  int32_t i4_size;
  int32_t i4_poc;
  int32_t i4_end_flag;
  int32_t i4_is_last_buf;
  void* pv_y_buf;
  void* pv_cb_buf;
  void* pv_cr_buf;
  int32_t i4_y_pixels;
  int32_t i4_uv_pixels;
} iv_enc_recon_data_buffs_t;

// Reconstructed frame callback function called by the encoder
extern "C" IV_API_CALL_STATUS_T reconCallback(void* pv_recon_cb_handle,
                                              void* pv_curr_out,
                                              WORD32 i4_bitrate_instance,
                                              WORD32 i4_res_instance) {
  (void)i4_bitrate_instance;
  (void)i4_res_instance;
  if (pv_recon_cb_handle && pv_curr_out) {
    EncHelper* helper = static_cast<EncHelper*>(pv_recon_cb_handle);
    std::string md5 = helper->computeReconFrameMd5(pv_curr_out);
    helper->addReconMd5(md5);
  }
  return IV_SUCCESS;
}
#endif

std::unique_ptr<EncHelper> EncHelper::Builder::build() {
  std::unique_ptr<EncHelper> helper(new EncHelper());
  helper->mFormat = mFormat;
  helper->mCores = mCores;
  helper->mArch = mArch;
  helper->mWidth = mWidth;
  helper->mHeight = mHeight;
  helper->mBitrate = mBitrate;
  helper->mFrameRateNum = mFrameRateNum;
  helper->mFrameRateDenom = mFrameRateDenom;
  helper->mRcMode = mRcMode;
  helper->mFrameQp = mFrameQp;
  helper->mQualityPreset = mQualityPreset;
  helper->mMaxClosedGop = mMaxClosedGop;
  helper->mMinClosedGop = mMinClosedGop;
  helper->mMaxCraOpenGop = mMaxCraOpenGop;
  helper->mProfile = mProfile;
  helper->mTier = mTier;
  helper->mInputFilePath = mInputFilePath;
  helper->mOutFilePath = mOutFilePath;
  helper->mReconMd5FilePath = mReconMd5FilePath;

  // Open input raw YUV file
  if (!mInputFilePath.has_value() ||
      !helper->mRawFile.open(*mInputFilePath, true)) {
    return nullptr;
  }

  // Open output bitstream file
  if (!mOutFilePath.has_value() ||
      !helper->mBitsFile.open(*mOutFilePath, false)) {
    return nullptr;
  }

  // Initialize native encoder
  if (!helper->initEncoder()) {
    return nullptr;
  }
  return helper;
}

bool EncHelper::initEncoder() {
  ihevce_static_cfg_params_t params = {};

  // Get default configuration parameters from encoder library
  if (IHEVCE_EFAIL == ihevce_set_def_params(&params)) {
    return false;
  }

  // Overwrite default configuration with builder properties
  params.s_src_prms.i4_width = mWidth;
  params.s_src_prms.i4_height = mHeight;
  params.s_multi_thrd_prms.i4_max_num_cores = mCores;

  // Map architecture to native enum using local mapper helper
  params.e_arch_type = getNativeArch(mArch);

  // Map chroma format to native enum
  IV_COLOR_FORMAT_T colorFmt = IV_YUV_420P;
  switch (mFormat) {
    case libhevc::test::Format::yuv400p:
      colorFmt = IV_GRAY;
      break;
    case libhevc::test::Format::yuv420p:
      colorFmt = IV_YUV_420P;
      break;
    case libhevc::test::Format::yuv420sp_uv:
      colorFmt = IV_YUV_420SP_UV;
      break;
    case libhevc::test::Format::yuv420sp_vu:
      colorFmt = IV_YUV_420SP_VU;
      break;
    case libhevc::test::Format::yuv422p:
      colorFmt = IV_YUV_422P;
      break;
    case libhevc::test::Format::yuv444p:
      colorFmt = IV_YUV_444P;
      break;
  }
  params.s_src_prms.inp_chr_format = colorFmt;

  // Set bitrates (Target & Peak)
  params.s_tgt_lyr_prms.as_tgt_params[0].ai4_tgt_bitrate[0] = mBitrate;
  params.s_tgt_lyr_prms.as_tgt_params[0].ai4_peak_bitrate[0] = mBitrate;

  // Set frame rate
  params.s_src_prms.i4_frm_rate_num = mFrameRateNum;
  params.s_src_prms.i4_frm_rate_denom = mFrameRateDenom;

  // Set rate control mode, QP, and quality preset
  params.s_config_prms.i4_rate_control_mode = mRcMode;
  params.s_tgt_lyr_prms.as_tgt_params[0].ai4_frame_qp[0] = mFrameQp;
  params.s_tgt_lyr_prms.as_tgt_params[0].i4_quality_preset =
      static_cast<IHEVCE_QUALITY_CONFIG_T>(mQualityPreset);

  // Set GOP parameters
  params.s_coding_tools_prms.i4_max_closed_gop_period = mMaxClosedGop;
  params.s_coding_tools_prms.i4_min_closed_gop_period = mMinClosedGop;
  params.s_coding_tools_prms.i4_max_cra_open_gop_period = mMaxCraOpenGop;

  // Set Profile & Tier
  params.s_out_strm_prms.i4_codec_profile = mProfile;
  params.s_out_strm_prms.i4_codec_tier = mTier;

  // Configure reconstruction frame dumping if requested
  if (mReconMd5FilePath.has_value()) {
    params.i4_save_recon = 1;
  } else {
    params.i4_save_recon = 0;
  }
  // TODO: Remove the following once recon save is verified
  params.i4_save_recon = 0;

  // Call native init
  if (ihevce_init(&params, &mCodec) != IHEVCE_EOK) {
    mCodec = nullptr;
    return false;
  }

  if (!mInputBuf.allocBuffer(mWidth, mHeight, /* bitDepth */ 8, mFormat)) {
    deinitEncoder();
    return false;
  }

  if (!mOutputBuf.allocBuffer(mWidth * mHeight * 3 >> 1)) {
    deinitEncoder();
    return false;
  }

  mReconMd5s.clear();
  return true;
}

void EncHelper::deinitEncoder() {
  if (mCodec) {
    ihevce_close(mCodec);
    mCodec = nullptr;
  }
}

void EncHelper::addReconMd5(const std::string& md5) {
  std::lock_guard<std::mutex> lock(mReconMutex);
  mReconMd5s.push_back(md5);
}

std::string EncHelper::computeReconFrameMd5(void* pv_curr_out) {
  iv_enc_recon_data_buffs_t* reconBuf =
      static_cast<iv_enc_recon_data_buffs_t*>(pv_curr_out);
  Md5Wrapper md5;
  md5.init();

  // Y plane (contiguous, size = mWidth * mHeight)
  if (reconBuf->pv_y_buf) {
    md5.update(static_cast<const uint8_t*>(reconBuf->pv_y_buf),
               mWidth * mHeight);
  }

  // Chroma planes (U and V packed contiguously in pv_cb_buf)
  if (mFormat != libhevc::test::Format::yuv400p && reconBuf->pv_cb_buf) {
    size_t uvWidth =
        (mFormat == libhevc::test::Format::yuv444p) ? mWidth : mWidth / 2;
    size_t uvHeight = (mFormat == libhevc::test::Format::yuv444p ||
                       mFormat == libhevc::test::Format::yuv422p)
                          ? mHeight
                          : mHeight / 2;
    size_t chromaSize = uvWidth * uvHeight;
    md5.update(static_cast<const uint8_t*>(reconBuf->pv_cb_buf),
               chromaSize * 2);
  }

  return md5.finalize();
}

bool EncHelper::encodeFrame(bool& bytesProduced, bool noInput) {
  bytesProduced = false;
  if (!mCodec) {
    return false;
  }

  ihevce_inp_buf_t inpPic = {};
  ihevce_out_buf_t outPic = {};

  ihevce_inp_buf_t* pInpPic = nullptr;
  if (!noInput) {
    // Map RawBuf planes, strides, and sizes directly into inpPic
    size_t planesCount = mInputBuf.numPlanes();
    for (size_t i = 0; i < planesCount; ++i) {
      inpPic.apv_inp_planes[i] = mInputBuf.planeData(i);
      inpPic.ai4_inp_strd[i] = mInputBuf.stride(i);
      inpPic.ai4_inp_size[i] = mInputBuf.stride(i) * mInputBuf.planeHeight(i);
    }

    inpPic.i4_curr_bitrate = mBitrate;
    inpPic.i4_curr_peak_bitrate = mBitrate;
    inpPic.u8_pts = 0;
    inpPic.i4_force_idr_flag = 0;
    pInpPic = &inpPic;
  }

  IHEVCE_PLUGIN_STATUS_T status = ihevce_encode(mCodec, pInpPic, &outPic);
  if (status != IHEVCE_EOK) {
    return false;
  }

  if (outPic.i4_bytes_generated > 0) {
    std::memcpy(mOutputBuf.data(), outPic.pu1_output_buf,
                outPic.i4_bytes_generated);
    mOutputBuf.setSize(outPic.i4_bytes_generated);
    mOutputBuf.setOffset(0);
    bytesProduced = true;

    // Write output bitstream chunk directly to file
    mBitsFile.write(mOutputBuf, outPic.i4_bytes_generated);
  }

  return true;
}

bool EncHelper::flushEncoder() {
  if (!mCodec) {
    return false;
  }

  // Loop internally and delegate to encodeFrame with noInput = true
  while (true) {
    bool bytesProduced = false;
    if (!encodeFrame(bytesProduced, true)) {
      return false;
    }
    if (!bytesProduced) {
      break;
    }
  }

  return true;
}

bool EncHelper::encodeFile() {
  mReconMd5s.clear();

  // Encode SPS/PPS header first
  ihevce_out_buf_t outHdr = {};
  if (IHEVCE_EOK == ihevce_encode_header(mCodec, &outHdr)) {
    if (outHdr.i4_bytes_generated > 0) {
      mOutputBuf.setSize(outHdr.i4_bytes_generated);
      mOutputBuf.setOffset(0);
      std::memcpy(mOutputBuf.data(), outHdr.pu1_output_buf,
                  outHdr.i4_bytes_generated);
      mBitsFile.write(mOutputBuf, outHdr.i4_bytes_generated);
    }
  }

  // Main frame encoding loop
  while (mRawFile.read(mInputBuf)) {
    bool bytesProduced = false;
    if (!encodeFrame(bytesProduced)) {
      return false;
    }
  }

  // Flush encoder
  if (!flushEncoder()) {
    return false;
  }

  // Write all computed reconstruction MD5s to the output path if provided
  if (mReconMd5FilePath.has_value()) {
    std::ofstream md5File(*mReconMd5FilePath);
    if (!md5File.is_open()) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mReconMutex);
    for (const auto& md5 : mReconMd5s) {
      md5File << md5 << " 0" << "\n";
    }
    mReconMd5s.clear();
  }

  return true;
}
