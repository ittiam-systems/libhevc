enable_testing()
libhevc_add_gtest_executable(
  ihevc_luma_inter_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_luma_inter_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_chroma_inter_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_chroma_inter_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_luma_intra_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_luma_intra_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_chroma_intra_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_chroma_intra_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_itrans_res_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_itrans_res_test.cc
)

libhevc_add_gtest_executable(
  ihevc_itrans_recon_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_itrans_recon_test.cc
)

libhevc_add_gtest_executable(
  ihevc_chroma_itrans_recon_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_chroma_itrans_recon_test.cc
)

libhevc_add_gtest_executable(
  ihevc_itrans_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_itrans_test.cc
)

libhevc_add_gtest_executable(
  ihevc_recon_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_recon_test.cc
)

libhevc_add_gtest_executable(
  ihevc_chroma_recon_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_chroma_recon_test.cc
)

libhevc_add_gtest_executable(
  ihevc_weighted_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_weighted_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_mem_fns_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_mem_fns_test.cc
)

libhevc_add_gtest_executable(
  ihevc_padding_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_padding_test.cc
)

libhevc_add_gtest_executable(
  ihevc_sao_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_sao_test.cc
)

libhevc_add_gtest_executable(
  ihevc_deblk_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_deblk_test.cc
)

include(GoogleTest)
