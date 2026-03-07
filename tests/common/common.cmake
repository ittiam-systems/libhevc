enable_testing()
libhevc_add_gtest_executable(
  ihevc_luma_inter_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_luma_inter_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_luma_intra_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_luma_intra_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_itrans_res_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_itrans_res_test.cc
)

libhevc_add_gtest_executable(
  ihevc_itrans_recon_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_itrans_recon_test.cc
)

include(GoogleTest)
