enable_testing()
add_executable(
  ihevc_luma_inter_pred_test
  ${HEVC_ROOT}/tests/common/func_selector.cc
  ${HEVC_ROOT}/tests/common/tests_common.cc
  ${HEVC_ROOT}/tests/common/ihevc_luma_inter_pred_test.cc
)
target_link_libraries(
  ihevc_luma_inter_pred_test
  libhevcdec
  GTest::gtest_main
)
if(DEFINED SANITIZE)
    set_target_properties(ihevc_luma_inter_pred_test PROPERTIES LINK_FLAGS
                                            -fsanitize=${SANITIZE})
endif()

include(GoogleTest)
