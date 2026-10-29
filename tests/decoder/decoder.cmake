enable_testing()
include(GoogleTest)

function(libhevc_add_decoder_gtest_executable NAME)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(ARG "" "" "${multi_value_args}" ${ARGN})

  libhevc_add_executable(
    ${NAME} libhevcdec
    SOURCES ${HEVC_ROOT}/tests/decoder/dec_func_selector.cc
            ${HEVC_ROOT}/tests/common/TestCommon.cc ${ARG_SOURCES}
    LIBS GTest::gtest_main)
  target_include_directories(${NAME} PRIVATE
    ${HEVC_ROOT}/tests/decoder
    ${HEVC_ROOT}/tests/common
  )
  gtest_discover_tests(${NAME} ${HEVC_GTEST_DISCOVERY_MODE})
endfunction()

libhevc_add_decoder_gtest_executable(
  ihevcd_itrans_recon_dc_test
  SOURCES ${HEVC_ROOT}/tests/decoder/ihevcd_itrans_recon_dc_test.cc
)
