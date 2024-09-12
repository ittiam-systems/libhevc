if(NOT "${SYSTEM_NAME}" STREQUAL "Darwin")
    libhevc_add_fuzzer(hevc_dec_fuzzer libhevcdec SOURCES
                       ${HEVC_ROOT}/fuzzer/hevc_dec_fuzzer.cpp)
endif()