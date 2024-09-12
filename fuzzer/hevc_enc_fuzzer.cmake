if(NOT "${SYSTEM_NAME}" STREQUAL "Darwin")
    libhevc_add_fuzzer(hevc_enc_fuzzer libhevcenc SOURCES
                       ${HEVC_ROOT}/fuzzer/hevc_enc_fuzzer.cpp)
endif()