libhevc_add_executable(hevcdec libhevcdec SOURCES
                       ${HEVC_ROOT}/examples/decoder/main.c)
target_compile_definitions(hevcdec
                       PRIVATE PROFILE_ENABLE MD5_DISABLE)
