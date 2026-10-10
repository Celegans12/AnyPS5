function(add_shader_cold_spirv_codec zlibRoot)
    if(TARGET shader_cold_spirv_codec)
        return()
    endif()

    add_library(shader_cold_spirv_zlib STATIC EXCLUDE_FROM_ALL
            "${zlibRoot}/adler32.c"
            "${zlibRoot}/compress.c"
            "${zlibRoot}/crc32.c"
            "${zlibRoot}/deflate.c"
            "${zlibRoot}/inffast.c"
            "${zlibRoot}/inflate.c"
            "${zlibRoot}/inftrees.c"
            "${zlibRoot}/trees.c"
            "${zlibRoot}/zutil.c"
    )
    target_include_directories(shader_cold_spirv_zlib SYSTEM PUBLIC "${zlibRoot}")
    target_compile_definitions(shader_cold_spirv_zlib PUBLIC
            Z_PREFIX
            z_errmsg=anyps5_shader_z_errmsg
            z_verbose=anyps5_shader_z_verbose
            z_error=anyps5_shader_z_error
    )
    set_target_properties(shader_cold_spirv_zlib PROPERTIES
            POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden C_STANDARD 99 C_EXTENSIONS OFF)

    add_library(shader_cold_spirv_codec STATIC EXCLUDE_FROM_ALL
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ColdSpirvCodec.cpp")
    target_compile_features(shader_cold_spirv_codec PUBLIC cxx_std_20)
    target_include_directories(shader_cold_spirv_codec PUBLIC "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    target_link_libraries(shader_cold_spirv_codec PRIVATE shader_cold_spirv_zlib)
    set_target_properties(shader_cold_spirv_codec PROPERTIES
            POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF CXX_VISIBILITY_PRESET hidden)

    if(BUILD_TESTING)
        find_package(Threads REQUIRED)
        add_executable(shader_cold_spirv_codec_tests
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tests/ColdSpirvCodec.cpp"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ColdSpirvCodec.cpp")
        target_compile_features(shader_cold_spirv_codec_tests PRIVATE cxx_std_20)
        target_compile_definitions(shader_cold_spirv_codec_tests PRIVATE ANYPS5_COLD_SPIRV_TESTING)
        target_include_directories(shader_cold_spirv_codec_tests PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
        target_link_libraries(shader_cold_spirv_codec_tests PRIVATE shader_cold_spirv_zlib Threads::Threads)
        set_target_properties(shader_cold_spirv_codec_tests PROPERTIES CXX_EXTENSIONS OFF)
        if(ANYPS5_TEST_OUTPUT_DIRECTORY)
            set_target_properties(shader_cold_spirv_codec_tests PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY "$<1:${ANYPS5_TEST_OUTPUT_DIRECTORY}>")
        endif()
        add_test(NAME shader_cold_spirv_codec COMMAND shader_cold_spirv_codec_tests)
        set_tests_properties(shader_cold_spirv_codec PROPERTIES TIMEOUT 60)

        add_executable(shader_cold_spirv_storage_tests
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tests/SharedSpirv.cpp"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/SharedSpirv.cpp"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ColdSpirvCodec.cpp")
        target_compile_features(shader_cold_spirv_storage_tests PRIVATE cxx_std_20)
        target_compile_definitions(shader_cold_spirv_storage_tests PRIVATE ANYPS5_COLD_SPIRV_TESTING)
        target_include_directories(shader_cold_spirv_storage_tests PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
        target_link_libraries(shader_cold_spirv_storage_tests PRIVATE shader_cold_spirv_zlib Threads::Threads)
        set_target_properties(shader_cold_spirv_storage_tests PROPERTIES CXX_EXTENSIONS OFF)
        if(ANYPS5_TEST_OUTPUT_DIRECTORY)
            set_target_properties(shader_cold_spirv_storage_tests PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY "$<1:${ANYPS5_TEST_OUTPUT_DIRECTORY}>")
        endif()
        add_test(NAME shader_cold_spirv_storage COMMAND shader_cold_spirv_storage_tests)
        set_tests_properties(shader_cold_spirv_storage PROPERTIES TIMEOUT 60)
    endif()
endfunction()
