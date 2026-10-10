add_library(libScePad SHARED EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PadState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PadScript.cpp
)
target_include_directories(libScePad PRIVATE ${LIBS_INCLUDE_DIR})
set_target_properties(libScePad PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(libScePad)

if(BUILD_TESTING)
    add_test_executable(guest_pad_tests ${LIBS_INCLUDE_DIR}/tests/GuestPad.cpp)
    target_include_directories(guest_pad_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/include)
    target_link_libraries(guest_pad_tests PRIVATE libScePad libc)
    configure_windows_unwind(guest_pad_tests)
    add_test(NAME guest_pad COMMAND guest_pad_tests)
    if(WIN32)
        set_tests_properties(guest_pad PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libScePad>")
    endif()

    add_test_executable(guest_pad_script_tests ${LIBS_INCLUDE_DIR}/tests/GuestPadScript.cpp)
    target_include_directories(guest_pad_script_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(guest_pad_script_tests PRIVATE libScePad libc)
    configure_windows_unwind(guest_pad_script_tests)
    add_test(NAME guest_pad_script COMMAND guest_pad_script_tests)
    add_test(NAME guest_pad_script_env COMMAND guest_pad_script_tests env)
    add_test(NAME guest_pad_script_invalid_env COMMAND guest_pad_script_tests invalid-env)
    set_tests_properties(guest_pad_script PROPERTIES ENVIRONMENT "APS5_PAD_SCRIPT=;APS5_PAD_SCRIPT_REPEAT=;APS5_TRACE_PAD=")
    set_tests_properties(guest_pad_script_env PROPERTIES ENVIRONMENT "APS5_PAD_SCRIPT=0:cross+up:1;APS5_PAD_SCRIPT_REPEAT=;APS5_TRACE_PAD=1")
    set_tests_properties(guest_pad_script_invalid_env PROPERTIES ENVIRONMENT "APS5_PAD_SCRIPT=150:bogus:200;APS5_PAD_SCRIPT_REPEAT=;APS5_TRACE_PAD=")
    if(WIN32)
        set_tests_properties(guest_pad_script guest_pad_script_env guest_pad_script_invalid_env PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libScePad>")
    endif()
endif()
