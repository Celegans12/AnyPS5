add_library(libSceSisrSys SHARED EXCLUDE_FROM_ALL Export.cpp)
target_include_directories(libSceSisrSys PRIVATE ${LIBS_INCLUDE_DIR})
configure_windows_unwind(libSceSisrSys)
