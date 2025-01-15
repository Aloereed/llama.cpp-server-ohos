# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5.0)
project(lm)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include
                    ${NATIVERENDER_ROOT_PATH}/openmp/${OHOS_ARCH}/include
                    ${NATIVERENDER_ROOT_PATH}/llama/${OHOS_ARCH}/include
                    ${CMAKE_CURRENT_BINARY_DIR})
add_library(openmp SHARED IMPORTED)
set_target_properties(openmp
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/openmp/${OHOS_ARCH}/lib/libomp.so)
add_library(llama SHARED IMPORTED)
set_target_properties(llama
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/llama/${OHOS_ARCH}/lib/libllama.so)
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/llama/${OHOS_ARCH}/lib/libggml.so)
add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/llama/${OHOS_ARCH}/lib/libggml-base.so)
add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/llama/${OHOS_ARCH}/lib/libggml-cpu.so)
add_library(llava_shared SHARED IMPORTED)
set_target_properties(llava_shared
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/llama/${OHOS_ARCH}/lib/libllava_shared.so)
option(LLAMA_SERVER_SSL "Build SSL support for the server" OFF)


set(TARGET_SRCS
    utils.hpp
    httplib.h
)
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)
# 打印${CMAKE_CURRENT_BINARY_DIR}
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "E:\\R\\lm\\entry\\src\\main\\cpp\\llama.cpp/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()
add_library(llama_common STATIC IMPORTED)
set_target_properties(llama_common PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/llama/${OHOS_ARCH}/lib/libcommon.a)
add_library(entry SHARED napi_init.cpp ${TARGET_SRCS})
target_link_libraries(entry PUBLIC libace_napi.z.so openmp llama llama_common ggml ggml-base ggml-cpu llava_shared libhilog_ndk.z.so ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(entry PRIVATE cxx_std_17)








