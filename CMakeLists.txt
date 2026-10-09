cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(batch_matmul_max_sum_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NPU_ARCH "dav-3510" CACHE STRING "Target NPU architecture")

add_executable(batch_matmul_max_sum_custom main.asc)
add_executable(cube_test cube_test.asc)

foreach(target IN ITEMS batch_matmul_max_sum_custom cube_test)
    target_link_libraries(${target} PRIVATE
        tiling_api
        register
        platform
        unified_dlog
        dl
        m
        graph_base
    )

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
    )
endforeach()
