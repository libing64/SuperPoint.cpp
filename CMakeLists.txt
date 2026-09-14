cmake_minimum_required(VERSION 3.18)
project(splg_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(SPLG_FETCH_ONNXRUNTIME "Download ONNX Runtime if not found" ON)

find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND AND SPLG_FETCH_ONNXRUNTIME)
    include(FetchContent)
    set(ORT_VERSION "1.24.4")
    FetchContent_Declare(onnxruntime_prebuilt
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
    )
    FetchContent_GetProperties(onnxruntime_prebuilt)
    if(NOT onnxruntime_prebuilt_POPULATED)
        FetchContent_Populate(onnxruntime_prebuilt)
    endif()
    set(onnxruntime_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}" CACHE PATH "ONNX Runtime root" FORCE)
    find_package(onnxruntime REQUIRED)
endif()
if(NOT onnxruntime_FOUND)
    message(FATAL_ERROR "ONNX Runtime not found. Set onnxruntime_ROOT or enable SPLG_FETCH_ONNXRUNTIME.")
endif()

add_library(splg_common STATIC
    src/common/npy.cpp
    src/common/io.cpp
    src/common/superpoint_post.cpp
    src/common/lightglue_post.cpp
)
target_include_directories(splg_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(splg_onnx STATIC
    src/onnx/onnx_session.cpp
    src/onnx/onnx_models.cpp
)
target_include_directories(splg_onnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(splg_onnx PUBLIC splg_common onnxruntime::onnxruntime ${CMAKE_DL_LIBS})

add_executable(infer_onnx tools/infer_onnx.cpp)
target_link_libraries(infer_onnx PRIVATE splg_onnx)
add_executable(bench_onnx tools/bench_pairs.cpp)
target_link_libraries(bench_onnx PRIVATE splg_onnx)
if(onnxruntime_LIBRARY)
    get_filename_component(_ort_libdir "${onnxruntime_LIBRARY}" DIRECTORY)
    set_target_properties(infer_onnx PROPERTIES BUILD_RPATH "${_ort_libdir}" INSTALL_RPATH "${_ort_libdir}")
    set_target_properties(bench_onnx PROPERTIES BUILD_RPATH "${_ort_libdir}" INSTALL_RPATH "${_ort_libdir}")
endif()

find_package(CUDAToolkit QUIET)
find_package(TensorRT QUIET)
if(TensorRT_FOUND AND CUDAToolkit_FOUND)
    message(STATUS "TensorRT enabled")
    add_library(splg_trt STATIC
        src/trt/trt_engine.cpp
        src/trt/trt_models.cpp
    )
    target_include_directories(splg_trt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(splg_trt PUBLIC SPLG_HAS_TENSORRT)
    target_link_libraries(splg_trt PUBLIC splg_common TensorRT::NvInfer CUDA::cudart)
    add_executable(infer_trt tools/infer_trt.cpp)
    target_compile_definitions(infer_trt PRIVATE SPLG_HAS_TENSORRT)
    target_link_libraries(infer_trt PRIVATE splg_trt)
    add_executable(bench_trt tools/bench_pairs.cpp)
    target_compile_definitions(bench_trt PRIVATE SPLG_HAS_TENSORRT SPLG_BENCH_TRT)
    target_link_libraries(bench_trt PRIVATE splg_trt)
    get_filename_component(_trt_libdir "${TensorRT_LIBRARY}" DIRECTORY)
    set_target_properties(infer_trt PROPERTIES
        BUILD_RPATH "${_trt_libdir}"
        INSTALL_RPATH "${_trt_libdir}"
    )
    set_target_properties(bench_trt PROPERTIES
        BUILD_RPATH "${_trt_libdir}"
        INSTALL_RPATH "${_trt_libdir}"
    )
else()
    message(STATUS "TensorRT not found; infer_trt will be a stub")
    add_executable(infer_trt tools/infer_trt.cpp)
endif()
