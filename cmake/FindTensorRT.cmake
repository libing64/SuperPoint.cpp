# Locate a system TensorRT install. Optional TENSORRT_ROOT overrides the prefix.

set(TENSORRT_ROOT "" CACHE PATH "TensorRT install prefix (optional)")

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        ${TENSORRT_ROOT}/include
        ${TENSORRT_ROOT}/include/x86_64-linux-gnu
    PATHS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
)

find_library(TensorRT_LIBRARY
    NAMES nvinfer
    HINTS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib64
        ${TENSORRT_ROOT}/lib/x86_64-linux-gnu
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

find_library(TensorRT_ONNX_LIBRARY
    NAMES nvonnxparser
    HINTS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib64
        ${TENSORRT_ROOT}/lib/x86_64-linux-gnu
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT DEFAULT_MSG TensorRT_INCLUDE_DIR TensorRT_LIBRARY)

if(TensorRT_FOUND AND NOT TARGET TensorRT::NvInfer)
    add_library(TensorRT::NvInfer SHARED IMPORTED)
    get_filename_component(TensorRT_LIBRARY_DIR "${TensorRT_LIBRARY}" DIRECTORY)
    set_target_properties(TensorRT::NvInfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        INTERFACE_LINK_DIRECTORIES "${TensorRT_LIBRARY_DIR}"
    )
    if(TensorRT_ONNX_LIBRARY)
        add_library(TensorRT::NvOnnxParser SHARED IMPORTED)
        set_target_properties(TensorRT::NvOnnxParser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_ONNX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()
