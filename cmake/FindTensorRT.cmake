# Locate TensorRT. Sets TensorRT_FOUND, TensorRT::NvInfer, TensorRT::NvOnnxParser.

set(TENSORRT_ROOT "" CACHE PATH "TensorRT install prefix")

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        ${TENSORRT_ROOT}/include
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/tensorrt/include
)

find_library(TensorRT_LIBRARY
    NAMES nvinfer
    HINTS
        ${TENSORRT_ROOT}/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/tensorrt/lib
)

find_library(TensorRT_ONNX_LIBRARY
    NAMES nvonnxparser
    HINTS
        ${TENSORRT_ROOT}/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/tensorrt/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT DEFAULT_MSG TensorRT_INCLUDE_DIR TensorRT_LIBRARY)

if(TensorRT_FOUND AND NOT TARGET TensorRT::NvInfer)
    add_library(TensorRT::NvInfer SHARED IMPORTED)
    set_target_properties(TensorRT::NvInfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
    if(TensorRT_ONNX_LIBRARY)
        add_library(TensorRT::NvOnnxParser SHARED IMPORTED)
        set_target_properties(TensorRT::NvOnnxParser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_ONNX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()
