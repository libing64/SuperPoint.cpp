# Locate ONNX Runtime. Sets onnxruntime::onnxruntime.

set(onnxruntime_ROOT "" CACHE PATH "ONNX Runtime install prefix")

if(NOT onnxruntime_ROOT AND EXISTS "/home/libing/source/mapping/colmap/build/_deps/onnxruntime-src/include/onnxruntime_cxx_api.h")
    set(onnxruntime_ROOT "/home/libing/source/mapping/colmap/build/_deps/onnxruntime-src")
endif()

find_path(onnxruntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${onnxruntime_ROOT}/include
        ${onnxruntime_ROOT}/include/onnxruntime
        ENV onnxruntime_ROOT
)

find_library(onnxruntime_LIBRARY
    NAMES onnxruntime libonnxruntime
    HINTS
        ${onnxruntime_ROOT}/lib
        ${onnxruntime_ROOT}/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_INCLUDE_DIR onnxruntime_LIBRARY)

if(onnxruntime_FOUND AND NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}"
    )
endif()
