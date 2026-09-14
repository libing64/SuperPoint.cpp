#pragma once

#ifdef SPLG_HAS_TENSORRT

#include "splg/types.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace splg {

class TrtLogger : public nvinfer1::ILogger {
 public:
    void log(Severity severity, const char *msg) noexcept override;
};

class TrtEngine {
 public:
    explicit TrtEngine(const std::string &engine_path);
    ~TrtEngine();

    TrtEngine(const TrtEngine &) = delete;
    TrtEngine &operator=(const TrtEngine &) = delete;

    void set_input_shape(const std::string &name, const std::vector<int32_t> &dims);
    void infer(const std::unordered_map<std::string, const float *> &inputs,
               const std::unordered_map<std::string, float *> &outputs_f32,
               const std::unordered_map<std::string, int64_t *> &outputs_i64 = {});

    std::vector<int64_t> tensor_shape(const std::string &name) const;
    nvinfer1::DataType tensor_dtype(const std::string &name) const;
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;

 private:
    void allocate();
    void *device_ptr(const std::string &name);

    TrtLogger logger_;
    nvinfer1::IRuntime *runtime_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    nvinfer1::IExecutionContext *context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::unordered_map<std::string, void *> device_;
    std::unordered_map<std::string, size_t> bytes_;
};

}  // namespace splg

#endif
