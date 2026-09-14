#ifdef SPLG_HAS_TENSORRT

#include "splg/trt_engine.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace splg {
namespace {

#define SPLG_CUDA(expr)                                                                      \
    do {                                                                                     \
        cudaError_t _e = (expr);                                                             \
        if (_e != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e));        \
        }                                                                                    \
    } while (0)

size_t elem_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kINT32:
            return 4;
        default:
            return 4;
    }
}

int64_t volume(const nvinfer1::Dims &d) {
    int64_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        n *= d.d[i] > 0 ? d.d[i] : 1;
    }
    return n;
}

}  // namespace

void TrtLogger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        fprintf(stderr, "[TRT] %s\n", msg);
    }
}

TrtEngine::TrtEngine(const std::string &engine_path) {
    std::ifstream in(engine_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open engine " + engine_path);
    }
    in.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    in.read(blob.data(), static_cast<std::streamsize>(size));

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        throw std::runtime_error("createInferRuntime failed");
    }
    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) {
        throw std::runtime_error("deserializeCudaEngine failed");
    }
    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("createExecutionContext failed");
    }
    SPLG_CUDA(cudaStreamCreate(&stream_));
}

TrtEngine::~TrtEngine() {
    for (auto &kv : device_) {
        if (kv.second) {
            cudaFree(kv.second);
        }
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    delete context_;
    delete engine_;
    delete runtime_;
}

std::vector<std::string> TrtEngine::input_names() const {
    std::vector<std::string> names;
    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            names.emplace_back(name);
        }
    }
    return names;
}

std::vector<std::string> TrtEngine::output_names() const {
    std::vector<std::string> names;
    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            names.emplace_back(name);
        }
    }
    return names;
}

void TrtEngine::set_input_shape(const std::string &name, const std::vector<int32_t> &dims) {
    nvinfer1::Dims d;
    d.nbDims = static_cast<int>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        d.d[i] = dims[i];
    }
    if (!context_->setInputShape(name.c_str(), d)) {
        throw std::runtime_error("setInputShape failed for " + name);
    }
}

nvinfer1::DataType TrtEngine::tensor_dtype(const std::string &name) const {
    return engine_->getTensorDataType(name.c_str());
}

std::vector<int64_t> TrtEngine::tensor_shape(const std::string &name) const {
    const nvinfer1::Dims d = context_->getTensorShape(name.c_str());
    std::vector<int64_t> s;
    for (int i = 0; i < d.nbDims; ++i) {
        s.push_back(d.d[i]);
    }
    return s;
}

void TrtEngine::allocate() {
    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char *name = engine_->getIOTensorName(i);
        const nvinfer1::Dims d = context_->getTensorShape(name);
        const size_t bytes = static_cast<size_t>(volume(d)) * elem_size(engine_->getTensorDataType(name));
        auto it = device_.find(name);
        if (it == device_.end() || bytes_[name] < bytes) {
            if (it != device_.end() && it->second) {
                cudaFree(it->second);
            }
            void *ptr = nullptr;
            SPLG_CUDA(cudaMalloc(&ptr, bytes > 0 ? bytes : 4));
            device_[name] = ptr;
            bytes_[name] = bytes;
        }
        if (!context_->setTensorAddress(name, device_[name])) {
            throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
        }
    }
}

void *TrtEngine::device_ptr(const std::string &name) {
    auto it = device_.find(name);
    if (it == device_.end()) {
        throw std::runtime_error("missing device buffer " + name);
    }
    return it->second;
}

void TrtEngine::infer(const std::unordered_map<std::string, const float *> &inputs,
                      const std::unordered_map<std::string, float *> &outputs_f32,
                      const std::unordered_map<std::string, int64_t *> &outputs_i64) {
    allocate();
    for (const auto &kv : inputs) {
        const nvinfer1::Dims d = context_->getTensorShape(kv.first.c_str());
        const size_t bytes = static_cast<size_t>(volume(d)) * sizeof(float);
        SPLG_CUDA(cudaMemcpyAsync(device_ptr(kv.first), kv.second, bytes, cudaMemcpyHostToDevice, stream_));
    }
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 failed");
    }
    for (const auto &kv : outputs_f32) {
        const nvinfer1::Dims d = context_->getTensorShape(kv.first.c_str());
        const size_t bytes = static_cast<size_t>(volume(d)) * sizeof(float);
        SPLG_CUDA(cudaMemcpyAsync(kv.second, device_ptr(kv.first), bytes, cudaMemcpyDeviceToHost, stream_));
    }
    for (const auto &kv : outputs_i64) {
        const nvinfer1::Dims d = context_->getTensorShape(kv.first.c_str());
        const auto dt = engine_->getTensorDataType(kv.first.c_str());
        const int64_t n = volume(d);
        if (dt == nvinfer1::DataType::kINT64) {
            SPLG_CUDA(cudaMemcpyAsync(kv.second, device_ptr(kv.first), static_cast<size_t>(n) * 8,
                                      cudaMemcpyDeviceToHost, stream_));
        } else if (dt == nvinfer1::DataType::kINT32) {
            std::vector<int32_t> tmp(static_cast<size_t>(n));
            SPLG_CUDA(cudaMemcpyAsync(tmp.data(), device_ptr(kv.first), static_cast<size_t>(n) * 4,
                                      cudaMemcpyDeviceToHost, stream_));
            SPLG_CUDA(cudaStreamSynchronize(stream_));
            for (int64_t i = 0; i < n; ++i) {
                kv.second[i] = tmp[static_cast<size_t>(i)];
            }
            continue;
        } else {
            std::vector<float> tmp(static_cast<size_t>(n));
            SPLG_CUDA(cudaMemcpyAsync(tmp.data(), device_ptr(kv.first), static_cast<size_t>(n) * 4,
                                      cudaMemcpyDeviceToHost, stream_));
            SPLG_CUDA(cudaStreamSynchronize(stream_));
            for (int64_t i = 0; i < n; ++i) {
                kv.second[i] = static_cast<int64_t>(tmp[static_cast<size_t>(i)]);
            }
            continue;
        }
    }
    SPLG_CUDA(cudaStreamSynchronize(stream_));
}

void TrtEngine::enqueue(const std::unordered_map<std::string, const float *> &inputs) {
    allocate();
    for (const auto &kv : inputs) {
        const nvinfer1::Dims d = context_->getTensorShape(kv.first.c_str());
        const size_t bytes = static_cast<size_t>(volume(d)) * sizeof(float);
        SPLG_CUDA(cudaMemcpyAsync(device_ptr(kv.first), kv.second, bytes, cudaMemcpyHostToDevice, stream_));
    }
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 failed");
    }
}

void *TrtEngine::gpu_ptr(const std::string &name) { return device_ptr(name); }

}  // namespace splg

#endif
