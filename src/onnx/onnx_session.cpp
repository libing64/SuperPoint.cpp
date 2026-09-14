#include "splg/onnx_session.hpp"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace splg {
namespace {

bool g_cudnn_preloaded = false;

void try_dlopen(const std::string &path) {
    if (path.empty()) {
        return;
    }
    dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
}

void preload_cudnn9() {
    if (g_cudnn_preloaded) {
        return;
    }
    g_cudnn_preloaded = true;
    if (dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL) != nullptr) {
        return;
    }

    std::vector<std::string> roots;
    if (const char *env = std::getenv("CUDNN_HOME")) {
        roots.emplace_back(env);
    }
    if (const char *env = std::getenv("CONDA_PREFIX")) {
        roots.emplace_back(env);
    }
    if (const char *home = std::getenv("HOME")) {
        roots.emplace_back(std::string(home) + "/.conda/envs/deep_matching");
    }

    namespace fs = std::filesystem;
    for (const auto &root : roots) {
        const fs::path lib = fs::path(root) / "lib";
        if (!fs::is_directory(lib)) {
            continue;
        }
        for (const auto &py : fs::directory_iterator(lib)) {
            if (!py.is_directory()) {
                continue;
            }
            const fs::path cand = py.path() / "site-packages/nvidia/cudnn/lib/libcudnn.so.9";
            if (fs::exists(cand)) {
                try_dlopen(cand.string());
                if (dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL) != nullptr) {
                    return;
                }
            }
        }
        const fs::path direct = fs::path(root) / "lib" / "libcudnn.so.9";
        if (fs::exists(direct)) {
            try_dlopen(direct.string());
            if (dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL) != nullptr) {
                return;
            }
        }
    }
}

}  // namespace

OnnxSession::OnnxSession(const std::string &model_path, int threads, bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "splg"),
      mem_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
      use_cuda_(false) {
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts_.SetIntraOpNumThreads(threads > 0 ? threads : 1);
    opts_.SetInterOpNumThreads(1);
    opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    if (use_cuda) {
        preload_cudnn9();
        try {
            OrtCUDAProviderOptionsV2 *cuda_opts = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts));
            const char *keys[] = {"device_id", "use_tf32"};
            const char *vals[] = {"0", "0"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda_opts, keys, vals, 2));
            opts_.AppendExecutionProvider_CUDA_V2(*cuda_opts);
            Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);
            use_cuda_ = true;
            cuda_mem_ = std::make_unique<Ort::MemoryInfo>("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        } catch (const Ort::Exception &ex) {
            throw std::runtime_error(std::string("ONNX Runtime CUDA EP failed (need GPU ORT + cuDNN 9): ") +
                                     ex.what());
        }
    }

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts_);

    const size_t n_in = session_->GetInputCount();
    const size_t n_out = session_->GetOutputCount();
    for (size_t i = 0; i < n_in; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_.emplace_back(name.get());
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.emplace_back(name.get());
    }
    for (auto &s : input_names_) {
        input_ptrs_.push_back(s.c_str());
    }
    for (auto &s : output_names_) {
        output_ptrs_.push_back(s.c_str());
    }
    std::fprintf(stderr, "[ORT] device=%s  model=%s\n", use_cuda_ ? "cuda" : "cpu", model_path.c_str());
}

Ort::Value OnnxSession::tensor_f32(std::vector<float> &data, const std::vector<int64_t> &shape) {
    return Ort::Value::CreateTensor<float>(mem_, data.data(), data.size(), shape.data(), shape.size());
}

Ort::Value OnnxSession::tensor_device(void *data, size_t n_elem, const std::vector<int64_t> &shape,
                                      ONNXTensorElementDataType dtype) {
    if (!cuda_mem_) {
        throw std::runtime_error("CUDA memory info not initialized");
    }
    return Ort::Value::CreateTensor(*cuda_mem_, data, n_elem * (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4),
                                    shape.data(), shape.size(), dtype);
}

void OnnxSession::run_io(std::vector<Ort::Value> &inputs, std::vector<Ort::Value> &outputs) {
    Ort::IoBinding bind(*session_);
    for (size_t i = 0; i < inputs.size() && i < input_names_.size(); ++i) {
        bind.BindInput(input_names_[i].c_str(), inputs[i]);
    }
    for (size_t i = 0; i < outputs.size() && i < output_names_.size(); ++i) {
        bind.BindOutput(output_names_[i].c_str(), outputs[i]);
    }
    session_->Run(Ort::RunOptions{nullptr}, bind);
}

std::vector<Ort::Value> OnnxSession::run(const std::vector<Ort::Value> &inputs) {
    return session_->Run(Ort::RunOptions{nullptr}, input_ptrs_.data(), inputs.data(), inputs.size(),
                         output_ptrs_.data(), output_ptrs_.size());
}

}  // namespace splg
