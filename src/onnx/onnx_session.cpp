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
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda_opts);
            use_cuda_ = true;
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

std::vector<Ort::Value> OnnxSession::run(const std::vector<Ort::Value> &inputs) {
    return session_->Run(Ort::RunOptions{nullptr}, input_ptrs_.data(), inputs.data(), inputs.size(),
                         output_ptrs_.data(), output_ptrs_.size());
}

}  // namespace splg
