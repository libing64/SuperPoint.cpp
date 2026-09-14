#include "splg/onnx_session.hpp"

#include <stdexcept>

namespace splg {

OnnxSession::OnnxSession(const std::string &model_path, int threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "splg"),
      mem_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts_.SetIntraOpNumThreads(threads > 0 ? threads : 1);
    opts_.SetInterOpNumThreads(1);
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
}

Ort::Value OnnxSession::tensor_f32(std::vector<float> &data, const std::vector<int64_t> &shape) {
    return Ort::Value::CreateTensor<float>(mem_, data.data(), data.size(), shape.data(), shape.size());
}

std::vector<Ort::Value> OnnxSession::run(const std::vector<Ort::Value> &inputs) {
    return session_->Run(Ort::RunOptions{nullptr}, input_ptrs_.data(), inputs.data(), inputs.size(),
                         output_ptrs_.data(), output_ptrs_.size());
}

}  // namespace splg
