#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace splg {

class OnnxSession {
 public:
    explicit OnnxSession(const std::string &model_path, int threads = 0, bool use_cuda = true);

    const char *device() const { return use_cuda_ ? "cuda" : "cpu"; }

    std::vector<Ort::Value> run(const std::vector<Ort::Value> &inputs);

    Ort::MemoryInfo &cpu_mem() { return mem_; }
    const std::vector<std::string> &input_names() const { return input_names_; }
    const std::vector<std::string> &output_names() const { return output_names_; }

    Ort::Value tensor_f32(std::vector<float> &data, const std::vector<int64_t> &shape);

 private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo mem_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char *> input_ptrs_;
    std::vector<const char *> output_ptrs_;
    bool use_cuda_ = false;
};

}  // namespace splg
