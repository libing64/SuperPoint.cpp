#pragma once

#include "splg/onnx_session.hpp"
#include "splg/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace splg {

struct DenseMaps {
    std::vector<float> score_logits;  // [1,65,h,w]
    std::vector<float> descriptors;   // [1,256,h,w]
    int h = 0;
    int w = 0;
};

class OnnxSuperPoint {
 public:
    explicit OnnxSuperPoint(const std::string &onnx_path, SuperPointConfig cfg = {}, bool use_cuda = true);
    ~OnnxSuperPoint();
    OnnxSuperPoint(const OnnxSuperPoint &) = delete;
    OnnxSuperPoint &operator=(const OnnxSuperPoint &) = delete;

    const char *device() const { return sess_.device(); }

    DenseMaps infer_dense(const float *image, int height, int width);
    Features extract(const float *image, int height, int width);

    const SuperPointConfig &config() const { return cfg_; }

 private:
    OnnxSession sess_;
    SuperPointConfig cfg_;
    void *d_image_ = nullptr;
    void *d_logits_ = nullptr;
    void *d_desc_ = nullptr;
    size_t cap_image_ = 0;
    size_t cap_logits_ = 0;
    size_t cap_desc_ = 0;
};

class OnnxLightGlue {
 public:
    explicit OnnxLightGlue(const std::string &onnx_path, bool use_cuda = true);
    ~OnnxLightGlue();
    OnnxLightGlue(const OnnxLightGlue &) = delete;
    OnnxLightGlue &operator=(const OnnxLightGlue &) = delete;

    const char *device() const { return sess_.device(); }

    Matches match(const Features &f0, const Features &f1, float width0, float height0, float width1,
                  float height1);

 private:
    OnnxSession sess_;
    void *d_k0_ = nullptr, *d_k1_ = nullptr, *d_d0_ = nullptr, *d_d1_ = nullptr;
    void *d_s0_ = nullptr, *d_s1_ = nullptr, *d_m0_ = nullptr, *d_ms_ = nullptr;
    size_t cap_k0_ = 0, cap_k1_ = 0, cap_d0_ = 0, cap_d1_ = 0;
    size_t cap_s0_ = 0, cap_s1_ = 0, cap_m0_ = 0, cap_ms_ = 0;
};

}  // namespace splg
