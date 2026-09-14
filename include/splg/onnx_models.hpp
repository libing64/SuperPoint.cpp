#pragma once

#include "splg/onnx_session.hpp"
#include "splg/types.hpp"

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
    explicit OnnxSuperPoint(const std::string &onnx_path, SuperPointConfig cfg = {});

    DenseMaps infer_dense(const float *image, int height, int width);
    Features extract(const float *image, int height, int width);

    const SuperPointConfig &config() const { return cfg_; }

 private:
    OnnxSession sess_;
    SuperPointConfig cfg_;
};

class OnnxLightGlue {
 public:
    explicit OnnxLightGlue(const std::string &onnx_path);

    Matches match(const Features &f0, const Features &f1, float width0, float height0, float width1,
                  float height1);

 private:
    OnnxSession sess_;
};

}  // namespace splg
