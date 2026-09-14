#pragma once

#ifdef SPLG_HAS_TENSORRT

#include "splg/trt_engine.hpp"
#include "splg/types.hpp"

#include <string>
#include <vector>

namespace splg {

struct DenseMaps;

class TrtSuperPoint {
 public:
    explicit TrtSuperPoint(const std::string &engine_path, SuperPointConfig cfg = {});

    void infer_dense(const float *image, int height, int width, std::vector<float> &logits,
                     std::vector<float> &desc, int &h, int &w);
    Features extract(const float *image, int height, int width);

 private:
    TrtEngine engine_;
    SuperPointConfig cfg_;
};

class TrtLightGlue {
 public:
    explicit TrtLightGlue(const std::string &engine_path);
    Matches match(const Features &f0, const Features &f1, float width0, float height0, float width1,
                  float height1);

 private:
    TrtEngine engine_;
};

}  // namespace splg

#endif
