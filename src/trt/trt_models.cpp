#ifdef SPLG_HAS_TENSORRT

#include "splg/trt_models.hpp"

#include "splg/superpoint_post.hpp"

#include <stdexcept>

namespace splg {

TrtSuperPoint::TrtSuperPoint(const std::string &engine_path, SuperPointConfig cfg)
    : engine_(engine_path), cfg_(cfg) {}

void TrtSuperPoint::infer_dense(const float *image, int height, int width, std::vector<float> &logits,
                                std::vector<float> &desc, int &h, int &w) {
    engine_.set_input_shape("image", {1, 1, height, width});
    auto sh = engine_.tensor_shape("score_logits");
    h = static_cast<int>(sh.size() > 2 ? sh[2] : height / 8);
    w = static_cast<int>(sh.size() > 3 ? sh[3] : width / 8);
    logits.resize(static_cast<size_t>(65 * h * w));
    desc.resize(static_cast<size_t>(256 * h * w));
    engine_.infer({{"image", image}}, {{"score_logits", logits.data()}, {"descriptors", desc.data()}});
}

Features TrtSuperPoint::extract(const float *image, int height, int width) {
    engine_.set_input_shape("image", {1, 1, height, width});
    auto sh = engine_.tensor_shape("score_logits");
    const int h = static_cast<int>(sh.size() > 2 ? sh[2] : height / 8);
    const int w = static_cast<int>(sh.size() > 3 ? sh[3] : width / 8);
#ifdef SPLG_HAS_CUDA
    engine_.enqueue({{"image", image}});
    return superpoint_postprocess_cuda(static_cast<const float *>(engine_.gpu_ptr("score_logits")),
                                       static_cast<const float *>(engine_.gpu_ptr("descriptors")), h, w, cfg_,
                                       engine_.stream());
#else
    std::vector<float> logits, desc;
    logits.resize(static_cast<size_t>(65 * h * w));
    desc.resize(static_cast<size_t>(256 * h * w));
    engine_.infer({{"image", image}}, {{"score_logits", logits.data()}, {"descriptors", desc.data()}});
    return superpoint_postprocess(logits.data(), desc.data(), h, w, cfg_);
#endif
}

TrtLightGlue::TrtLightGlue(const std::string &engine_path) : engine_(engine_path) {}

Matches TrtLightGlue::match(const Features &f0, const Features &f1, float width0, float height0,
                            float width1, float height1) {
    Matches m;
    m.matches0.assign(static_cast<size_t>(f0.n), -1);
    m.scores0.assign(static_cast<size_t>(f0.n), 0.0f);
    if (f0.n == 0 || f1.n == 0) {
        return m;
    }
    engine_.set_input_shape("kpts0", {1, f0.n, 2});
    engine_.set_input_shape("kpts1", {1, f1.n, 2});
    engine_.set_input_shape("desc0", {1, f0.n, f0.desc_dim});
    engine_.set_input_shape("desc1", {1, f1.n, f1.desc_dim});
    engine_.set_input_shape("image_size0", {1, 2});
    engine_.set_input_shape("image_size1", {1, 2});

    std::vector<float> s0 = {width0, height0};
    std::vector<float> s1 = {width1, height1};
    std::vector<float> mscores(static_cast<size_t>(f0.n), 0.0f);
    engine_.infer({{"kpts0", f0.keypoints.data()},
                   {"kpts1", f1.keypoints.data()},
                   {"desc0", f0.descriptors.data()},
                   {"desc1", f1.descriptors.data()},
                   {"image_size0", s0.data()},
                   {"image_size1", s1.data()}},
                  {{"mscores0", mscores.data()}}, {{"matches0", m.matches0.data()}});
    m.scores0 = std::move(mscores);
    return m;
}

}  // namespace splg

#endif
