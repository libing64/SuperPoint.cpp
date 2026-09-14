#include "splg/onnx_models.hpp"

#include "splg/superpoint_post.hpp"

#include <algorithm>
#include <stdexcept>

namespace splg {

OnnxSuperPoint::OnnxSuperPoint(const std::string &onnx_path, SuperPointConfig cfg, bool use_cuda)
    : sess_(onnx_path, 0, use_cuda), cfg_(cfg) {}

DenseMaps OnnxSuperPoint::infer_dense(const float *image, int height, int width) {
    std::vector<float> input(static_cast<size_t>(height * width));
    std::copy(image, image + height * width, input.begin());
    auto tin = sess_.tensor_f32(input, {1, 1, height, width});
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(tin));
    auto outs = sess_.run(inputs);
    if (outs.size() < 2) {
        throw std::runtime_error("SuperPoint ONNX must output score_logits and descriptors");
    }
    DenseMaps maps;
    const auto info0 = outs[0].GetTensorTypeAndShapeInfo();
    const auto sh0 = info0.GetShape();
    if (sh0.size() != 4) {
        throw std::runtime_error("score_logits rank != 4");
    }
    maps.h = static_cast<int>(sh0[2]);
    maps.w = static_cast<int>(sh0[3]);
    const size_t n_logits = static_cast<size_t>(1 * 65 * maps.h * maps.w);
    const size_t n_desc = static_cast<size_t>(1 * 256 * maps.h * maps.w);
    const float *lptr = outs[0].GetTensorData<float>();
    const float *dptr = outs[1].GetTensorData<float>();
    maps.score_logits.assign(lptr, lptr + n_logits);
    maps.descriptors.assign(dptr, dptr + n_desc);
    return maps;
}

Features OnnxSuperPoint::extract(const float *image, int height, int width) {
    DenseMaps maps = infer_dense(image, height, width);
    return superpoint_postprocess(maps.score_logits.data(), maps.descriptors.data(), maps.h, maps.w, cfg_);
}

OnnxLightGlue::OnnxLightGlue(const std::string &onnx_path, bool use_cuda) : sess_(onnx_path, 0, use_cuda) {}

Matches OnnxLightGlue::match(const Features &f0, const Features &f1, float width0, float height0,
                             float width1, float height1) {
    if (f0.n == 0 || f1.n == 0) {
        Matches empty;
        empty.matches0.assign(static_cast<size_t>(f0.n), -1);
        empty.scores0.assign(static_cast<size_t>(f0.n), 0.0f);
        return empty;
    }
    std::vector<float> k0 = f0.keypoints;
    std::vector<float> k1 = f1.keypoints;
    std::vector<float> d0 = f0.descriptors;
    std::vector<float> d1 = f1.descriptors;
    std::vector<float> s0 = {width0, height0};
    std::vector<float> s1 = {width1, height1};

    std::vector<Ort::Value> inputs;
    inputs.push_back(sess_.tensor_f32(k0, {1, f0.n, 2}));
    inputs.push_back(sess_.tensor_f32(k1, {1, f1.n, 2}));
    inputs.push_back(sess_.tensor_f32(d0, {1, f0.n, f0.desc_dim}));
    inputs.push_back(sess_.tensor_f32(d1, {1, f1.n, f1.desc_dim}));
    inputs.push_back(sess_.tensor_f32(s0, {1, 2}));
    inputs.push_back(sess_.tensor_f32(s1, {1, 2}));

    auto outs = sess_.run(inputs);
    Matches m;
    const size_t n0 = static_cast<size_t>(f0.n);
    m.matches0.resize(n0, -1);
    m.scores0.resize(n0, 0.0f);

    auto shape0 = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = 1;
    for (auto d : shape0) {
        count *= static_cast<size_t>(d < 0 ? 0 : d);
    }
    if (count == 0) {
        count = n0;
    }
    const ONNXTensorElementDataType t0 = outs[0].GetTensorTypeAndShapeInfo().GetElementType();
    if (t0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t *p = outs[0].GetTensorData<int64_t>();
        for (size_t i = 0; i < n0 && i < count; ++i) {
            m.matches0[i] = p[i];
        }
    } else {
        const float *p = outs[0].GetTensorData<float>();
        for (size_t i = 0; i < n0 && i < count; ++i) {
            m.matches0[i] = static_cast<int64_t>(p[i]);
        }
    }
    if (outs.size() > 1) {
        const float *s = outs[1].GetTensorData<float>();
        auto shs = outs[1].GetTensorTypeAndShapeInfo().GetShape();
        size_t sc = 1;
        for (auto d : shs) {
            sc *= static_cast<size_t>(d < 0 ? 0 : d);
        }
        if (sc == 0) {
            sc = n0;
        }
        for (size_t i = 0; i < n0 && i < sc; ++i) {
            m.scores0[i] = s[i];
        }
    }
    return m;
}

}  // namespace splg
