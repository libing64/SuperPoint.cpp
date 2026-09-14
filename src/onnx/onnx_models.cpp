#include "splg/onnx_models.hpp"

#include "splg/superpoint_post.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef SPLG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace splg {
namespace {

#ifdef SPLG_HAS_CUDA
void ensure_dev(void **ptr, size_t *cap, size_t bytes) {
    if (*ptr != nullptr && *cap >= bytes) {
        return;
    }
    if (*ptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
    }
    if (cudaMalloc(ptr, bytes > 0 ? bytes : 4) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    *cap = bytes;
}

void h2d(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy H2D failed");
    }
}

void d2h(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy D2H failed");
    }
}
#endif

}  // namespace

OnnxSuperPoint::OnnxSuperPoint(const std::string &onnx_path, SuperPointConfig cfg, bool use_cuda)
    : sess_(onnx_path, 0, use_cuda), cfg_(cfg) {}

OnnxSuperPoint::~OnnxSuperPoint() {
#ifdef SPLG_HAS_CUDA
    cudaFree(d_image_);
    cudaFree(d_logits_);
    cudaFree(d_desc_);
#endif
}

DenseMaps OnnxSuperPoint::infer_dense(const float *image, int height, int width) {
#ifdef SPLG_HAS_CUDA
    if (sess_.use_cuda()) {
        const int h = height / 8;
        const int w = width / 8;
        const size_t n_img = static_cast<size_t>(height) * static_cast<size_t>(width);
        const size_t n_logits = static_cast<size_t>(65 * h * w);
        const size_t n_desc = static_cast<size_t>(256 * h * w);
        ensure_dev(&d_image_, &cap_image_, n_img * 4);
        ensure_dev(&d_logits_, &cap_logits_, n_logits * 4);
        ensure_dev(&d_desc_, &cap_desc_, n_desc * 4);
        h2d(d_image_, image, n_img * 4);
        std::vector<Ort::Value> ins;
        std::vector<Ort::Value> outs;
        ins.push_back(sess_.tensor_device(d_image_, n_img, {1, 1, height, width}));
        outs.push_back(sess_.tensor_device(d_logits_, n_logits, {1, 65, h, w}));
        outs.push_back(sess_.tensor_device(d_desc_, n_desc, {1, 256, h, w}));
        sess_.run_io(ins, outs);
        DenseMaps maps;
        maps.h = h;
        maps.w = w;
        maps.score_logits.resize(n_logits);
        maps.descriptors.resize(n_desc);
        d2h(maps.score_logits.data(), d_logits_, n_logits * 4);
        d2h(maps.descriptors.data(), d_desc_, n_desc * 4);
        return maps;
    }
#endif
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
#ifdef SPLG_HAS_CUDA
    if (sess_.use_cuda()) {
        const int h = height / 8;
        const int w = width / 8;
        const size_t n_img = static_cast<size_t>(height) * static_cast<size_t>(width);
        const size_t n_logits = static_cast<size_t>(65 * h * w);
        const size_t n_desc = static_cast<size_t>(256 * h * w);
        ensure_dev(&d_image_, &cap_image_, n_img * 4);
        ensure_dev(&d_logits_, &cap_logits_, n_logits * 4);
        ensure_dev(&d_desc_, &cap_desc_, n_desc * 4);
        h2d(d_image_, image, n_img * 4);
        std::vector<Ort::Value> ins;
        std::vector<Ort::Value> outs;
        ins.push_back(sess_.tensor_device(d_image_, n_img, {1, 1, height, width}));
        outs.push_back(sess_.tensor_device(d_logits_, n_logits, {1, 65, h, w}));
        outs.push_back(sess_.tensor_device(d_desc_, n_desc, {1, 256, h, w}));
        sess_.run_io(ins, outs);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize after SuperPoint failed");
        }
        return superpoint_postprocess_cuda(static_cast<const float *>(d_logits_),
                                           static_cast<const float *>(d_desc_), h, w, cfg_, nullptr);
    }
#endif
    DenseMaps maps = infer_dense(image, height, width);
    return superpoint_postprocess(maps.score_logits.data(), maps.descriptors.data(), maps.h, maps.w, cfg_);
}

OnnxLightGlue::OnnxLightGlue(const std::string &onnx_path, bool use_cuda) : sess_(onnx_path, 0, use_cuda) {}

OnnxLightGlue::~OnnxLightGlue() {
#ifdef SPLG_HAS_CUDA
    cudaFree(d_k0_);
    cudaFree(d_k1_);
    cudaFree(d_d0_);
    cudaFree(d_d1_);
    cudaFree(d_s0_);
    cudaFree(d_s1_);
    cudaFree(d_m0_);
    cudaFree(d_ms_);
#endif
}

Matches OnnxLightGlue::match(const Features &f0, const Features &f1, float width0, float height0,
                             float width1, float height1) {
    Matches m;
    m.matches0.assign(static_cast<size_t>(f0.n), -1);
    m.scores0.assign(static_cast<size_t>(f0.n), 0.0f);
    if (f0.n == 0 || f1.n == 0) {
        return m;
    }

#ifdef SPLG_HAS_CUDA
    if (sess_.use_cuda()) {
        const size_t n0 = static_cast<size_t>(f0.n);
        const size_t n1 = static_cast<size_t>(f1.n);
        const size_t dim = static_cast<size_t>(f0.desc_dim);
        ensure_dev(&d_k0_, &cap_k0_, n0 * 2 * 4);
        ensure_dev(&d_k1_, &cap_k1_, n1 * 2 * 4);
        ensure_dev(&d_d0_, &cap_d0_, n0 * dim * 4);
        ensure_dev(&d_d1_, &cap_d1_, n1 * dim * 4);
        ensure_dev(&d_s0_, &cap_s0_, 8);
        ensure_dev(&d_s1_, &cap_s1_, 8);
        ensure_dev(&d_m0_, &cap_m0_, n0 * 8);
        ensure_dev(&d_ms_, &cap_ms_, n0 * 4);
        h2d(d_k0_, f0.keypoints.data(), n0 * 2 * 4);
        h2d(d_k1_, f1.keypoints.data(), n1 * 2 * 4);
        h2d(d_d0_, f0.descriptors.data(), n0 * dim * 4);
        h2d(d_d1_, f1.descriptors.data(), n1 * dim * 4);
        const float s0[2] = {width0, height0};
        const float s1[2] = {width1, height1};
        h2d(d_s0_, s0, 8);
        h2d(d_s1_, s1, 8);

        std::vector<Ort::Value> ins;
        std::vector<Ort::Value> outs;
        ins.push_back(sess_.tensor_device(d_k0_, n0 * 2, {1, f0.n, 2}));
        ins.push_back(sess_.tensor_device(d_k1_, n1 * 2, {1, f1.n, 2}));
        ins.push_back(sess_.tensor_device(d_d0_, n0 * dim, {1, f0.n, f0.desc_dim}));
        ins.push_back(sess_.tensor_device(d_d1_, n1 * dim, {1, f1.n, f1.desc_dim}));
        ins.push_back(sess_.tensor_device(d_s0_, 2, {1, 2}));
        ins.push_back(sess_.tensor_device(d_s1_, 2, {1, 2}));
        outs.push_back(sess_.tensor_device(d_m0_, n0, {1, f0.n}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        outs.push_back(sess_.tensor_device(d_ms_, n0, {1, f0.n}));
        sess_.run_io(ins, outs);
        d2h(m.matches0.data(), d_m0_, n0 * 8);
        d2h(m.scores0.data(), d_ms_, n0 * 4);
        return m;
    }
#endif

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
    const size_t n0 = static_cast<size_t>(f0.n);
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
