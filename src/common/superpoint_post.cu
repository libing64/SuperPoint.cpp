#include "splg/superpoint_post.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace splg {
namespace {

#define SPLG_CUDA_CHECK(expr)                                                                    \
    do {                                                                                         \
        cudaError_t _e = (expr);                                                                 \
        if (_e != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA post: ") + cudaGetErrorString(_e));       \
        }                                                                                        \
    } while (0)

void *dev_realloc(void *p, size_t *cap, size_t need) {
    if (need <= *cap && p != nullptr) {
        return p;
    }
    if (p) {
        cudaFree(p);
    }
    void *n = nullptr;
    SPLG_CUDA_CHECK(cudaMalloc(&n, need));
    *cap = need;
    return n;
}

struct Workspace {
    float *scores = nullptr, *pooled = nullptr, *max_mask = nullptr, *supp = nullptr, *supp_scores = nullptr;
    float *kxy = nullptr, *sc = nullptr, *desc = nullptr;
    size_t cap_scores = 0, cap_pooled = 0, cap_mask = 0, cap_supp = 0, cap_ss = 0;
    size_t cap_kxy = 0, cap_sc = 0, cap_desc = 0;

    void ensure_pix(size_t n) {
        const size_t b = n * sizeof(float);
        scores = static_cast<float *>(dev_realloc(scores, &cap_scores, b));
        pooled = static_cast<float *>(dev_realloc(pooled, &cap_pooled, b));
        max_mask = static_cast<float *>(dev_realloc(max_mask, &cap_mask, b));
        supp = static_cast<float *>(dev_realloc(supp, &cap_supp, b));
        supp_scores = static_cast<float *>(dev_realloc(supp_scores, &cap_ss, b));
    }

    void ensure_kpts(int n, int dim) {
        kxy = static_cast<float *>(dev_realloc(kxy, &cap_kxy, static_cast<size_t>(n) * 2 * sizeof(float)));
        sc = static_cast<float *>(dev_realloc(sc, &cap_sc, static_cast<size_t>(n) * sizeof(float)));
        desc = static_cast<float *>(
            dev_realloc(desc, &cap_desc, static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(float)));
    }

    ~Workspace() {
        cudaFree(scores);
        cudaFree(pooled);
        cudaFree(max_mask);
        cudaFree(supp);
        cudaFree(supp_scores);
        cudaFree(kxy);
        cudaFree(sc);
        cudaFree(desc);
    }
};

Workspace &ws() {
    static Workspace w;
    return w;
}

__global__ void k_softmax_scatter(const float *logits, float *scores, int h, int w, int H, int W) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (y >= h || x >= w) {
        return;
    }
    const int hw = h * w;
    const int base = y * w + x;
    float m = logits[base];
    for (int c = 1; c < 65; ++c) {
        m = fmaxf(m, logits[c * hw + base]);
    }
    float probs[65];
    float sum = 0.0f;
    for (int c = 0; c < 65; ++c) {
        probs[c] = expf(logits[c * hw + base] - m);
        sum += probs[c];
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            scores[(y * 8 + i) * W + (x * 8 + j)] = probs[i * 8 + j] / sum;
        }
    }
}

__global__ void k_max_pool(const float *in, float *out, int H, int W, int radius) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (y >= H || x >= W) {
        return;
    }
    float m = -1e30f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= H) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= W) {
                continue;
            }
            m = fmaxf(m, in[yy * W + xx]);
        }
    }
    out[y * W + x] = m;
}

__global__ void k_eq_mask(const float *scores, const float *pooled, float *mask, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        mask[i] = (scores[i] == pooled[i]) ? 1.0f : 0.0f;
    }
}

__global__ void k_suppress_scores(const float *supp, const float *scores, float *out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = (supp[i] > 0.0f) ? 0.0f : scores[i];
    }
}

__global__ void k_update_mask(const float *supp_scores, const float *pooled, const float *supp, float *mask,
                              int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const bool new_max = supp_scores[i] == pooled[i];
    const bool not_supp = !(supp[i] > 0.0f);
    if (new_max && not_supp) {
        mask[i] = 1.0f;
    }
}

__global__ void k_apply_mask(float *scores, const float *mask, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n && mask[i] == 0.0f) {
        scores[i] = 0.0f;
    }
}

__global__ void k_border(float *scores, int H, int W, int pad) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (y >= H || x >= W) {
        return;
    }
    if (y < pad || x < pad || y >= H - pad || x >= W - pad) {
        scores[y * W + x] = -1.0f;
    }
}

__device__ float desc_at(const float *desc, int h, int w, int c, int yy, int xx) {
    if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
        return 0.0f;
    }
    return desc[(c * h + yy) * w + xx];
}

__global__ void k_sample_desc(const float *kxy, const float *desc, float *out, int n, int h, int w, int dim) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float s = 8.0f;
    const float den_x = static_cast<float>(w) * s - s / 2.0f - 0.5f;
    const float den_y = static_cast<float>(h) * s - s / 2.0f - 0.5f;
    float x = kxy[i * 2 + 0] - s / 2.0f + 0.5f;
    float y = kxy[i * 2 + 1] - s / 2.0f + 0.5f;
    x = x / den_x * 2.0f - 1.0f;
    y = y / den_y * 2.0f - 1.0f;
    const float px = ((x + 1.0f) / 2.0f) * static_cast<float>(w - 1);
    const float py = ((y + 1.0f) / 2.0f) * static_cast<float>(h - 1);
    const int x0 = static_cast<int>(floorf(px));
    const int y0 = static_cast<int>(floorf(py));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float wx = px - static_cast<float>(x0);
    const float wy = py - static_cast<float>(y0);
    float norm = 0.0f;
    for (int c = 0; c < dim; ++c) {
        const float v = (1.0f - wy) * (1.0f - wx) * desc_at(desc, h, w, c, y0, x0) +
                        (1.0f - wy) * wx * desc_at(desc, h, w, c, y0, x1) +
                        wy * (1.0f - wx) * desc_at(desc, h, w, c, y1, x0) +
                        wy * wx * desc_at(desc, h, w, c, y1, x1);
        out[i * dim + c] = v;
        norm += v * v;
    }
    norm = sqrtf(fmaxf(norm, 1e-12f));
    for (int c = 0; c < dim; ++c) {
        out[i * dim + c] /= norm;
    }
}

dim3 grid2(int H, int W, int bx = 16, int by = 16) {
    return dim3((W + bx - 1) / bx, (H + by - 1) / by);
}

int grid1(int n, int b = 256) { return (n + b - 1) / b; }

}  // namespace

Features superpoint_postprocess_cuda(const float *logits_dev, const float *desc_dev, int h, int w,
                                     const SuperPointConfig &cfg, cudaStream_t stream) {
    if (h <= 0 || w <= 0) {
        throw std::runtime_error("invalid SuperPoint feature map size");
    }
    const int H = h * 8;
    const int W = w * 8;
    const int n_pix = H * W;
    SPLG_CUDA_CHECK(cudaSetDevice(0));
    auto &wk = ws();
    wk.ensure_pix(static_cast<size_t>(n_pix));

    const dim3 block(16, 16);
    k_softmax_scatter<<<grid2(h, w), block, 0, stream>>>(logits_dev, wk.scores, h, w, H, W);
    SPLG_CUDA_CHECK(cudaGetLastError());

    const int radius = cfg.nms_radius;
    k_max_pool<<<grid2(H, W), block, 0, stream>>>(wk.scores, wk.pooled, H, W, radius);
    k_eq_mask<<<grid1(n_pix), 256, 0, stream>>>(wk.scores, wk.pooled, wk.max_mask, n_pix);
    for (int iter = 0; iter < 2; ++iter) {
        k_max_pool<<<grid2(H, W), block, 0, stream>>>(wk.max_mask, wk.supp, H, W, radius);
        k_suppress_scores<<<grid1(n_pix), 256, 0, stream>>>(wk.supp, wk.scores, wk.supp_scores, n_pix);
        k_max_pool<<<grid2(H, W), block, 0, stream>>>(wk.supp_scores, wk.pooled, H, W, radius);
        k_update_mask<<<grid1(n_pix), 256, 0, stream>>>(wk.supp_scores, wk.pooled, wk.supp, wk.max_mask, n_pix);
    }
    k_apply_mask<<<grid1(n_pix), 256, 0, stream>>>(wk.scores, wk.max_mask, n_pix);
    SPLG_CUDA_CHECK(cudaGetLastError());

    if (cfg.remove_borders > 0) {
        k_border<<<grid2(H, W), block, 0, stream>>>(wk.scores, H, W, cfg.remove_borders);
    }

    std::vector<float> scores_h(static_cast<size_t>(n_pix));
    SPLG_CUDA_CHECK(cudaMemcpyAsync(scores_h.data(), wk.scores, scores_h.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
    SPLG_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> ys, xs;
    std::vector<float> sc;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float s = scores_h[static_cast<size_t>(y * W + x)];
            if (s > cfg.detection_threshold) {
                ys.push_back(y);
                xs.push_back(x);
                sc.push_back(s);
            }
        }
    }
    std::vector<int> order(sc.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return sc[static_cast<size_t>(a)] > sc[static_cast<size_t>(b)]; });
    if (cfg.max_keypoints > 0 && static_cast<int>(order.size()) > cfg.max_keypoints) {
        order.resize(static_cast<size_t>(cfg.max_keypoints));
    }
    const int n = static_cast<int>(order.size());
    Features feat;
    feat.n = n;
    feat.desc_dim = cfg.desc_dim;
    feat.keypoints.resize(static_cast<size_t>(n) * 2);
    feat.scores.resize(static_cast<size_t>(n));
    feat.descriptors.resize(static_cast<size_t>(n) * static_cast<size_t>(cfg.desc_dim));
    if (n == 0) {
        return feat;
    }
    for (int i = 0; i < n; ++i) {
        const int idx = order[static_cast<size_t>(i)];
        feat.keypoints[static_cast<size_t>(i * 2 + 0)] = static_cast<float>(xs[static_cast<size_t>(idx)]);
        feat.keypoints[static_cast<size_t>(i * 2 + 1)] = static_cast<float>(ys[static_cast<size_t>(idx)]);
        feat.scores[static_cast<size_t>(i)] = sc[static_cast<size_t>(idx)];
    }

    wk.ensure_kpts(n, cfg.desc_dim);
    SPLG_CUDA_CHECK(cudaMemcpyAsync(wk.kxy, feat.keypoints.data(), feat.keypoints.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
    k_sample_desc<<<grid1(n), 256, 0, stream>>>(wk.kxy, desc_dev, wk.desc, n, h, w, cfg.desc_dim);
    SPLG_CUDA_CHECK(cudaMemcpyAsync(feat.descriptors.data(), wk.desc, feat.descriptors.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
    SPLG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return feat;
}

}  // namespace splg
