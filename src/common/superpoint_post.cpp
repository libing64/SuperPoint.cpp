#include "splg/superpoint_post.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace splg {
namespace {

void max_pool2d(const std::vector<float> &in, std::vector<float> &out, int h, int w, int radius) {
    out.assign(static_cast<size_t>(h * w), 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float m = -1e30f;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= h) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= w) {
                        continue;
                    }
                    m = std::max(m, in[static_cast<size_t>(yy * w + xx)]);
                }
            }
            out[static_cast<size_t>(y * w + x)] = m;
        }
    }
}

void simple_nms(std::vector<float> &scores, int h, int w, int radius) {
    std::vector<float> pooled;
    max_pool2d(scores, pooled, h, w, radius);
    std::vector<float> max_mask(static_cast<size_t>(h * w), 0.0f);
    for (int i = 0; i < h * w; ++i) {
        max_mask[static_cast<size_t>(i)] = (scores[static_cast<size_t>(i)] == pooled[static_cast<size_t>(i)]) ? 1.0f : 0.0f;
    }
    std::vector<float> zeros(static_cast<size_t>(h * w), 0.0f);
    for (int iter = 0; iter < 2; ++iter) {
        std::vector<float> supp;
        max_pool2d(max_mask, supp, h, w, radius);
        std::vector<float> supp_scores(static_cast<size_t>(h * w));
        for (int i = 0; i < h * w; ++i) {
            supp_scores[static_cast<size_t>(i)] =
                (supp[static_cast<size_t>(i)] > 0.0f) ? 0.0f : scores[static_cast<size_t>(i)];
        }
        max_pool2d(supp_scores, pooled, h, w, radius);
        for (int i = 0; i < h * w; ++i) {
            const bool new_max = supp_scores[static_cast<size_t>(i)] == pooled[static_cast<size_t>(i)];
            const bool not_supp = !(supp[static_cast<size_t>(i)] > 0.0f);
            if (new_max && not_supp) {
                max_mask[static_cast<size_t>(i)] = 1.0f;
            }
        }
    }
    for (int i = 0; i < h * w; ++i) {
        if (max_mask[static_cast<size_t>(i)] == 0.0f) {
            scores[static_cast<size_t>(i)] = 0.0f;
        }
    }
}

void softmax_drop_dustbin(const float *logits, int h, int w, std::vector<float> &full_hw) {
    const int H = h * 8;
    const int W = w * 8;
    std::vector<float> cell(64);
    full_hw.assign(static_cast<size_t>(H * W), 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float m = logits[0 * h * w + y * w + x];
            for (int c = 1; c < 65; ++c) {
                m = std::max(m, logits[c * h * w + y * w + x]);
            }
            float sum = 0.0f;
            float probs[65];
            for (int c = 0; c < 65; ++c) {
                probs[c] = std::exp(logits[c * h * w + y * w + x] - m);
                sum += probs[c];
            }
            for (int c = 0; c < 64; ++c) {
                cell[static_cast<size_t>(c)] = probs[c] / sum;
            }
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    const int yy = y * 8 + i;
                    const int xx = x * 8 + j;
                    full_hw[static_cast<size_t>(yy * W + xx)] = cell[static_cast<size_t>(i * 8 + j)];
                }
            }
        }
    }
}

void sample_descriptors(const std::vector<float> &kpts_xy, const float *desc, int h, int w,
                        int dim, std::vector<float> &out) {
    const int n = static_cast<int>(kpts_xy.size() / 2);
    out.assign(static_cast<size_t>(n * dim), 0.0f);
    const float s = 8.0f;
    const float den_x = w * s - s / 2.0f - 0.5f;
    const float den_y = h * s - s / 2.0f - 0.5f;
    for (int i = 0; i < n; ++i) {
        float x = kpts_xy[static_cast<size_t>(i * 2 + 0)] - s / 2.0f + 0.5f;
        float y = kpts_xy[static_cast<size_t>(i * 2 + 1)] - s / 2.0f + 0.5f;
        x = x / den_x * 2.0f - 1.0f;
        y = y / den_y * 2.0f - 1.0f;
        // align_corners=True
        const float px = ((x + 1.0f) / 2.0f) * static_cast<float>(w - 1);
        const float py = ((y + 1.0f) / 2.0f) * static_cast<float>(h - 1);
        const int x0 = static_cast<int>(std::floor(px));
        const int y0 = static_cast<int>(std::floor(py));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float wx = px - static_cast<float>(x0);
        const float wy = py - static_cast<float>(y0);
        float norm = 0.0f;
        for (int c = 0; c < dim; ++c) {
            auto at = [&](int yy, int xx) -> float {
                if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
                    return 0.0f;
                }
                return desc[(c * h + yy) * w + xx];
            };
            const float v = (1.0f - wy) * (1.0f - wx) * at(y0, x0) + (1.0f - wy) * wx * at(y0, x1) +
                            wy * (1.0f - wx) * at(y1, x0) + wy * wx * at(y1, x1);
            out[static_cast<size_t>(i * dim + c)] = v;
            norm += v * v;
        }
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int c = 0; c < dim; ++c) {
            out[static_cast<size_t>(i * dim + c)] /= norm;
        }
    }
}

}  // namespace

Features superpoint_postprocess(const float *score_logits, const float *desc_dense, int h, int w,
                                const SuperPointConfig &cfg) {
    if (h <= 0 || w <= 0) {
        throw std::runtime_error("invalid SuperPoint feature map size");
    }
    const int H = h * 8;
    const int W = w * 8;
    std::vector<float> score_map;
    softmax_drop_dustbin(score_logits, h, w, score_map);
    simple_nms(score_map, H, W, cfg.nms_radius);

    if (cfg.remove_borders > 0) {
        const int pad = cfg.remove_borders;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (y < pad || x < pad || y >= H - pad || x >= W - pad) {
                    score_map[static_cast<size_t>(y * W + x)] = -1.0f;
                }
            }
        }
    }

    std::vector<int> ys;
    std::vector<int> xs;
    std::vector<float> sc;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float s = score_map[static_cast<size_t>(y * W + x)];
            if (s > cfg.detection_threshold) {
                ys.push_back(y);
                xs.push_back(x);
                sc.push_back(s);
            }
        }
    }

    std::vector<int> order(sc.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return sc[static_cast<size_t>(a)] > sc[static_cast<size_t>(b)]; });
    if (cfg.max_keypoints > 0 && static_cast<int>(order.size()) > cfg.max_keypoints) {
        order.resize(static_cast<size_t>(cfg.max_keypoints));
    }

    Features feat;
    feat.n = static_cast<int>(order.size());
    feat.desc_dim = cfg.desc_dim;
    feat.keypoints.resize(static_cast<size_t>(feat.n * 2));
    feat.scores.resize(static_cast<size_t>(feat.n));
    for (int i = 0; i < feat.n; ++i) {
        const int idx = order[static_cast<size_t>(i)];
        feat.keypoints[static_cast<size_t>(i * 2 + 0)] = static_cast<float>(xs[static_cast<size_t>(idx)]);
        feat.keypoints[static_cast<size_t>(i * 2 + 1)] = static_cast<float>(ys[static_cast<size_t>(idx)]);
        feat.scores[static_cast<size_t>(i)] = sc[static_cast<size_t>(idx)];
    }
    sample_descriptors(feat.keypoints, desc_dense, h, w, cfg.desc_dim, feat.descriptors);
    return feat;
}

}  // namespace splg
