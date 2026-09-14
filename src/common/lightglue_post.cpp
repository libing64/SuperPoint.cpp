#include "splg/lightglue_post.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace splg {

void normalize_keypoints(const float *kpts_xy, int n, float width, float height, float *out_xy) {
    const float shift_x = width / 2.0f;
    const float shift_y = height / 2.0f;
    const float scale = std::max(width, height) / 2.0f;
    for (int i = 0; i < n; ++i) {
        out_xy[i * 2 + 0] = (kpts_xy[i * 2 + 0] - shift_x) / scale;
        out_xy[i * 2 + 1] = (kpts_xy[i * 2 + 1] - shift_y) / scale;
    }
}

Matches filter_matches(const float *scores, int m, int n, float threshold) {
    Matches out;
    out.matches0.assign(static_cast<size_t>(m), -1);
    out.scores0.assign(static_cast<size_t>(m), 0.0f);
    // scores is [m+1, n+1]
    std::vector<int> m0(static_cast<size_t>(m), -1);
    std::vector<int> m1(static_cast<size_t>(n), -1);
    std::vector<float> max0(static_cast<size_t>(m), -1e30f);
    std::vector<float> max1(static_cast<size_t>(n), -1e30f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const float v = scores[i * (n + 1) + j];
            if (v > max0[static_cast<size_t>(i)]) {
                max0[static_cast<size_t>(i)] = v;
                m0[static_cast<size_t>(i)] = j;
            }
            if (v > max1[static_cast<size_t>(j)]) {
                max1[static_cast<size_t>(j)] = v;
                m1[static_cast<size_t>(j)] = i;
            }
        }
    }
    for (int i = 0; i < m; ++i) {
        const int j = m0[static_cast<size_t>(i)];
        if (j < 0) {
            continue;
        }
        const bool mutual = m1[static_cast<size_t>(j)] == i;
        const float s = std::exp(max0[static_cast<size_t>(i)]);
        if (mutual && s > threshold) {
            out.matches0[static_cast<size_t>(i)] = j;
            out.scores0[static_cast<size_t>(i)] = s;
        }
    }
    return out;
}

}  // namespace splg
