#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace splg {

struct Features {
    std::vector<float> keypoints;    // N * 2, (x, y)
    std::vector<float> scores;       // N
    std::vector<float> descriptors;  // N * 256
    int n = 0;
    int desc_dim = 256;
};

struct Matches {
    std::vector<int64_t> matches0;  // N0, -1 if unmatched
    std::vector<float> scores0;     // N0
};

struct SuperPointConfig {
    int nms_radius = 4;
    float detection_threshold = 0.0005f;
    int remove_borders = 4;
    int max_keypoints = 2048;
    int desc_dim = 256;
};

struct ImageHW {
    int height = 0;
    int width = 0;
};

inline std::string join_path(const std::string &dir, const std::string &name) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

}  // namespace splg
