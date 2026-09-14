#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace splg {

struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<float> f32;
    std::vector<int64_t> i64;
    bool is_int64 = false;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            n *= d;
        }
        return n;
    }
};

NpyArray load_npy(const std::string &path);
void save_npy_f32(const std::string &path, const std::vector<int64_t> &shape, const float *data);
void save_npy_i64(const std::string &path, const std::vector<int64_t> &shape, const int64_t *data);

}  // namespace splg
