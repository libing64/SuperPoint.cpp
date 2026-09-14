#include "splg/io.hpp"

#include <stdexcept>
#include <sys/stat.h>

namespace splg {
namespace {

void ensure_dir(const std::string &dir) {
    if (dir.empty()) {
        return;
    }
    mkdir(dir.c_str(), 0755);
}

}  // namespace

std::pair<std::vector<float>, ImageHW> load_gray_npy(const std::string &path) {
    NpyArray a = load_npy(path);
    if (a.is_int64) {
        throw std::runtime_error("expected float image: " + path);
    }
    ImageHW hw;
    if (a.shape.size() == 2) {
        hw.height = static_cast<int>(a.shape[0]);
        hw.width = static_cast<int>(a.shape[1]);
    } else if (a.shape.size() == 4) {
        hw.height = static_cast<int>(a.shape[2]);
        hw.width = static_cast<int>(a.shape[3]);
    } else if (a.shape.size() == 3) {
        hw.height = static_cast<int>(a.shape[1]);
        hw.width = static_cast<int>(a.shape[2]);
    } else {
        throw std::runtime_error("unexpected image shape in " + path);
    }
    return {a.f32, hw};
}

void write_features(const std::string &dir, const std::string &suffix, const Features &f) {
    ensure_dir(dir);
    save_npy_f32(join_path(dir, "kpts" + suffix + ".npy"), {f.n, 2}, f.keypoints.data());
    save_npy_f32(join_path(dir, "scores" + suffix + ".npy"), {f.n}, f.scores.data());
    save_npy_f32(join_path(dir, "desc" + suffix + ".npy"), {f.n, f.desc_dim}, f.descriptors.data());
}

void write_matches(const std::string &dir, const Matches &m) {
    ensure_dir(dir);
    const int n = static_cast<int>(m.matches0.size());
    save_npy_i64(join_path(dir, "matches0.npy"), {n}, m.matches0.data());
    save_npy_f32(join_path(dir, "mscores0.npy"), {n}, m.scores0.data());
    int nm = 0;
    for (int i = 0; i < n; ++i) {
        if (m.matches0[static_cast<size_t>(i)] >= 0) {
            ++nm;
        }
    }
    std::vector<int64_t> pairs(static_cast<size_t>(nm * 2));
    std::vector<float> sc(static_cast<size_t>(nm));
    int w = 0;
    for (int i = 0; i < n; ++i) {
        if (m.matches0[static_cast<size_t>(i)] >= 0) {
            pairs[static_cast<size_t>(w * 2 + 0)] = i;
            pairs[static_cast<size_t>(w * 2 + 1)] = m.matches0[static_cast<size_t>(i)];
            sc[static_cast<size_t>(w)] = m.scores0[static_cast<size_t>(i)];
            ++w;
        }
    }
    save_npy_i64(join_path(dir, "match_pairs.npy"), {nm, 2}, pairs.data());
    save_npy_f32(join_path(dir, "match_scores.npy"), {nm}, sc.data());
}

Features load_features(const std::string &dir, const std::string &suffix) {
    NpyArray k = load_npy(join_path(dir, "kpts" + suffix + ".npy"));
    NpyArray s = load_npy(join_path(dir, "scores" + suffix + ".npy"));
    NpyArray d = load_npy(join_path(dir, "desc" + suffix + ".npy"));
    Features f;
    f.n = static_cast<int>(k.shape[0]);
    f.desc_dim = d.shape.size() > 1 ? static_cast<int>(d.shape[1]) : 256;
    f.keypoints = k.f32;
    f.scores = s.f32;
    f.descriptors = d.f32;
    return f;
}

void write_dense(const std::string &dir, const std::string &suffix, const std::vector<float> &logits,
                 const std::vector<float> &desc, int h, int w) {
    ensure_dir(dir);
    save_npy_f32(join_path(dir, "score_logits" + suffix + ".npy"), {1, 65, h, w}, logits.data());
    save_npy_f32(join_path(dir, "desc_dense" + suffix + ".npy"), {1, 256, h, w}, desc.data());
}

}  // namespace splg
