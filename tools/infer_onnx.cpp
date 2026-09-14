#include "splg/io.hpp"
#include "splg/onnx_models.hpp"
#include "splg/superpoint_post.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

static void usage() {
    std::fprintf(stderr,
                 "Usage: infer_onnx --sp superpoint.onnx --lg lightglue.onnx --ref outputs/ref "
                 "--out outputs/onnx [--lg-from-ref] [--max-keypoints N] [--cpu]\n");
}

int main(int argc, char **argv) {
    std::string sp_path, lg_path, ref, out;
    bool lg_from_ref = false;
    bool use_cuda = true;
    splg::SuperPointConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sp") == 0 && i + 1 < argc) {
            sp_path = argv[++i];
        } else if (std::strcmp(argv[i], "--lg") == 0 && i + 1 < argc) {
            lg_path = argv[++i];
        } else if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out = argv[++i];
        } else if (std::strcmp(argv[i], "--lg-from-ref") == 0) {
            lg_from_ref = true;
        } else if (std::strcmp(argv[i], "--max-keypoints") == 0 && i + 1 < argc) {
            cfg.max_keypoints = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            use_cuda = false;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        }
    }
    if (sp_path.empty() || lg_path.empty() || ref.empty() || out.empty()) {
        usage();
        return 1;
    }

    try {
        auto img0 = splg::load_gray_npy(splg::join_path(ref, "image0.npy"));
        auto img1 = splg::load_gray_npy(splg::join_path(ref, "image1.npy"));

        splg::OnnxSuperPoint sp(sp_path, cfg, use_cuda);
        auto maps0 = sp.infer_dense(img0.first.data(), img0.second.height, img0.second.width);
        auto maps1 = sp.infer_dense(img1.first.data(), img1.second.height, img1.second.width);
        splg::write_dense(out, "0", maps0.score_logits, maps0.descriptors, maps0.h, maps0.w);
        splg::write_dense(out, "1", maps1.score_logits, maps1.descriptors, maps1.h, maps1.w);

        splg::Features f0 = sp.extract(img0.first.data(), img0.second.height, img0.second.width);
        splg::Features f1 = sp.extract(img1.first.data(), img1.second.height, img1.second.width);
        splg::write_features(out, "0", f0);
        splg::write_features(out, "1", f1);

        splg::OnnxLightGlue lg(lg_path, use_cuda);
        const float w0 = static_cast<float>(img0.second.width);
        const float h0 = static_cast<float>(img0.second.height);
        const float w1 = static_cast<float>(img1.second.width);
        const float h1 = static_cast<float>(img1.second.height);

        splg::Features lg0 = lg_from_ref ? splg::load_features(ref, "0") : f0;
        splg::Features lg1 = lg_from_ref ? splg::load_features(ref, "1") : f1;
        splg::Matches m = lg.match(lg0, lg1, w0, h0, w1, h1);
        splg::write_matches(out, m);

        try {
            splg::Features r0 = splg::load_features(ref, "0");
            splg::Features r1 = splg::load_features(ref, "1");
            splg::Matches rm = lg.match(r0, r1, w0, h0, w1, h1);
            splg::save_npy_i64(splg::join_path(out, "lg_ref_matches0.npy"),
                               {static_cast<int64_t>(rm.matches0.size())}, rm.matches0.data());
            splg::save_npy_f32(splg::join_path(out, "lg_ref_mscores0.npy"),
                               {static_cast<int64_t>(rm.scores0.size())}, rm.scores0.data());
        } catch (const std::exception &) {
        }

        int nm = 0;
        for (auto idx : m.matches0) {
            if (idx >= 0) {
                ++nm;
            }
        }
        std::printf("SuperPoint N0=%d N1=%d  LightGlue matches=%d  device=%s  wrote %s\n", f0.n, f1.n, nm,
                    sp.device(), out.c_str());
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "infer_onnx failed: %s\n", ex.what());
        return 1;
    }
}
