#ifdef SPLG_HAS_TENSORRT

#include "splg/io.hpp"
#include "splg/superpoint_post.hpp"
#include "splg/trt_models.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void usage() {
    std::fprintf(stderr,
                 "Usage: infer_trt --sp superpoint.engine --lg lightglue.engine --ref outputs/ref "
                 "--out outputs/trt [--lg-from-ref] [--max-keypoints N]\n");
}

int main(int argc, char **argv) {
    std::string sp_path, lg_path, ref, out;
    bool lg_from_ref = false;
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
        }
    }
    if (sp_path.empty() || lg_path.empty() || ref.empty() || out.empty()) {
        usage();
        return 1;
    }
    try {
        auto img0 = splg::load_gray_npy(splg::join_path(ref, "image0.npy"));
        auto img1 = splg::load_gray_npy(splg::join_path(ref, "image1.npy"));
        splg::TrtSuperPoint sp(sp_path, cfg);

        std::vector<float> l0, d0, l1, d1;
        int h0 = 0, w0 = 0, h1 = 0, w1 = 0;
        sp.infer_dense(img0.first.data(), img0.second.height, img0.second.width, l0, d0, h0, w0);
        sp.infer_dense(img1.first.data(), img1.second.height, img1.second.width, l1, d1, h1, w1);
        splg::write_dense(out, "0", l0, d0, h0, w0);
        splg::write_dense(out, "1", l1, d1, h1, w1);
        splg::Features f0 = sp.extract(img0.first.data(), img0.second.height, img0.second.width);
        splg::Features f1 = sp.extract(img1.first.data(), img1.second.height, img1.second.width);
        splg::write_features(out, "0", f0);
        splg::write_features(out, "1", f1);

        splg::TrtLightGlue lg(lg_path);
        splg::Features a0 = lg_from_ref ? splg::load_features(ref, "0") : f0;
        splg::Features a1 = lg_from_ref ? splg::load_features(ref, "1") : f1;
        splg::Matches m = lg.match(a0, a1, static_cast<float>(img0.second.width),
                                   static_cast<float>(img0.second.height),
                                   static_cast<float>(img1.second.width),
                                   static_cast<float>(img1.second.height));
        splg::write_matches(out, m);
        try {
            auto r0 = splg::load_features(ref, "0");
            auto r1 = splg::load_features(ref, "1");
            auto rm = lg.match(r0, r1, static_cast<float>(img0.second.width),
                               static_cast<float>(img0.second.height),
                               static_cast<float>(img1.second.width),
                               static_cast<float>(img1.second.height));
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
        std::printf("TRT SuperPoint N0=%d N1=%d  LightGlue matches=%d  wrote %s\n", f0.n, f1.n, nm,
                    out.c_str());
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "infer_trt failed: %s\n", ex.what());
        return 1;
    }
}

#else

#include <cstdio>
int main() {
    std::fprintf(stderr, "infer_trt was built without TensorRT (SPLG_HAS_TENSORRT).\n");
    return 1;
}

#endif
