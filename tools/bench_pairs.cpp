#include "splg/io.hpp"
#include "splg/types.hpp"

#ifdef SPLG_BENCH_TRT
#include "splg/trt_models.hpp"
#else
#include "splg/onnx_models.hpp"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace {

void usage() {
#ifdef SPLG_BENCH_TRT
    std::fprintf(stderr,
                 "Usage: bench_trt --sp superpoint.engine --lg lightglue.engine "
                 "--list pairs.txt --out outputs/hpatches/trt [--max-keypoints N] [--warmup N]\n");
#else
    std::fprintf(stderr,
                 "Usage: bench_onnx --sp superpoint.onnx --lg lightglue.onnx "
                 "--list pairs.txt --out outputs/hpatches/onnx [--max-keypoints N] [--warmup N] [--cpu]\n");
#endif
}

void ensure_dir(const std::string &dir) {
    if (dir.empty()) {
        return;
    }
    mkdir(dir.c_str(), 0755);
}

struct PairItem {
    std::string pair_id;
    std::string image0;
    std::string image1;
};

std::vector<PairItem> load_list(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open pair list " + path);
    }
    std::vector<PairItem> items;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        PairItem item;
        if (!(ss >> item.pair_id >> item.image0 >> item.image1)) {
            throw std::runtime_error("bad pair list line: " + line);
        }
        items.push_back(std::move(item));
    }
    return items;
}

int count_matches(const splg::Matches &m) {
    int n = 0;
    for (auto idx : m.matches0) {
        if (idx >= 0) {
            ++n;
        }
    }
    return n;
}

}  // namespace

int main(int argc, char **argv) {
    std::string sp_path, lg_path, list_path, out;
    int warmup = 3;
    bool use_cuda = true;
    splg::SuperPointConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sp") == 0 && i + 1 < argc) {
            sp_path = argv[++i];
        } else if (std::strcmp(argv[i], "--lg") == 0 && i + 1 < argc) {
            lg_path = argv[++i];
        } else if (std::strcmp(argv[i], "--list") == 0 && i + 1 < argc) {
            list_path = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out = argv[++i];
        } else if (std::strcmp(argv[i], "--max-keypoints") == 0 && i + 1 < argc) {
            cfg.max_keypoints = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            use_cuda = false;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        }
    }
    if (sp_path.empty() || lg_path.empty() || list_path.empty() || out.empty()) {
        usage();
        return 1;
    }

    try {
        auto items = load_list(list_path);
        if (items.empty()) {
            throw std::runtime_error("pair list is empty");
        }
        ensure_dir(out);

#ifdef SPLG_BENCH_TRT
        splg::TrtSuperPoint sp(sp_path, cfg);
        splg::TrtLightGlue lg(lg_path);
        const char *backend = "trt";
        const char *device = "cuda";
        (void)use_cuda;
#else
        splg::OnnxSuperPoint sp(sp_path, cfg, use_cuda);
        splg::OnnxLightGlue lg(lg_path, use_cuda);
        const char *backend = "onnx";
        const char *device = sp.device();
#endif
        std::printf("%s device=%s  pairs=%zu\n", backend, device, items.size());

        std::ofstream jsonl(splg::join_path(out, "times.jsonl"));
        if (!jsonl) {
            throw std::runtime_error("cannot write " + splg::join_path(out, "times.jsonl"));
        }

        using Clock = std::chrono::steady_clock;
        double sum_sp = 0.0;
        double sum_lg = 0.0;
        double sum_e2e = 0.0;
        int timed = 0;

        for (size_t i = 0; i < items.size(); ++i) {
            const auto &item = items[i];
            auto img0 = splg::load_gray_npy(item.image0);
            auto img1 = splg::load_gray_npy(item.image1);
            const float w0 = static_cast<float>(img0.second.width);
            const float h0 = static_cast<float>(img0.second.height);
            const float w1 = static_cast<float>(img1.second.width);
            const float h1 = static_cast<float>(img1.second.height);

            const auto t0 = Clock::now();
            splg::Features f0 = sp.extract(img0.first.data(), img0.second.height, img0.second.width);
            splg::Features f1 = sp.extract(img1.first.data(), img1.second.height, img1.second.width);
            const auto t1 = Clock::now();
            splg::Matches m = lg.match(f0, f1, w0, h0, w1, h1);
            const auto t2 = Clock::now();

            const double ms_sp = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double ms_lg = std::chrono::duration<double, std::milli>(t2 - t1).count();
            const double ms_e2e = std::chrono::duration<double, std::milli>(t2 - t0).count();
            const bool count_time = static_cast<int>(i) >= warmup;
            if (count_time) {
                sum_sp += ms_sp;
                sum_lg += ms_lg;
                sum_e2e += ms_e2e;
                ++timed;
            }

            const std::string pair_dir = splg::join_path(out, item.pair_id);
            ensure_dir(pair_dir);
            splg::save_npy_f32(splg::join_path(pair_dir, "kpts0.npy"), {f0.n, 2}, f0.keypoints.data());
            splg::save_npy_f32(splg::join_path(pair_dir, "kpts1.npy"), {f1.n, 2}, f1.keypoints.data());
            splg::write_matches(pair_dir, m);

            const int nm = count_matches(m);
            jsonl << "{\"pair_id\":\"" << item.pair_id << "\",\"n0\":" << f0.n << ",\"n1\":" << f1.n
                  << ",\"n_matches\":" << nm << ",\"ms_sp\":" << ms_sp << ",\"ms_lg\":" << ms_lg
                  << ",\"ms_e2e\":" << ms_e2e << ",\"warmup\":" << (count_time ? 0 : 1) << ",\"device\":\""
                  << device << "\"}\n";
            std::printf("[%s %zu/%zu] %s  N0=%d N1=%d matches=%d  SP=%.2f ms LG=%.2f ms e2e=%.2f ms%s\n",
                        backend, i + 1, items.size(), item.pair_id.c_str(), f0.n, f1.n, nm, ms_sp, ms_lg,
                        ms_e2e, count_time ? "" : " (warmup)");
        }

        const double n = timed > 0 ? static_cast<double>(timed) : 1.0;
        std::printf("%s mean over %d pairs (warmup %d): SP=%.2f ms  LG=%.2f ms  e2e=%.2f ms\n", backend,
                    timed, warmup, sum_sp / n, sum_lg / n, sum_e2e / n);
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "bench_pairs failed: %s\n", ex.what());
        return 1;
    }
}
