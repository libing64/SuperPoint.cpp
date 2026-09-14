#pragma once

#include "splg/npy.hpp"
#include "splg/types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace splg {

std::pair<std::vector<float>, ImageHW> load_gray_npy(const std::string &path);
void write_features(const std::string &dir, const std::string &suffix, const Features &f);
void write_matches(const std::string &dir, const Matches &m);
Features load_features(const std::string &dir, const std::string &suffix);
void write_dense(const std::string &dir, const std::string &suffix, const std::vector<float> &logits,
                 const std::vector<float> &desc, int h, int w);

}  // namespace splg
