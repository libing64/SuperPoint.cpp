#pragma once

#include "splg/types.hpp"

#include <vector>

namespace splg {

// Normalize keypoints to LightGlue coords. size_wh = {W, H}.
void normalize_keypoints(const float *kpts_xy, int n, float width, float height, float *out_xy);

// scores: [M+1, N+1] log assignment (no batch). Returns matches0 of length M.
Matches filter_matches(const float *scores, int m, int n, float threshold);

}  // namespace splg
