#pragma once

#include "splg/types.hpp"

#include <vector>

namespace splg {

// score_logits: [1, 65, h, w] or [65, h, w] row-major NCHW
// desc_dense:   [1, 256, h, w] L2-normalized dense map
Features superpoint_postprocess(const float *score_logits, const float *desc_dense, int h, int w,
                                const SuperPointConfig &cfg);

}  // namespace splg
