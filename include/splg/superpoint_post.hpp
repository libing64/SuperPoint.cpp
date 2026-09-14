#pragma once

#include "splg/types.hpp"

#include <vector>

#ifdef SPLG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace splg {

// score_logits: [1, 65, h, w] or [65, h, w] row-major NCHW
// desc_dense:   [1, 256, h, w] L2-normalized dense map
Features superpoint_postprocess(const float *score_logits, const float *desc_dense, int h, int w,
                                const SuperPointConfig &cfg);

#ifdef SPLG_HAS_CUDA
// Device pointers. logits_dev [65,h,w], desc_dev [256,h,w], both float32 NCHW.
Features superpoint_postprocess_cuda(const float *logits_dev, const float *desc_dev, int h, int w,
                                     const SuperPointConfig &cfg, cudaStream_t stream);
#endif

}  // namespace splg
