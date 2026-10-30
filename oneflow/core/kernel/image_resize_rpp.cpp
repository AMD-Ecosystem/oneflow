/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/kernel/image_resize_rpp.h"

#if defined(WITH_CUDA) && defined(USE_HIP)

#include <rpp/rpp.h>
#include <stdexcept>

namespace oneflow {

struct RppImageResizer::Impl {
  rppHandle_t handle = nullptr;
  // The resize kernel reads its roi and output size from pinned host memory.
  RpptROI* roi = nullptr;
  RpptImagePatch* dst_size = nullptr;
};

RppImageResizer::RppImageResizer(hipStream_t stream) : impl_(new Impl) {
  if (rppCreate(&impl_->handle, 1, 0, stream, RppBackend::RPP_HIP_BACKEND) != rppStatusSuccess
      || hipHostMalloc(reinterpret_cast<void**>(&impl_->roi), sizeof(RpptROI)) != hipSuccess
      || hipHostMalloc(reinterpret_cast<void**>(&impl_->dst_size), sizeof(RpptImagePatch))
             != hipSuccess) {
    throw std::runtime_error("failed to set up the RPP image resizer");
  }
}

RppImageResizer::~RppImageResizer() {
  if (impl_->handle != nullptr) { (void)rppDestroy(impl_->handle, RppBackend::RPP_HIP_BACKEND); }
  if (impl_->roi != nullptr) { (void)hipHostFree(impl_->roi); }
  if (impl_->dst_size != nullptr) { (void)hipHostFree(impl_->dst_size); }
  delete impl_;
}

bool RppImageResizer::ResizeRgb(const unsigned char* src, int src_width, int src_height, int roi_x,
                                int roi_y, int roi_w, int roi_h, unsigned char* dst, int dst_width,
                                int dst_height) {
  RpptDesc src_desc(1, 3, src_height, src_width, RpptDataType::U8, RpptLayout::NHWC);
  RpptDesc dst_desc(1, 3, dst_height, dst_width, RpptDataType::U8, RpptLayout::NHWC);
  impl_->roi->xywhROI.xy.x = roi_x;
  impl_->roi->xywhROI.xy.y = roi_y;
  impl_->roi->xywhROI.roiWidth = roi_w;
  impl_->roi->xywhROI.roiHeight = roi_h;
  impl_->dst_size->width = dst_width;
  impl_->dst_size->height = dst_height;
  return rppt_resize(const_cast<unsigned char*>(src), &src_desc, dst, &dst_desc, impl_->dst_size,
                     RpptInterpolationType::BILINEAR, impl_->roi, RpptRoiType::XYWH, impl_->handle,
                     RppBackend::RPP_HIP_BACKEND)
         == RPP_SUCCESS;
}

}  // namespace oneflow

#endif  // defined(WITH_CUDA) && defined(USE_HIP)
