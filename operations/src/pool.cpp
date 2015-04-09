// Copyright Lin Min 2015

#include "operations/include/pool.hpp"

namespace purine {

// Update cudnn R2
Pool::Pool(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(method, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) = args;
  Shape bottom_shape = inputs_[0]->shape();
  Shape top_shape = outputs_[0]->shape();
  Stride bottom_stride = inputs_[0]->stride();
  Stride top_stride = outputs_[0]->stride();
  CHECK_EQ(top_shape[0], bottom_shape[0]);
  CHECK_EQ(top_shape[1], bottom_shape[1]);
  CHECK_EQ(top_shape[2], static_cast<int>(ceil(static_cast<float>(
      bottom_shape[2] + 2 * pad_h - kernel_h) / stride_h)) + 1);
  CHECK_EQ(top_shape[3], static_cast<int>(ceil(static_cast<float>(
      bottom_shape[3] + 2 * pad_w - kernel_w) / stride_w)) + 1);
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_shape, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_shape, top_stride);
  cudnnPoolingMode_t mode;
  if (method == "max") {
    mode = CUDNN_POOLING_MAX;
  } else if (method == "average") {
    mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else if (method == "average_exclude_padding") {
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  } else {
    LOG(FATAL) << "unknown pooling method: " << method;
  }
  cudnn::createPoolingDesc<DTYPE>(&pool_desc_, mode, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w);
}

Pool::~Pool() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
}

void Pool::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnPoolingForward(cudnn_handle(), pool_desc_, &alpha,
          bottom_desc_, inputs_[0]->gpu_data(), &beta, top_desc_,
          outputs_[0]->mutable_gpu_data()));
}

PoolDown::PoolDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(method, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) = args;
  Shape bottom_shape = outputs_[0]->shape();
  Shape top_shape = inputs_[0]->shape();
  CHECK_EQ(top_shape[0], bottom_shape[0]);
  CHECK_EQ(top_shape[1], bottom_shape[1]);
  CHECK_EQ(top_shape[2], static_cast<int>(ceil(static_cast<float>(
      bottom_shape[2] + 2 * pad_h - kernel_h) / stride_h)) + 1);
  CHECK_EQ(top_shape[3], static_cast<int>(ceil(static_cast<float>(
      bottom_shape[3] + 2 * pad_w - kernel_w) / stride_w)) + 1);
  Stride bottom_stride = outputs_[0]->stride();
  Stride top_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_shape, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_shape, top_stride);
  cudnnPoolingMode_t mode;
  if (method == "max") {
    mode = CUDNN_POOLING_MAX;
  } else if (method == "average") {
    mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else if (method == "average_exclude_padding") {
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  } else {
    LOG(FATAL) << "unknown pooling method: " << method;
  }
  cudnn::createPoolingDesc<DTYPE>(&pool_desc_, mode, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w);
}

PoolDown::~PoolDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
}

void PoolDown::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnPoolingBackward(cudnn_handle(), pool_desc_, &alpha,
          top_desc_, inputs_[1]->gpu_data(), top_desc_, inputs_[0]->gpu_data(),
          bottom_desc_, inputs_[2]->gpu_data(), &beta, bottom_desc_,
          outputs_[0]->mutable_gpu_data()));
}

}
