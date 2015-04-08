// Copyright Lin Min 2015
#include <string>
#include <cfloat>

#include "operations/include/softmax.hpp"
#include "operations/cudnn.hpp"
#include "caffeine/math_functions.hpp"

using std::string;

namespace purine {

Softmax::Softmax(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(mode) = args;
  CHECK_EQ(inputs_[0]->shape(), outputs_[0]->shape());
  Shape bottom_shape = inputs_[0]->shape();
  Shape top_shape = outputs_[0]->shape();
  Stride bottom_stride = inputs_[0]->stride();
  Stride top_stride = outputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_shape, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_shape, top_stride);
  if (mode == "channel") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_CHANNEL;
  } else if (mode == "instance") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
  } else {
    LOG(FATAL) << "Unknown softmax mode " << mode;
  }
}

Softmax::~Softmax() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void Softmax::compute_cpu(const vector<bool>& add) {

}

void Softmax::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnSoftmaxForward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
          softmax_mode_, &alpha, bottom_desc_, inputs_[0]->gpu_data(), &beta,
          top_desc_, outputs_[0]->mutable_gpu_data()));
}

SoftmaxDown::SoftmaxDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(mode) = args;
  CHECK_EQ(inputs_[0]->shape(), outputs_[0]->shape());
  CHECK_EQ(inputs_[1]->shape(), outputs_[0]->shape());
  Shape bottom_shape = outputs_[0]->shape();
  Shape top_shape = inputs_[0]->shape();
  Stride bottom_stride = outputs_[0]->stride();
  Stride top_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_shape, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_shape, top_stride);
  if (mode == "channel") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_CHANNEL;
  } else if (mode == "instance") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
  } else {
    LOG(FATAL) << "Unknown softmax mode " << mode;
  }
}

SoftmaxDown::~SoftmaxDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void SoftmaxDown::compute_cpu(const vector<bool>& add) {

}

void SoftmaxDown::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnSoftmaxBackward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
          softmax_mode_, &alpha, top_desc_, inputs_[1]->gpu_data(),
          top_desc_, inputs_[0]->gpu_data(), &beta, bottom_desc_,
          outputs_[0]->mutable_gpu_data()));
}

SoftmaxLoss::SoftmaxLoss(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  CHECK_EQ(outputs_[0]->shape(), Shape({1, 1, 1, 1}));
  CHECK_EQ(inputs_[1]->shape()[0], inputs_[0]->shape()[0]);
  CHECK_EQ(inputs_[1]->shape()[2], inputs_[0]->shape()[2]);
  CHECK_EQ(inputs_[1]->shape()[3], inputs_[0]->shape()[3]);
}

void SoftmaxLoss::compute_cpu(const vector<bool>& add) {
  const DTYPE* softmax_data = inputs_[0]->cpu_data();
  const DTYPE* label_data = inputs_[1]->cpu_data();
  Shape softmax_shape = inputs_[0]->shape();
  int num = softmax_shape[0];
  int dim = softmax_shape.Count() / num;
  int spatial_dim = softmax_shape[2] * softmax_shape[3];
  DTYPE loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      loss -= log(std::max(softmax_data[i * dim + static_cast<int>(
          label_data[i * spatial_dim + j]) * spatial_dim + j], DTYPE(FLT_MIN)));
    }
  }
  *(outputs_[0]->mutable_cpu_data()) = loss / num / spatial_dim;
}

SoftmaxLossDown::SoftmaxLossDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  CHECK_EQ(inputs_[0]->shape(), outputs_[0]->shape());
}

void SoftmaxLossDown::compute_cpu(const vector<bool>& add) {
  DTYPE* bottom_diff = outputs_[0]->mutable_cpu_data();
  const DTYPE* softmax_data = inputs_[0]->cpu_data();
  Shape softmax_shape = inputs_[0]->shape();
  caffe::caffe_cpu_copy(softmax_shape.Count(), softmax_data, bottom_diff);
  const DTYPE* label = inputs_[1]->cpu_data();
  int num = softmax_shape[0];
  int dim = softmax_shape.Count() / num;
  int spatial_dim = softmax_shape[2] * softmax_shape[3];
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
          * spatial_dim + j] -= 1;
    }
  }
  // Scale gradient
  const DTYPE loss_weight = inputs_[2]->cpu_data()[0];
  caffe::caffe_scal(softmax_shape.Count(), loss_weight / num / spatial_dim,
      bottom_diff);
}

}
