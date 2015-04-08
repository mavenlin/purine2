// Copyright Lin Min 2015
#include "operations/include/bias.hpp"

namespace purine {

Bias::Bias(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs)  {
  Shape top_shape = outputs_[0]->shape();
  Shape bias_shape = inputs_[0]->shape();
  CHECK_EQ(top_shape[1], bias_shape[1]);
  CHECK_EQ(bias_shape[0], 1);
  CHECK_EQ(bias_shape[2], 1);
  CHECK_EQ(bias_shape[3], 1);
  Stride bias_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bias_desc_, bias_shape, bias_stride);
  Stride top_stride = outputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_shape, top_stride);
}

Bias::~Bias() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void Bias::compute_cpu(const vector<bool>& add) {

}

void Bias::compute_gpu(const vector<bool>& add) {
  Shape s = outputs_[0]->shape();
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnAddTensor(cudnn_handle(), CUDNN_ADD_SAME_C, &alpha,
          bias_desc_, inputs_[0]->gpu_data(), &beta, top_desc_,
          outputs_[0]->mutable_gpu_data()));
}

BiasDown::BiasDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  Shape top_shape = inputs_[0]->shape();
  Shape bias_shape = outputs_[0]->shape();
  CHECK_EQ(top_shape[1], bias_shape[1]);
  CHECK_EQ(bias_shape[0], 1);
  CHECK_EQ(bias_shape[2], 1);
  CHECK_EQ(bias_shape[3], 1);
  Stride bias_stride = outputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bias_desc_, bias_shape, bias_stride);
  Stride top_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_shape, top_stride);
}

BiasDown::~BiasDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void BiasDown::compute_cpu(const vector<bool>& add) {

}

void BiasDown::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn_handle(), &alpha, top_desc_,
          inputs_[0]->gpu_data(), &beta, bias_desc_,
          outputs_[0]->mutable_gpu_data()));
}

}
