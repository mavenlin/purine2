// Copyright Lin Min 2015
#include "operations/include/inner.hpp"

namespace purine {

Inner::Inner(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  int bottom_num = inputs_[0]->shape()[0];
  int top_num = outputs_[0]->shape()[0];
  CHECK_EQ(bottom_num, top_num);
  int bottom_channel = inputs_[0]->shape().Count() / inputs_[0]->shape()[0];
  int weight_channel = inputs_[1]->shape()[1];
  int weight_num = inputs_[1]->shape()[0];
  int top_channel = outputs_[0]->shape()[1];
  CHECK_EQ(bottom_channel, weight_channel);
  CHECK_EQ(top_channel, weight_num);
  CHECK_EQ(inputs_[1]->shape()[2] * inputs_[1]->shape()[3], 1);
}

void Inner::compute_cpu(const vector<bool>& add) {
  Shape bottom_shape = inputs_[0]->shape();
  Shape top_shape = outputs_[0]->shape();
  caffe::caffe_cpu_gemm<DTYPE>(CblasNoTrans, CblasTrans, bottom_shape[0],
      top_shape[1], bottom_shape.Count() / bottom_shape[0], (DTYPE)1.,
      inputs_[0]->cpu_data(), inputs_[1]->cpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_cpu_data());
}

void Inner::compute_gpu(const vector<bool>& add) {
  Shape bottom_shape = inputs_[0]->shape();
  Shape top_shape = outputs_[0]->shape();
  caffe::caffe_gpu_gemm<DTYPE>(CblasNoTrans, CblasTrans, bottom_shape[0],
      top_shape[1], bottom_shape.Count() / bottom_shape[0], (DTYPE)1.,
      inputs_[0]->gpu_data(), inputs_[1]->gpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_gpu_data());
}

InnerDown::InnerDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  int top_num = inputs_[0]->shape()[0];
  int bottom_num = outputs_[0]->shape()[0];
  CHECK_EQ(bottom_num, top_num);
  int bottom_channel = outputs_[0]->shape().Count()
      / outputs_[0]->shape()[0];
  int weight_channel = inputs_[1]->shape()[1];
  int weight_num = inputs_[1]->shape()[0];
  int top_channel = inputs_[0]->shape()[1];
  CHECK_EQ(bottom_channel, weight_channel);
  CHECK_EQ(top_channel, weight_num);
  CHECK_EQ(inputs_[1]->shape()[2] * inputs_[1]->shape()[3], 1);
}

void InnerDown::compute_cpu(const vector<bool>& add) {
  Shape top_shape = inputs_[0]->shape();
  Shape bottom_shape = outputs_[0]->shape();
  caffe::caffe_cpu_gemm<DTYPE>(CblasNoTrans, CblasNoTrans, bottom_shape[0],
      bottom_shape.Count() / bottom_shape[0], top_shape[1], (DTYPE)1.,
      inputs_[0]->cpu_data(), inputs_[1]->cpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_cpu_data());
}

void InnerDown::compute_gpu(const vector<bool>& add) {
  Shape top_shape = inputs_[0]->shape();
  Shape bottom_shape = outputs_[0]->shape();
  caffe::caffe_gpu_gemm<DTYPE>(CblasNoTrans, CblasNoTrans, bottom_shape[0],
      bottom_shape.Count() / bottom_shape[0], top_shape[1], (DTYPE)1.,
      inputs_[0]->gpu_data(), inputs_[1]->gpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_gpu_data());
}

InnerWeight::InnerWeight(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  int top_num = inputs_[0]->shape()[0];
  int bottom_num = inputs_[1]->shape()[0];
  CHECK_EQ(bottom_num, top_num);
  int bottom_channel = inputs_[1]->shape().Count() / inputs_[1]->shape()[0];
  int weight_channel = outputs_[0]->shape()[1];
  int weight_num = outputs_[0]->shape()[0];
  int top_channel = inputs_[0]->shape()[1];
  CHECK_EQ(bottom_channel, weight_channel);
  CHECK_EQ(top_channel, weight_num);
  CHECK_EQ(outputs_[0]->shape()[2] * outputs_[0]->shape()[3], 1);
}

void InnerWeight::compute_cpu(const vector<bool>& add) {
  Shape top_shape = inputs_[0]->shape();
  Shape bottom_shape = inputs_[1]->shape();
  caffe::caffe_cpu_gemm<DTYPE>(CblasTrans, CblasNoTrans, top_shape[1],
      bottom_shape.Count() / bottom_shape[0], bottom_shape[0], (DTYPE)1.,
      inputs_[0]->cpu_data(), inputs_[1]->cpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_cpu_data());
}

void InnerWeight::compute_gpu(const vector<bool>& add) {
  Shape top_shape = inputs_[0]->shape();
  Shape bottom_shape = inputs_[1]->shape();
  caffe::caffe_gpu_gemm<DTYPE>(CblasTrans, CblasNoTrans, top_shape[1],
      bottom_shape.Count() / bottom_shape[0], bottom_shape[0], (DTYPE)1.,
      inputs_[0]->gpu_data(), inputs_[1]->gpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_gpu_data());
}

}
