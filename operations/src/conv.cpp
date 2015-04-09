// Copyright Lin Min 2015
#include "operations/include/conv.hpp"

namespace purine {

// Update cudnn R2
Conv::Conv(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(pad_h, pad_w, stride_h, stride_w) = args;
  CHECK_EQ(inputs_.size(), 2);
  CHECK_EQ(outputs_.size(), 1);
  Shape bottom_shape = inputs_[0]->shape();
  Stride bottom_stride = inputs_[0]->stride();
  Shape top_size = outputs_[0]->shape();
  Stride top_stride = outputs_[0]->stride();
  Shape kernel_size = inputs_[1]->shape();

  CHECK_EQ(bottom_shape[0], top_size[0]);
  CHECK_EQ(bottom_shape[1], kernel_size[1]);
  CHECK_EQ(kernel_size[0], top_size[1]);
  CHECK_EQ((bottom_shape[2] + 2 * pad_h - kernel_size[2])
      / stride_h + 1, top_size[2]);
  CHECK_EQ((bottom_shape[3] + 2 * pad_w - kernel_size[3])
      / stride_w + 1, top_size[3]);
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_shape, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
  cudnn::createFilterDesc<DTYPE>(&filter_desc_, kernel_size);
  cudnn::createConvolutionDesc<DTYPE>(&conv_desc_, pad_h, pad_w, stride_h,
      stride_w);
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn_handle(), bottom_desc_,
          filter_desc_, conv_desc_, top_desc_,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
          bottom_desc_, filter_desc_, conv_desc_, top_desc_, algo_,
          &workspace_size_));
}

Conv::~Conv() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void Conv::compute_gpu(const vector<bool>& add) {
  if (!workspace_ && workspace_size_ != 0) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    workspace_.reset(new Tensor(current_rank(), device,
            {1, 1, 1, workspace_size_}));
  }
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle(), &alpha, bottom_desc_,
          inputs_[0]->gpu_data(), filter_desc_, inputs_[1]->gpu_data(),
          conv_desc_, algo_, workspace_ ? workspace_->mutable_gpu_data() : 0,
          workspace_size_, &beta, top_desc_, outputs_[0]->mutable_gpu_data()));
}

ConvDown::ConvDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(pad_h, pad_w, stride_h, stride_w) = args;
  Shape bottom_shape = outputs_[0]->shape();
  Stride bottom_stride = outputs_[0]->stride();
  Shape top_size = inputs_[0]->shape();
  Stride top_stride = inputs_[0]->stride();
  Shape kernel_size = inputs_[1]->shape();
  CHECK_EQ(bottom_shape[0], top_size[0]);
  CHECK_EQ(bottom_shape[1], kernel_size[1]);
  CHECK_EQ(kernel_size[0], top_size[1]);
  CHECK_EQ((bottom_shape[2] + 2 * pad_h - kernel_size[2])
      / stride_h + 1, top_size[2]);
  CHECK_EQ((bottom_shape[3] + 2 * pad_w - kernel_size[3])
      / stride_w + 1, top_size[3]);
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_shape, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
  cudnn::createFilterDesc<DTYPE>(&filter_desc_, kernel_size);
  cudnn::createConvolutionDesc<DTYPE>(&conv_desc_, pad_h, pad_w, stride_h,
      stride_w);
}

ConvDown::~ConvDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void ConvDown::compute_gpu(const vector<bool>& add) {
  const DTYPE* weight_data = inputs_[1]->gpu_data();
  const DTYPE* top_diff = inputs_[0]->gpu_data();
  DTYPE* bottom_diff = outputs_[0]->mutable_gpu_data();
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn_handle(), &alpha,
          filter_desc_, weight_data, top_desc_, top_diff, conv_desc_, &beta,
          bottom_desc_, bottom_diff));
}

ConvWeight::ConvWeight(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(pad_h, pad_w, stride_h, stride_w) = args;
  Shape bottom_shape = inputs_[1]->shape();
  Shape top_size = inputs_[0]->shape();
  Shape kernel_size = outputs_[0]->shape();
  Stride bottom_stride = inputs_[1]->stride();
  Stride top_stride = inputs_[0]->stride();
  CHECK_EQ(bottom_shape[0], top_size[0]);
  CHECK_EQ(bottom_shape[1], kernel_size[1]);
  CHECK_EQ(kernel_size[0], top_size[1]);
  CHECK_EQ((bottom_shape[2] + 2 * pad_h - kernel_size[2])
      / stride_h + 1, top_size[2]);
  CHECK_EQ((bottom_shape[3] + 2 * pad_w - kernel_size[3])
      / stride_w + 1, top_size[3]);
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_shape, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
  cudnn::createFilterDesc<DTYPE>(&filter_desc_, kernel_size);
  cudnn::createConvolutionDesc<DTYPE>(&conv_desc_, pad_h, pad_w, stride_h,
      stride_w);
}

ConvWeight::~ConvWeight() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void ConvWeight::compute_gpu(const vector<bool>& add) {
  const DTYPE* top_diff = inputs_[0]->gpu_data();
  const DTYPE* bottom_data = inputs_[1]->gpu_data();
  DTYPE* weight_diff = outputs_[0]->mutable_gpu_data();
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn_handle(), &alpha,
          bottom_desc_, bottom_data, top_desc_, top_diff, conv_desc_, &beta,
          filter_desc_, weight_diff));
}

}
