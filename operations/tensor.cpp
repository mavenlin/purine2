// Copyright Lin Min 2015
#include "operations/tensor.hpp"

namespace purine {

Tensor::Tensor(int rank, int device, const Shape& shape, const Offset& offset,
    const Stride& stride) : shape_(shape), offset_(offset), stride_(stride),
                            rank_(rank), device_(device) {
}

Tensor::Tensor(int rank, int device, const Shape& shape)
    : shape_(shape), rank_(rank), device_(device) {
  offset_ = Offset{0, 0, 0, 0};
  stride_ = Stride(shape);
}

Tensor::~Tensor() {
  data_.reset();
}

int Tensor::offset(const Offset& off, const Stride& stride) {
  return off[0] * stride[0] + off[1] * stride[1] + off[2] * stride[2]
      + off[3] * stride[3];
}

void Tensor::alloc_mem(DTYPE** data, const Shape& shape, int rank, int device) {
  CHECK_GT(shape.Count(), 0);
  CHECK_EQ(current_rank(), rank) << "Can't allocate memory on another machine";
  if (device < 0) {
// #ifndef NDEBUG
//     cudaHostAlloc(data, sizeof(DTYPE) * (1 + shape.Count()),
//         cudaHostAllocPortable);
// #else
    cudaHostAlloc(data, sizeof(DTYPE) * shape.Count(), cudaHostAllocPortable);
// #endif
  } else {
    SWITCH_DEVICE(device);
// #ifndef NDEBUG
//     CUDA_CHECK(cudaMalloc(data, sizeof(DTYPE) * (1 + shape.Count())));
// #else
    CUDA_CHECK(cudaMalloc(data, sizeof(DTYPE) * shape.Count()));
// #endif
    SWITCH_BACK(device);
  }
}

void Tensor::free_mem(DTYPE* data, int rank, int device) {
  if (data == NULL) {
    return;
  }
  CHECK_EQ(current_rank(), rank) << "can't delete memory on another machine";
  if (device < 0) {
    cudaFreeHost(data);
  } else {
    SWITCH_DEVICE(device);
    CUDA_CHECK(cudaFree(data));
    SWITCH_BACK(device);
  }
}

void Tensor::swap_memory(Tensor* other) {
// #ifndef NDEBUG
//   DTYPE* tmp = other->past_the_end_;
//   other->past_the_end_ = past_the_end_;
//   past_the_end_ = tmp;
// #endif
  CHECK_EQ(other->shape_, shape_);
  CHECK_EQ(other->stride_, stride_);
  CHECK_EQ(other->offset_, offset_);
  this->data_.swap(other->data_);
}

void Tensor::slice_from(Tensor* other, const Offset& off, const Shape& shape) {
// #ifndef NDEBUG
//   past_the_end_ = other->past_the_end_;
// #endif
  rank_ = other->rank_;
  device_ = other->device_;
  stride_ = other->stride_;
  data_ = other->data_;
  shape_ = shape;
  offset_ += off;
}

void Tensor::share_from(Tensor* other) {
// #ifndef NDEBUG
//   past_the_end_ = other->past_the_end_;
// #endif
  rank_ = other->rank_;
  device_ = other->device_;
  stride_ = other->stride_;
  data_ = other->data_;
  shape_ = other->shape_;
  offset_ = other->offset_;
}

void Tensor::delete_data() {
  data_.reset();
}

const DTYPE* Tensor::data() const {
  CHECK(data_);
// #ifndef NDEBUG
//   if (device_ < 0) {
//     CHECK_EQ(*past_the_end_, 555.);
//   } else {
//     DTYPE flag = 0;
//     SWITCH_DEVICE(device_);
//     CUDA_CHECK(cudaMemcpy(&flag, past_the_end_, sizeof(DTYPE) * 1,
//             cudaMemcpyDeviceToHost));
//     SWITCH_BACK(device_);
//     CHECK_EQ(flag, 555.);
//   }
// #endif
  return data_.get() + Tensor::offset(offset_, stride_);
}

DTYPE* Tensor::mutable_data() {
  CHECK_EQ(current_rank(), rank_) << "can't access data from a different rank";
  if (!data_) {
    CHECK(is_contiguous());
    DTYPE* ptr;
    Tensor::alloc_mem(&ptr, shape_, rank_, device_);
    data_.reset(ptr, bind(Tensor::free_mem, std::placeholders::_1, rank_,
            device_));
// #ifndef NDEBUG
//     past_the_end_ = data_.get() + shape_.Count();
//     if (device_ < 0) {
//       *past_the_end_ = 555.;
//     } else {
//       DTYPE flag = 555.;
//       SWITCH_DEVICE(device_);
//       CUDA_CHECK(cudaMemcpy(past_the_end_, &flag, sizeof(DTYPE) * 1,
//               cudaMemcpyHostToDevice));
//       SWITCH_BACK(device_);
//     }
// #endif
  }
// #ifndef NDEBUG
//   if (device_ < 0) {
//     CHECK_EQ(*past_the_end_, 555.);
//   } else {
//     DTYPE flag = 0;
//     SWITCH_DEVICE(device_);
//     CUDA_CHECK(cudaMemcpy(&flag, past_the_end_, sizeof(DTYPE) * 1,
//             cudaMemcpyDeviceToHost));
//     SWITCH_BACK(device_);
//     CHECK_EQ(flag, 555.);
//   }
// #endif
  return data_.get() + Tensor::offset(offset_, stride_);
}

bool Tensor::is_contiguous() const {
  return Stride(shape_) == stride_;
}

}
