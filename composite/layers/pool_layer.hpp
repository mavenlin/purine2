// Copyright Lin Min 2015
#ifndef PURINE_POOL_LAYER
#define PURINE_POOL_LAYER

#include "composite/layer.hpp"
#include "operations/include/pool.hpp"

namespace purine {

typedef vector<Blob*> B;

class PoolLayer : public Layer {
 protected:
  string method;
  size_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
  Pool::param_tuple args_;
 public:
  typedef Pool::param_tuple param_tuple;
  PoolLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(method, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
        = args;
    args_ = args;
  }
  virtual ~PoolLayer() override {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Shape bottom_shape = bottom_[0]->tensor()->shape();
    size_t out_h = static_cast<int>(ceil(static_cast<float>(bottom_shape[2]
                + 2 * pad_h - kernel_h) / stride_h)) + 1;
    size_t out_w = static_cast<int>(ceil(static_cast<float>(bottom_shape[3]
                + 2 * pad_w - kernel_w) / stride_w)) + 1;
    Shape expect_top_shape = { bottom_shape[0], bottom_shape[1], out_h, out_w };

    // check top
    if (top_.size() != 0) {
      CHECK_EQ(top_.size(), 2);
      for (auto top : top_) {
        CHECK_EQ(top->tensor()->shape(), expect_top_shape);
      }
    } else {
      top_ = {
        create("top", expect_top_shape),
        create("top_diff", expect_top_shape)
      };
    }
    // create ops
    Op<Pool>* pool_up = create<Pool>("pool_up", "main", args_);
    Op<PoolDown>* pool_down = create<PoolDown>("pool_down", "main", args_);

    // forward
    B{ bottom_[0] } >> *pool_up >> B{ top_[0] };
    // backward
    B{ top_[1], top_[0], bottom_[0] } >> *pool_down >> B{ bottom_[1] };
  }
};

}

#endif
