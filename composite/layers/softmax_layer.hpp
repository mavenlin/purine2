// Copyright Lin Min 2015
#ifndef PURINE_SOFTMAX_LAYER
#define PURINE_SOFTMAX_LAYER

#include "composite/layer.hpp"
#include "operations/include/softmax.hpp"

namespace purine {

class SoftmaxLayer : public Layer {
 protected:
  string mode;
  bool inplace;
 public:
  typedef tuple<string> param_tuple;
  SoftmaxLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(mode) = args;
  }
  virtual ~SoftmaxLayer() {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Shape bottom_shape = bottom_[0]->tensor()->shape();

    // check top
    if (top_.size() != 0) {
      CHECK_EQ(top_.size(), 2);
      for (auto top : top_) {
        CHECK_EQ(top->tensor()->shape(), bottom_shape);
      }
    } else {
      if (!inplace) {
        top_ = {
          create("top", bottom_shape),
          create("top_diff", bottom_shape)
        };
      } else {
        top_ = {
          create("top", bottom_[0]->shared_tensor()),
          create("top_diff", bottom_[0]->shared_tensor())
        };
      }
    }

    // create ops
    Op<Softmax>* softmax_up = create<Softmax>("softmax_up", "main",
        make_tuple(mode));
    Op<SoftmaxDown>* softmax_down = create<SoftmaxDown>("softmax_down", "main",
        make_tuple(mode));

    // forward
    B{ bottom_[0] } >> *softmax_up >> B{ top_[0] };
    // backward
    B{ top_[1], top_[0] } >> *softmax_down >> B{ bottom_[1] };
  }
};

}

#endif
