// Copyright Lin Min 2015
#ifndef PURINE_SPLIT_LAYER
#define PURINE_SPLIT_LAYER

#include "composite/layer.hpp"
#include "composite/graph/split.hpp"
#include "composite/graph/concat.hpp"

namespace purine {

// delete set top, SplitLayer generates top
class SplitLayer;
const vector<Blob*>& operator >> (SplitLayer& split,
    const vector<Blob*>& top) = delete;

class SplitLayer : public Layer {
 protected:
  Split::DIM dim;
  vector<size_t> dims;
 public:
  typedef tuple<Split::DIM, vector<size_t> > param_tuple;
  SplitLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(dim, dims) = args;
  }
  virtual ~SplitLayer() {}
 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Shape bottom_shape = bottom_[0]->tensor()->shape();
    Split* split = createGraph<Split>("split", Split::param_tuple(dim), dims);
    bottom_data() >> *split;
    top_.insert(top_.end(), split->top().begin(), split->top().end());
    // concat
    // create top_diff
    for (Blob* top : split->top()) {
      top_.push_back(create("top_diff", top->tensor()->shape()));
    }
    Concat* concat = createGraph<Concat>("concat", Concat::param_tuple(dim));
    top_diff() >> *concat >> bottom_diff();
  }
};

}

#endif
