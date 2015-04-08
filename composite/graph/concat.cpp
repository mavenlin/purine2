// Copyright Lin Min 2015
#include "composite/graph/concat.hpp"
#include "operations/include/dummy.hpp"

namespace purine {

void Concat::setup() {
  CHECK(bottom_setup_);
  CHECK_GT(bottom_.size(), 1);
  // check bottom shape
  Shape s = bottom_[0]->tensor()->shape();
  Shape expected_top_shape;
  size_t sum = 0;
  if (dim == Split::DIM::NUM) {
    for (Blob* b : bottom_) {
      CHECK_EQ(b->tensor()->shape()[1], s[1]);
      CHECK_EQ(b->tensor()->shape()[2], s[2]);
      CHECK_EQ(b->tensor()->shape()[3], s[3]);
      sum += b->tensor()->shape()[0];
    }
    expected_top_shape = { sum, s[1], s[2], s[3] };
  } else {
    for (Blob* b : bottom_) {
      CHECK_EQ(b->tensor()->shape()[0], s[0]);
      CHECK_EQ(b->tensor()->shape()[2], s[2]);
      CHECK_EQ(b->tensor()->shape()[3], s[3]);
      sum += b->tensor()->shape()[1];
    }
    expected_top_shape = { s[0], sum, s[2], s[3] };
  }

  // check top
  if (top_.size() != 0) {
    CHECK_EQ(expected_top_shape, top_[0]->tensor()->shape());
  } else {
    top_ = {
      create("top", expected_top_shape)
    };
  }

  if (current_rank() == rank_) {
    top_[0]->tensor()->mutable_data();
    // create sliced blobs from top
    size_t off = 0;
    if (dim == Split::DIM::NUM) {
      for (int i = 0; i < bottom_.size(); ++i) {
        bottom_[i]->tensor()->slice_from(top_[0]->tensor(), { off, 0, 0, 0},
            bottom_[i]->tensor()->shape());
        off += bottom_[i]->tensor()->shape()[0];
      }
    } else {
      for (int i = 0; i < bottom_.size(); ++i) {
        bottom_[i]->tensor()->slice_from(top_[0]->tensor(), { 0, off, 0, 0},
            bottom_[i]->tensor()->shape());
        off += bottom_[i]->tensor()->shape()[1];
      }
    }
  }
  // create op
  Op<Dummy>* dummy = create<Dummy>("concat", "main", Dummy::param_tuple());
  bottom_ >> *dummy >> top_;
}

}
