// Copyright Lin Min 2015
#include <numeric>
#include "composite/graph/split.hpp"
#include "operations/include/dummy.hpp"

namespace purine {

void Split::setup() {
  CHECK(bottom_setup_);
  CHECK_EQ(bottom_.size(), 1);
  if (top_.size() != 0) {
    dims = vector<size_t>(top_.size());
    if (dim == DIM::NUM) {
      for (int i = 0; i < top_.size(); ++i) {
        dims[i] = top_[i]->tensor()->shape()[0];
      }
    } else {
      for (int i = 0; i < top_.size(); ++i) {
        dims[i] = top_[i]->tensor()->shape()[1];
      }
    }
  } else {
    top_ = vector<Blob*>(dims.size());
    for (int i = 0; i < top_.size(); ++i) {
      top_[i] = create("top", {0, 0, 0, 0});
    }
  }
  int sum = std::accumulate(dims.begin(), dims.end(), 0);
  if (dim == DIM::NUM) {
    CHECK_EQ(sum, bottom_[0]->tensor()->shape()[0]);
  } else {
    CHECK_EQ(sum, bottom_[0]->tensor()->shape()[1]);
  }

  if (current_rank() == rank_) {
    // create sliced blobs from bottom
    Shape bottom_shape = bottom_[0]->tensor()->shape();
    bottom_[0]->tensor()->mutable_data();
    size_t off = 0;
    if (dim == DIM::NUM) {
      for (int i = 0; i < dims.size(); ++i) {
        Shape tmp_shape = { dims[i], bottom_shape[1], bottom_shape[2],
                          bottom_shape[3] };
        Offset tmp_offset = { off, 0, 0, 0 };
        off += dims[i];
        top_[i]->tensor()->slice_from(bottom_[0]->tensor(),
            tmp_offset, tmp_shape);
      }
    } else {
      for (int i = 0; i < dims.size(); ++i) {
        Shape tmp_shape = { bottom_shape[0], dims[i], bottom_shape[2],
                          bottom_shape[3] };
        Offset tmp_offset = { 0, off, 0, 0 };
        off += dims[i];
        top_[i]->tensor()->slice_from(bottom_[0]->tensor(),
            tmp_offset, tmp_shape);
      }
    }
  }
  // create op
  Op<Dummy>* dummy = create<Dummy>("slice", "main", Dummy::param_tuple());
  bottom_ >> *dummy >> top_;
}

}
