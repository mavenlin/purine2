// Copyright Lin Min 2015
#include "dispatch/blob.hpp"
#include "dispatch/graph_template.hpp"
#include "operations/include/mem_copy.hpp"
#include "operations/include/eltwise.hpp"
#include "dispatch/op_template.hpp"
#include "composite/graph/copy.hpp"
#include "composite/vectorize.hpp"
#include "operations/include/dummy.hpp"

namespace purine {
typedef vector<Blob*> B;
void Copy::setup() {
  static int tag = 0;
  CHECK(bottom_setup_);

  // check top
  if (top_.size() != 0) {
    CHECK_EQ(bottom_.size(), 1);
    CHECK_EQ(top_.size(), 1);
  } else {
    CHECK_EQ(bottom_.size(), 1);
    top_ = vector<Blob*>(bottom_.size());
    if (rank_ == bottom_[0]->rank() &&
        device_ == bottom_[0]->device()) {
      top_[0] = bottom_[0];
    } else {
      top_[0] = create("dest", rank_, device_, bottom_[0]->tensor()->shape());
    }
  }
  // setup internal routes
  if (bottom_[0] == top_[0]) {
    return;
  }
  if (bottom_[0]->rank() == top_[0]->rank()) { // on the same machine
    string thread;
    if (bottom_[0]->device() >= 0) {
      thread = "outbound";
    } else if (top_[0]->device() >= 0) {
      thread = "inbound";
    } else {
      thread = "main";
    }
    B{ bottom_[0] } >> *create<MemCopy>("memcopy", thread,
        MemCopy::param_tuple()) >> B{ top_[0] };
  } else { // on different machines
    Blob* src;
    if (bottom_[0]->device() < 0) {
      src = bottom_[0];
    } else {
      src = create("src_cpu", bottom_[0]->rank(), -1,
          bottom_[0]->tensor()->shape());
      B{ bottom_[0] } >> *create<MemCopy>("memcopy", "outbound",
          MemCopy::param_tuple()) >> B{ src };
    }
    Blob* dest;
    if (top_[0]->device() < 0) {
      dest = top_[0];
    } else {
      dest = create("dest_cpu", top_[0]->rank(), -1, top_[0]->tensor()->shape());
      B{ dest } >> *create<MemCopy>("memcopy", "inbound",
          MemCopy::param_tuple()) >> B{ top_[0] };
    }
    B{ src } >> *create<Isend>("isend", src->rank(), src->device(), "main",
        Isend::param_tuple(tag, dest->rank()));
    *create<Irecv>("irecv", dest->rank(), dest->device(), "main",
        Irecv::param_tuple(tag++, src->rank())) >> B{ dest };
  }
}

void Distribute::setup() {
  CHECK(bottom_setup_);
  // check top
  if (top_.size() != 0) {
    for (Blob* t : top_) {
      CHECK_EQ(t->tensor()->shape(), bottom_[0]->tensor()->shape());
    }
  } else {
    top_ = vector<Blob*>(rank_device.size());
    for (int i = 0; i < rank_device.size(); ++i) {
      if (rank_device[i].first == bottom_[0]->rank() &&
          rank_device[i].second == bottom_[0]->device()) {
        top_[i] = bottom_[0];
      } else {
        top_[i] = create("dist_dest_" + to_string(i), rank_device[i].first,
            rank_device[i].second, bottom_[0]->tensor()->shape());
      }
    }
  }
  map<int, vector<Blob*> > blobs;
  map<int, Blob*> routes;
  for (Blob* b : top_) {
    // if top is on the same machine as bottom
    if (b->rank() == bottom_[0]->rank()) {
      if (b != bottom_[0]) {
        bottom_ >> *createAny<Copy>("dist_same_machine",
            Copy::param_tuple()) >> vector<Blob*>{ b };
      }
      continue;
    }
    // not on the same machine
    if (blobs.count(b->rank()) == 1) {
      blobs[b->rank()].push_back(b);
    } else {
      blobs[b->rank()] = { b };
    }
    // if top is on CPU, it can be the route
    if (b->device() < 0 && routes.count(b->rank()) == 0) {
      routes[b->rank()] = create("route", b->shared_tensor());
      bottom_ >> *createAny<Copy>("dist_to_router", Copy::param_tuple())
      >> vector<Blob*>{ routes[b->rank()] };
    }
  }
  for (auto kv : blobs) {
    if (routes.count(kv.first) == 0) {
      routes[kv.first] = create("route", kv.first, -1,
          bottom_[0]->tensor()->shape());
      bottom_ >> *createAny<Copy>("dist_to_router", Copy::param_tuple())
      >> vector<Blob*>{ routes[kv.first] };
    }
    Blob* route = routes[kv.first];
    for (Blob* top : kv.second) {
      if (route->tensor() == top->tensor()) {
        vector<Blob*>{ route } >> *create<Dummy>("dummy_redist", route->rank(),
            route->device(), "main", Dummy::param_tuple())
                                      >> vector<Blob*>{ top };
      } else {
        vector<Blob*>{ route } >> *createAny<Copy>("redist",
            Copy::param_tuple()) >> vector<Blob*>{ top };
      }
    }
  }
}

void Aggregate::setup() {
  CHECK(bottom_setup_);
  // check top
  if (top_.size() != 0) {
    CHECK_EQ(top_.size(), 1);
    for (Blob* b : bottom_) {
      CHECK_EQ(b->tensor()->shape(), top_[0]->tensor()->shape());
    }
  } else {
    top_ = {
      create("top", rank_, device_, bottom_[0]->tensor()->shape())
    };
  }

  map<int, Aggregate*> local_aggs;
  map<int, vector<Blob*> > local_blobs;
  vector<Blob*> dest_blobs;
  for (int i = 0; i < bottom_.size(); ++i) {
    if (bottom_[i]->rank() != top_[0]->rank() &&
        local_aggs.count(bottom_[i]->rank()) == 0) {
      local_aggs[bottom_[i]->rank()] = createAny<Aggregate>("local_agg",
          param_tuple(Type::SUM, bottom_[i]->rank(), -1));
      local_blobs[bottom_[i]->rank()] = { bottom_[i] };
    } else if (bottom_[i]->rank() != top_[0]->rank()) {
      local_blobs[bottom_[i]->rank()].push_back(bottom_[i]);
    } else {
      dest_blobs.push_back(bottom_[i]);
    }
  }

  for (auto kv : local_aggs) {
    local_blobs[kv.first] >> *(kv.second);
  }
  vector<Blob*> agged(local_aggs.size());
  transform(local_aggs.begin(), local_aggs.end(), agged.begin(),
      [](std::pair<const int, Aggregate*>& agg)->Blob* {
        return agg.second->top()[0];
      });
  dest_blobs.insert(dest_blobs.end(), agged.begin(), agged.end());

  auto copy = createAny<Vectorize<Copy> >("agg_to_dest",
      vector<Copy::param_tuple>(dest_blobs.size(),
          Copy::param_tuple(top_[0]->rank(), top_[0]->device())));
  vector<vector<Blob*> >{ dest_blobs } >> *copy;
  vector<Blob*> copied = copy->top()[0];

  if (agg_type_ == Aggregate::SUM || bottom_.size() == 1) {
    if (copied.size() > 1){
      copied >> *create<Sum>("sum", top_[0]->rank(), top_[0]->device(),
          "main", Sum::param_tuple()) >> top_;
    } else {
      top_[0]->share_from(copied[0]);
      copied >> *create<Dummy>("dummy_sum", "main",
          Dummy::param_tuple()) >> top_;
    }
  } else {
    DTYPE scale = 1. / DTYPE(bottom_.size());
    if (copied.size() > 1) {
      Blob* sum = create("summed", top_[0]->shared_tensor());
      copied >> *create<Sum>("sum", top_[0]->rank(), top_[0]->device(),
          "main", Sum::param_tuple()) >> vector<Blob*>{ sum } >>
          *create<Scale>("scale", top_[0]->rank(), top_[0]->device(), "main",
              Scale::param_tuple(scale)) >> top_;
    } else {
      top_[0]->share_from(copied[0]);
      copied >> *create<Scale>("scale", top_[0]->rank(), top_[0]->device(),
          "main", Scale::param_tuple(scale)) >> top_;
    }
  }
}

}
