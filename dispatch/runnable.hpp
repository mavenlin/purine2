// Copyright Lin Min 2015
#ifndef PURINE_RUNNABLE
#define PURINE_RUNNABLE

#include <condition_variable>
#include <map>
#include <mutex>

#include "common/loop.hpp"
#include "dispatch/graph.hpp"
#include "dispatch/node.hpp"

using std::mutex;
using std::unique_lock;
using std::condition_variable;

namespace purine {

class Runnable : public Graph {
 public:

  class SinkCounter {
   private:
    condition_variable cv_;
    mutex mtx_;
    int count_ = 0;
   public:
    SinkCounter() {
    }
    int operator++ () {
      std::unique_lock<std::mutex> lck(mtx_);
      ++count_;
      cv_.notify_all();
    }
    bool operator== (int num) {
      std::unique_lock<std::mutex> lck(mtx_);
      while (count_ != num) {
        cv_.wait(lck);
      }
      count_ = 0;
      return true;
    }
  };

 protected:
  vector<Node*> cached_sources_;
  vector<Node*> cached_sinks_;
  bool prepared_ = false;
  void prepare_once();
  SinkCounter sink_counter_;
  mutex mutex_;
  map<tuple<int, string>, shared_ptr<LoopInterface> > loops_;
 public:
  explicit Runnable(int rank = 0, int device = 0);
  virtual ~Runnable();

  inline SinkCounter& sink_counter() { return sink_counter_; }

  virtual vector<Node*> nodes() override;
  LoopInterface& task_loop(int device, const string& thread);
  virtual void run();
  virtual void run_async();
  virtual void sync();
  vector<vector<string> > print();
};

}

#endif
