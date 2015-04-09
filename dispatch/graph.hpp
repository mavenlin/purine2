#ifndef PURINE_GRAPH
#define PURINE_GRAPH

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "operations/operation.hpp"
#include "operations/tensor.hpp"

namespace purine {

using std::map;
using std::atomic;
using std::string;
using std::vector;
using std::shared_ptr;

/**
 * Inheritance tree:
 * Graph -> Node -> Op
 *  \         \
 *   \         ---> Blob
 *    \
 *     --> Connectable -> Layer
 *      \
 *       \
 *        -----> Runnable
 */
class Node;
template <typename O> class Op;
class Blob;
class Layer;
class Connectable;
class Runnable;

class Graph {
  friend class Runnable;
 protected:
  string cached_name_;
  Graph* cached_root_;

  int rank_;
  int device_;
  vector<shared_ptr<Graph> > subgraphs_;
  map<const Graph*, string> graph_name_;
  Graph* parent_ = NULL;
  virtual void setup() {}
 public:
  explicit Graph(int rank = 0, int device = 0);
  virtual ~Graph();

  inline int rank() const { return rank_; }
  inline int device() const { return device_; }
  string name() const;
  void prune(const vector<Node*>& sinks);

  virtual vector<Node*> nodes();
  vector<Node*> sources();
  vector<Node*> sinks();
  vector<vector<string> > print();

  DTYPE memory_cost_cpu();
  DTYPE memory_cost_gpu();

  void delete_subgraph(Graph* g);

  // create op
  template <typename O>
  Op<O>* create(const string& name, int rank, int device, const string& thread,
      const typename O::param_tuple& param);
  template <typename O>
  Op<O>* create(const string& name, const string& thread,
      const typename O::param_tuple& param);

  template <typename G, typename... Args>
  G* createGraph(const string& name, int rank, int device, const Args&... args);
  template <typename G, typename... Args>
  G* createGraph(const string& name, const Args&... args);

  template <typename G, typename... Args>
  G* createAny(const string& name, const Args&... args);

  // create blob
  Blob* create(const string& name, int rank, int device, const Shape& shape);
  Blob* create(const string& name, const Shape& shape);
  Blob* create(const string& name, shared_ptr<Tensor> tensor);
  inline string cached_name() const { return cached_name_; }
};

}

#endif
