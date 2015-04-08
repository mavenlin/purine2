// Copyright Lin Min 2014

#ifndef PURINE_SIZE
#define PURINE_SIZE

#include <glog/logging.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>
#include <initializer_list>

using std::equal;
using std::ostream;
using std::vector;
using std::initializer_list;
using std::accumulate;
using std::multiplies;

namespace purine {

class Shape {
 private:
  vector<size_t> shape_;
 public:
  explicit Shape() {}
  inline size_t Dims() const { return shape_.size(); }
  Shape(const Shape& s) {
    shape_ = s.shape_;
  }
  Shape(const initializer_list<size_t> list) {
    shape_ = list;
  }
  /**
   * @brief access shape by index.
   */
  inline const size_t& operator[] (size_t index) const {
    return shape_[index];
  }
  inline size_t& operator[] (size_t index) {
    return shape_[index];
  }
  /**
   * @brief return total size
   */
  inline size_t Count() const {
    return accumulate(shape_.begin(), shape_.end(), 1, multiplies<size_t>());
  }
  /**
   * @brief check if two shapes are equal
   */
  inline bool operator == (const Shape& other) const {
    return Dims() == other.Dims() &&
        equal(shape_.begin(), shape_.end(), other.shape_.begin());
  }
  inline Shape& operator += (const Shape& add) {
    transform(shape_.begin(), shape_.end(), add.shape_.begin(), shape_.begin(),
        std::plus<size_t>());
    return *this;
  }
};

inline Shape operator + (const Shape& shape, const Shape& add) {
  Shape ret = shape;
  return ret += add;
}

class Stride {
 protected:
  vector<size_t> stride_;
 public:
  explicit Stride() {}
  explicit Stride(const Shape& shape) {
    stride_ = vector<size_t>(shape.Dims());
    stride_[shape.Dims() - 1] = 1;
    for (int i = shape.Dims() - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape[i + 1];
    }
  }
  Stride(const initializer_list<size_t> list) {
    stride_ = list;
  }
  inline size_t Dims() const { return stride_.size(); }
  inline const size_t& operator[] (size_t index) const {
    return stride_[index];
  }
  inline size_t& operator[] (size_t index) {
    return stride_[index];
  }
  inline bool operator == (const Stride& other) const {
    return Dims() == other.Dims() &&
        equal(stride_.begin(), stride_.end(), other.stride_.begin());
  }
};

class Offset {
 protected:
  vector<size_t> offset_;
 public:
  Offset() {}
  Offset(const initializer_list<size_t> list) {
    offset_ = list;
  }
  inline size_t Dims() const { return offset_.size(); }
  inline const size_t& operator[] (size_t index) const {
    return offset_[index];
  }
  inline size_t& operator[] (size_t index) {
    return offset_[index];
  }
  inline bool operator == (const Offset& other) const {
    return Dims() == other.Dims() &&
        equal(offset_.begin(), offset_.end(), other.offset_.begin());
  }
  friend Offset operator + (const Offset& offset, const Offset& add);
  inline Offset& operator += (const Offset& add) {
    transform(offset_.begin(), offset_.end(), add.offset_.begin(),
        offset_.begin(), std::plus<size_t>());
    return *this;
  }
};

inline Offset operator + (const Offset& offset, const Offset& add) {
  Offset ret = offset;
  return ret += add;
}

/**
 * @brief print out the shape
 */
inline ostream& operator<< (ostream& os, const Shape& s) {
  os << "(";
  for (int i = 0; i < s.Dims(); ++i) {
    os << s[i];
    if (i != s.Dims()) {
      os <<  ", ";
    }
  }
  os << ")";
  return os;
}

inline ostream& operator<< (ostream& os, const Stride& s) {
  os << "(";
  for (int i = 0; i < s.Dims(); ++i) {
    os << s[i];
    if (i != s.Dims()) {
      os <<  ", ";
    }
  }
  os << ")";
  return os;
}

inline ostream& operator<< (ostream& os, const Offset& s) {
  os << "(";
  for (int i = 0; i < s.Dims(); ++i) {
    os << s[i];
    if (i != s.Dims()) {
      os <<  ", ";
    }
  }
  os << ")";
  return os;
}

}

#endif
