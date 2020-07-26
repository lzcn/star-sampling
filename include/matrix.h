/** @file
 *	@author Zhi Lu
 *  @brief Class for Matrices
 */
#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cstring>
#include <fstream>
#include <iostream>

#include <unordered_map>
#include "common.h"
#include "random.h"


/**
 * 	@brief Point2d for (x,y)
 */
class Point2d {
 public:
  Point2d(uint i, uint j);
  ~Point2d(){};
  bool operator<(const Point2d &p) const;
  bool operator==(const Point2d &p) const;
  bool operator>(const Point2d &p) const;

 public:
  uint x;
  uint y;
};

/** @brief Point3d for (x,y,z)
 *
 */
class Point3d {
 public:
  Point3d(uint i, uint j, uint k);
  ~Point3d(){};
  bool operator<(const Point3d &p) const;
  bool operator==(const Point3d &p) const;
  bool operator>(const Point3d &p) const;

 public:
  uint x;
  uint y;
  uint z;
};

class PointNd {
 public:
  PointNd(uint *p, uint n);
  ~PointNd(){};
  bool operator<(const PointNd &p) const;

 public:
  uint length;
  uint *index;
};

typedef std::pair<Point2d, double> Point2dValuePair;
typedef std::pair<Point3d, double> Point3dValuePair;
typedef std::pair<PointNd, double> PointNdValuePair;

template <typename T>
bool compgt(const T &p, const T &q) {
  return (p.second > q.second);
}

template <typename T>
bool complt(const T &p, const T &q) {
  return (p.second < q.second);
}

struct _Hash2d {
  size_t operator()(const Point2d &k) const {
    size_t h1 = std::hash<uint>()(k.x);
    size_t h2 = std::hash<uint>()(k.y);
    return (h2 ^ (h1 << 1));
  }
};
struct _Hash3d {
  size_t operator()(const Point3d &k) const {
    size_t h1 = std::hash<uint>()(k.x);
    size_t h2 = std::hash<uint>()(k.y);
    size_t h3 = std::hash<uint>()(k.z);
    return (h1 ^ (h2 << 1)) ^ h3;
  }
};

struct _Equal2d {
  bool operator()(const Point2d &p, const Point2d &q) const {
    return (p.x == q.x) && (p.y == q.y);
  }
};

struct _Equal3d {
  bool operator()(const Point3d &p, const Point3d &q) const {
    return (p.x == q.x) && (p.y == q.y) && (p.z == q.z);
  }
};

typedef std::unordered_map<Point2d, double, _Hash2d, _Equal2d> Point2dValueMap;
typedef std::unordered_map<Point3d, double, _Hash3d, _Equal3d> Point3dValueMap;

enum MATRIX_INIT_TYPE : unsigned int {
  MATRIX_INIT_RANDU,  //< sampled from uniform distribution
  MATRIX_INIT_RANDN,  //< sampled normal distribution
  MATRIX_INIT_ZEROS,  //< assign all zeros
  MATRIX_INIT_ONES,   //< assign all ones
};

enum MATRIX_SUM_TYPE : unsigned int {
  MATRIX_NONE_SUM,  ///< no one norm will be computed
  MATRIX_COL_SUM,   ///< compute the one norm of each column
  MATRIX_ROW_SUM,   ///< compute the one norm of each row asum
  MATRIX_FULL_SUM,  ///< compute all one norm
};

class Mat {
 public:
  Mat();
  Mat(uint r, uint c);
  Mat(uint r, uint c, double *ptr);
  Mat(uint r, uint c, double *ptr, MATRIX_SUM_TYPE type);
  ~Mat();
  double &operator()(uint i, uint j);
  double operator()(uint i, uint j) const;
  double norm();

  void init(MATRIX_INIT_TYPE type, double mean, double stdev);
  void load(std::string filename);
  void save(std::string filename);

  double GetElement(uint i, uint j);
  double GetColSum(uint column);
  void transpose(double *ptr);
  void accumulation(double *ptr);

  uint randRow(uint n);
  uint randCol(uint m);
  uint randRow(uint n, double sum);
  uint randCol(uint m, double sum);

 public:
  uint n_row;
  uint n_col;
  double *value;
  double *col_abs_sum;
  double *row_abs_sum;

 private:
  MATRIX_SUM_TYPE sum_type_;
  bool self_value_;
};

class DenseMat {
 public:
  DenseMat();
  DenseMat(uint r_dim, uint c_dim);
  DenseMat(uint r_dim, uint c_dim, double *ptr);
  ~DenseMat();
  double &operator()(uint i, uint j);
  double operator()(uint i, uint j) const;
  uint randRow(uint n, double sum);
  uint randCol(uint m, double sum);
  void init(MATRIX_INIT_TYPE type, double mean, double stdev);
  double norm();
  void load(std::string filename);
  void save(std::string filename);

 public:
  uint n_row, n_col;
  double *value;

 private:
  bool self_value_;
};

class FactorMat {
 public:
  uint n_row;
  uint n_col;
  double *value;
  double *col_abs_sum;
  double *row_abs_sum;
  FactorMat(uint r, uint c);
  FactorMat(uint r, uint c, double *ptr);
  FactorMat(uint r, uint c, double *ptr, MATRIX_SUM_TYPE type);
  ~FactorMat();
  double GetElement(uint i, uint j);
  double GetColSum(uint column);
  double &operator()(uint i, uint j) const { return value[j * n_row + i]; }
  void transpose(double *ptr);
  /**
   *  @brief compute the absolute accumulation of each column
   *
   * \f$ m_{ij} = \sum_{s=0}^{s=j}|p_{is}| \f$
   */
  void accumulation(double *ptr);
  uint randRow(uint n);
  /** \biref	return n with probability abs(M_{m,n})/SumofRow(m)
   *
   */
  uint randCol(uint m);

 private:
  MATRIX_SUM_TYPE sum_type_;
  bool self_value_;
};

/**
 *  @brief insert p into a ascending sequence
 *				 if p less than the last element
 */
void doInsertReverse(double v, std::list<double> &v_list);
/**
 *  @brief insert p into a descending sequence
 *				 if p is greater than the last element
 */
void doInsert(double v, std::list<double> &v_list);

void doInsert(double v, std::list<double> &v_list, Point3d &p,
              std::list<Point3d> &p_list);

/**
 *  @brief compute \f$\sum_{r}{a_{x,r}*b_{y,r}}\f$
 */
double MatrixRowMul(const Point2d &v, FactorMat &A, FactorMat &B);
/**
 *  @brief compute \f$\sum_{r}{a_{x,r}*b_{y,r}*c_{z,r}}\f$
 */
double MatrixRowMul(const Point3d &v, FactorMat &A, FactorMat &B, FactorMat &C);
/**
 *  @brief compute \f$\sum_{r}{a_{r,x}*b_{r,y}}\f$
 */
double MatrixColMul(const Point2d &v, FactorMat &A, FactorMat &B);
/**
 *  @brief compute \f$\sum_{r}{a_{r,x}*b_{r,y}*c_{r,z}}\f$
 */
double MatrixColMul(const Point3d &v, FactorMat &A, FactorMat &B, FactorMat &C);
/**
 *  @brief compute \f$\sum_{r}{a_{r,i}*b_{r,j}}\f$
 */
double MatrixColMul(FactorMat &A, FactorMat &B, uint i, uint j);
/**
 *  @brief compute \f$\sum_{r}{a_{r,i}*b_{r,j}*c_{r,k}}\f$
 */
double MatrixColMul(FactorMat &A, FactorMat &B, FactorMat &C, uint i, uint j,
                    uint k);
double vectors_mul(const PointNd &p, std::vector<FactorMat *> &vMat);
double vectors_mul(const Point2d &v, FactorMat &A, FactorMat &B);
double vectors_mul(const Point3d &v, FactorMat &A, FactorMat &B, FactorMat &C);



#endif  // __MATRIX_H__
