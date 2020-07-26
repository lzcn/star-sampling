#include "matrix.h"

#include <cmath>
#include <cstdio>

Point2d::Point2d(uint i, uint j) : x{i}, y{j} {}

bool Point2d::operator<(const Point2d &p) const {
  if (x < p.x) {
    return true;
  } else if (x == p.x) {
    if (y < p.y) return true;
  }
  return false;
}

bool Point2d::operator==(const Point2d &p) const {
  if (x == p.x && y == p.y) return true;
  return false;
}

bool Point2d::operator>(const Point2d &p) const {
  if (x > p.x) {
    return true;
  } else if (x == p.x) {
    if (y > p.y) return true;
  }
  return false;
}

Point3d::Point3d(uint i, uint j, uint k) : x{i}, y{j}, z{k} {}

bool Point3d::operator<(const Point3d &p) const {
  if (x < p.x) {
    return true;
  }
  if (x == p.x) {
    if (y < p.y) return true;
  }
  if (x == p.x && y == p.y) {
    if (z < p.z) return true;
  }
  return false;
}

bool Point3d::operator==(const Point3d &p) const {
  if (x == p.x && y == p.y && z == p.z) return true;
  return false;
}

bool Point3d::operator>(const Point3d &p) const {
  if (x > p.x) {
    return true;
  }
  if (x == p.x) {
    if (y > p.y) return true;
  }
  if (x == p.x && y == p.y) {
    if (z > p.z) return true;
  }
  return false;
}

PointNd::PointNd(uint *v, uint n) {
  index = v;
  length = n;
}

bool PointNd::operator<(const PointNd &p) const {
  for (uint i = 0; i < length; ++i) {
    if (index[i] < p.index[i]) {
      return true;
    } else if (index[i] > p.index[i]) {
      return false;
    }
  }
  if (index[length - 1] < p.index[length - 1])
    return true;
  else
    return false;
}

Mat::Mat() {
  n_row = 0;
  n_col = 0;
  value = nullptr;
  col_abs_sum = nullptr;
  row_abs_sum = nullptr;
  sum_type_ = MATRIX_NONE_SUM;
  self_value_ = false;
}

Mat::Mat(uint r, uint c) {
  n_row = r;
  n_col = c;
  sum_type_ = MATRIX_NONE_SUM;
  col_abs_sum = nullptr;
  row_abs_sum = nullptr;
  value = (double *)malloc(n_row * n_col * sizeof(double));
  memset(value, 0, n_row * n_col * sizeof(double));
  self_value_ = true;
}

Mat::Mat(uint r, uint c, double *ptr) {
  n_row = r;
  n_col = c;
  sum_type_ = MATRIX_NONE_SUM;
  col_abs_sum = nullptr;
  row_abs_sum = nullptr;
  value = ptr;
  self_value_ = false;
}

Mat::Mat(uint r, uint c, double *ptr, MATRIX_SUM_TYPE type) {
  n_row = r;
  n_col = c;
  value = ptr;
  self_value_ = false;
  sum_type_ = type;
  switch (type) {
    case MATRIX_FULL_SUM:
      col_abs_sum = (double *)malloc(n_col * sizeof(double));
      // memset(col_abs_sum, 0, n_col * sizeof(double));
      row_abs_sum = (double *)malloc(n_row * sizeof(double));
      // memset(row_abs_sum, 0, n_row * sizeof(double));
      for (uint i = 0; i < n_col; ++i) {
        for (uint j = 0; j < n_row; ++j) {
          col_abs_sum[i] += abs(value[i * n_row + j]);
          row_abs_sum[j] += abs(value[i * n_row + j]);
        }
      }
      break;
    case MATRIX_NONE_SUM:
      col_abs_sum = nullptr;
      row_abs_sum = nullptr;
      break;
    case MATRIX_COL_SUM:
      col_abs_sum = (double *)malloc(n_col * sizeof(double));
      memset(col_abs_sum, 0, n_col * sizeof(double));
      row_abs_sum = nullptr;
      for (uint i = 0; i < n_col; ++i) {
        for (uint j = 0; j < n_row; ++j) {
          col_abs_sum[i] += abs(value[i * n_row + j]);
        }
      }
      break;
    case MATRIX_ROW_SUM:
      col_abs_sum = nullptr;
      row_abs_sum = (double *)malloc(n_row * sizeof(double));
      memset(row_abs_sum, 0, n_row * sizeof(double));
      for (uint i = 0; i < n_col; ++i) {
        for (uint j = 0; j < n_row; ++j) {
          row_abs_sum[j] += abs(value[i * n_row + j]);
        }
      }
      break;
    default:
      break;
  }
}

Mat::~Mat() {
  if (self_value_) free(value);
  switch (sum_type_) {
    case MATRIX_COL_SUM:
      free(col_abs_sum);
      break;
    case MATRIX_ROW_SUM:
      free(row_abs_sum);
      break;
    case MATRIX_FULL_SUM:
      free(col_abs_sum);
      free(row_abs_sum);
      break;
    default:
      break;
  }
}

double &Mat::operator()(uint i, uint j) { return value[j * n_row + i]; }
double Mat::operator()(uint i, uint j) const { return value[j * n_row + i]; }

double Mat::norm() {
  double result = 0.0;
  for (uint i = 0; i < n_col * n_row; i++) {
    result += value[i] * value[i];
  }
  return result;
}

void Mat::init(MATRIX_INIT_TYPE type, double mean, double stdev) {
  switch (type) {
    case MATRIX_INIT_RANDN:
      for (uint i = 0; i < n_col * n_row; ++i) {
        // normal distribution with \mu = mean,\sigma^2 = stdev
        value[i] = rnd::randn(mean, stdev);
      }
      break;
    case MATRIX_INIT_RANDU:
      for (uint i = 0; i < n_col * n_row; ++i) value[i] = rnd::randu();
      break;
    case MATRIX_INIT_ONES:
      memset(value, 1, n_col * n_row * sizeof(double));
      break;
    case MATRIX_INIT_ZEROS:
      memset(value, 0, n_col * n_row * sizeof(double));
      break;
    default:
      break;
  }
}

void Mat::load(std::string filename) {
  std::ifstream in_file(filename.c_str());
  if (!in_file.is_open()) {
    throw "Unable to open file " + filename;
  }
  for (uint c = 0; c < n_col; c++) {
    for (uint r = 0; r < n_row; r++) {
      double v;
      in_file >> v;
      value[c * n_row + r] = v;
    }
  }
  in_file.close();
}

void Mat::save(std::string filename) {
  std::ofstream out_file(filename.c_str());
  if (out_file.is_open()) {
    for (uint c = 0; c < n_col; c++) {
      for (uint r = 0; r < n_row; r++) {
        out_file << value[c * n_row + r] << "\t";
      }
      out_file << std::endl;
    }
    out_file.close();
  } else {
    std::cout << "Unable to open file " << filename;
  }
}

double Mat::GetElement(uint i, uint j) { return value[j * n_row + i]; }

double Mat::GetColSum(uint n) { return col_abs_sum[n]; }

void Mat::transpose(double *ptr) {
  for (uint r = 0; r < n_row; ++r) {
    for (uint c = 0; c < n_col; ++c) {
      value[c * n_row + r] = ptr[r * n_col + c];
    }
  }
}

void Mat::accumulation(double *ptr) {
  for (uint c = 0; c < n_col; ++c) {
    double sum = 0.0;
    size_t offset = c * n_row;
    for (uint r = 0; r < n_row; ++r) {
      sum += abs(ptr[offset + r]);
      value[c * n_row + r] = sum;
    }
  }
}

uint Mat::randRow(uint n) {
  double x, temp;
  x = col_abs_sum[n] * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint i = 0; i < n_row; ++i) {
    temp += abs(value[i + n * n_row]);
    if (x <= temp) {
      return i;
    }
  }
  return (n_row - 1);
}
uint Mat::randCol(uint m) {
  double x, temp;
  x = row_abs_sum[m] * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint j = 0; j < n_col; ++j) {
    temp += abs(value[m + j * n_row]);
    if (x <= temp) {
      return j;
    }
  }
  return (n_col - 1);
}

uint Mat::randRow(uint n, double sum) {
  double x, temp;
  x = sum * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint i = 0; i < n_row; ++i) {
    temp += abs(value[i + n * n_row]);
    if (x <= temp) {
      return i;
    }
  }
  return (n_row - 1);
}

uint Mat::randCol(uint m, double sum) {
  double x, temp;
  x = sum * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint j = 0; j < n_col; ++j) {
    temp += abs(value[m + j * n_row]);
    if (x <= temp) {
      return j;
    }
  }
  return (n_col - 1);
}

DenseMat::~DenseMat() {
  if (self_value_) free(value);
}

DenseMat::DenseMat() {
  n_row = 0;
  n_col = 0;
  value = nullptr;
  self_value_ = false;
}
DenseMat::DenseMat(uint r_dim, uint c_dim) {
  n_row = r_dim;
  n_col = c_dim;
  value = (double *)malloc(n_row * n_col * sizeof(double));
  memset(value, 0, n_row * n_col * sizeof(double));
  self_value_ = true;
}
DenseMat::DenseMat(uint r_dim, uint c_dim, double *ptr) {
  n_row = r_dim;
  n_col = c_dim;
  value = ptr;
  self_value_ = false;
}

void DenseMat::init(MATRIX_INIT_TYPE TYPE, double mean, double stdev) {
  switch (TYPE) {
    case MATRIX_INIT_RANDN:
      for (uint i = 0; i < n_col * n_row; ++i) {
        // normal distribution with \mu = mean,\sigma^2 = stdev
        value[i] = rnd::randn(mean, stdev);
      }
      break;
    case MATRIX_INIT_RANDU:
      for (uint i = 0; i < n_col * n_row; ++i) value[i] = rnd::randu();
      break;
    case MATRIX_INIT_ONES:
      memset(value, 1, n_col * n_row * sizeof(double));
      break;
    case MATRIX_INIT_ZEROS:
      memset(value, 0, n_col * n_row * sizeof(double));
      break;
    default:
      break;
  }
}

double &DenseMat::operator()(uint i, uint j) { return value[j * n_row + i]; }
double DenseMat::operator()(uint i, uint j) const {
  return value[j * n_row + i];
}

uint DenseMat::randRow(uint n, double sum) {
  double x, temp;
  x = sum * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint i = 0; i < n_row; ++i) {
    temp += abs(value[i + n * n_row]);
    if (x <= temp) {
      return i;
    }
  }
  return (n_row - 1);
}

uint DenseMat::randCol(uint m, double sum) {
  double x, temp;
  x = sum * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint j = 0; j < n_col; ++j) {
    temp += abs(value[m + j * n_row]);
    if (x <= temp) {
      return j;
    }
  }
  return (n_col - 1);
}

void DenseMat::load(std::string filename) {
  std::ifstream in_file(filename.c_str());
  if (!in_file.is_open()) {
    throw "Unable to open file " + filename;
  }
  for (uint c = 0; c < n_col; c++) {
    for (uint r = 0; r < n_row; r++) {
      double v;
      in_file >> v;
      value[c * n_row + r] = v;
    }
  }
  in_file.close();
}

void DenseMat::save(std::string filename) {
  std::ofstream out_file(filename.c_str());
  if (out_file.is_open()) {
    for (uint c = 0; c < n_col; c++) {
      for (uint r = 0; r < n_row; r++) {
        out_file << value[c * n_row + r] << "\t";
      }
      out_file << std::endl;
    }
    out_file.close();
  } else {
    std::cout << "Unable to open file " << filename;
  }
}

FactorMat::FactorMat(uint r, uint c) {
  n_row = r;
  n_col = c;
  value = (double *)malloc(n_row * n_col * sizeof(double));
  self_value_ = true;
  sum_type_ = MATRIX_NONE_SUM;
}

FactorMat::FactorMat(uint r, uint c, double *ptr) {
  n_row = r;
  n_col = c;
  value = ptr;
  self_value_ = false;
  sum_type_ = MATRIX_FULL_SUM;
  col_abs_sum = (double *)malloc(n_col * sizeof(double));
  memset(col_abs_sum, 0, n_col * sizeof(double));
  row_abs_sum = (double *)malloc(n_row * sizeof(double));
  memset(row_abs_sum, 0, n_row * sizeof(double));
  // get the absolute sum of each columns
  for (uint i = 0; i < n_col; ++i) {
    for (uint j = 0; j < n_row; ++j) {
      col_abs_sum[i] += abs(value[i * n_row + j]);
      row_abs_sum[j] += abs(value[i * n_row + j]);
    }
  }
}
FactorMat::FactorMat(uint r, uint c, double *ptr, MATRIX_SUM_TYPE TYPE) {
  n_row = r;
  n_col = c;
  value = ptr;
  self_value_ = false;
  sum_type_ = TYPE;
  switch (TYPE) {
    case MATRIX_FULL_SUM:
      col_abs_sum = (double *)malloc(n_col * sizeof(double));
      memset(col_abs_sum, 0, n_col * sizeof(double));
      row_abs_sum = (double *)malloc(n_row * sizeof(double));
      memset(row_abs_sum, 0, n_row * sizeof(double));
      for (uint i = 0; i < n_col; ++i) {
        for (uint j = 0; j < n_row; ++j) {
          col_abs_sum[i] += abs(value[i * n_row + j]);
          row_abs_sum[j] += abs(value[i * n_row + j]);
        }
      }
      break;
    case MATRIX_NONE_SUM:
      col_abs_sum = nullptr;
      row_abs_sum = nullptr;
      break;
    case MATRIX_COL_SUM:
      col_abs_sum = (double *)malloc(n_col * sizeof(double));
      memset(col_abs_sum, 0, n_col * sizeof(double));
      row_abs_sum = nullptr;
      for (uint i = 0; i < n_col; ++i) {
        for (uint j = 0; j < n_row; ++j) {
          col_abs_sum[i] += abs(value[i * n_row + j]);
        }
      }
      break;
    case MATRIX_ROW_SUM:
      col_abs_sum = nullptr;
      row_abs_sum = (double *)malloc(n_row * sizeof(double));
      memset(row_abs_sum, 0, n_row * sizeof(double));
      for (uint i = 0; i < n_col; ++i) {
        for (uint j = 0; j < n_row; ++j) {
          row_abs_sum[j] += abs(value[i * n_row + j]);
        }
      }
      break;
    default:
      break;
  }
}
FactorMat::~FactorMat() {
  switch (sum_type_) {
    case MATRIX_COL_SUM:
      free(col_abs_sum);
      break;
    case MATRIX_ROW_SUM:
      free(row_abs_sum);
      break;
    case MATRIX_FULL_SUM:
      free(col_abs_sum);
      free(row_abs_sum);
      break;
    default:
      break;
  }
  if (self_value_) {
    free(value);
  }
}

double FactorMat::GetElement(uint i, uint j) { return value[j * n_row + i]; }

double FactorMat::GetColSum(uint column) { return col_abs_sum[column]; }
void FactorMat::transpose(double *ptr) {
  for (uint r = 0; r < n_row; ++r) {
    for (uint c = 0; c < n_col; ++c) {
      value[c * n_row + r] = ptr[r * n_col + c];
    }
  }
}
void FactorMat::accumulation(double *ptr) {
  for (uint c = 0; c < n_col; ++c) {
    double sum = 0.0;
    size_t offset = c * n_row;
    for (uint r = 0; r < n_row; ++r) {
      sum += abs(ptr[offset + r]);
      value[c * n_row + r] = sum;
    }
  }
}
uint FactorMat::randRow(uint n) {
  double x, temp;
  x = col_abs_sum[n] * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint i = 0; i < n_row; ++i) {
    temp += abs(value[i + n * n_row]);
    if (x <= temp) {
      return i;
    }
  }
  return (n_row - 1);
}
uint FactorMat::randCol(uint m) {
  double x, temp;
  x = row_abs_sum[m] * ((double)rand() / (double)RAND_MAX);
  temp = 0;
  for (uint j = 0; j < n_col; ++j) {
    temp += abs(value[m + j * n_row]);
    if (x <= temp) {
      return j;
    }
  }
  return (n_col - 1);
}

double MatrixRowMul(const Point2d &p, FactorMat &A, FactorMat &B) {
  double ans = 0.0;
  uint r1 = p.x;
  uint r2 = p.y;
  for (uint c = 0; c < A.n_col; ++c) {
    ans += A(r1, c) * B(r2, c);
  }
  return ans;
}
double MatrixRowMul(const Point3d &p, FactorMat &A, FactorMat &B,
                    FactorMat &C) {
  double ans = 0.0;
  uint r1 = p.x;
  uint r2 = p.y;
  uint r3 = p.z;
  for (uint c = 0; c < A.n_col; ++c) {
    ans += A(r1, c) * B(r2, c) * C(r3, c);
  }
  return ans;
}
double MatrixColMul(const Point2d &p, FactorMat &A, FactorMat &B) {
  double ans = 0.0;
  uint c1 = p.x;
  uint c2 = p.y;
  for (uint r = 0; r < A.n_row; ++r) {
    ans += A(r, c1) * B(r, c2);
  }
  return ans;
}

double MatrixColMul(const Point3d &p, FactorMat &A, FactorMat &B,
                    FactorMat &C) {
  double ans = 0.0;
  uint c1 = p.x;
  uint c2 = p.y;
  uint c3 = p.z;
  for (uint r = 0; r < A.n_row; ++r) {
    ans += A(r, c1) * B(r, c2) * C(r, c3);
  }
  return ans;
}

double MatrixColMul(FactorMat &A, FactorMat &B, uint c1, uint c2) {
  double ans = 0.0;
  for (uint r = 0; r < A.n_row; ++r) {
    ans += A(r, c1) * B(r, c2);
  }
  return ans;
}

double MatrixColMul(FactorMat &A, FactorMat &B, FactorMat &C, uint c1, uint c2,
                    uint c3) {
  double ans = 0.0;
  for (uint r = 0; r < A.n_row; ++r) {
    ans += A(r, c1) * B(r, c2) * C(r, c3);
  }
  return ans;
}
double vectors_mul(const Point2d &coord, FactorMat &A, FactorMat &B) {
  double ans = 0;
  uint i = coord.x;
  uint j = coord.y;
  for (uint r = 0; r < A.n_row; ++r) {
    ans += A(r, i) * B(j, r);
  }
  return ans;
}
double vectors_mul(const Point3d &coord, FactorMat &A, FactorMat &B,
                   FactorMat &C) {
  double ans = 0;
  uint i = coord.x;
  uint j = coord.y;
  uint k = coord.z;
  for (uint r = 0; r < A.n_row; ++r) {
    ans += A(r, i) * B(j, r) * C(k, r);
  }
  return ans;
}

double vectors_mul(const PointNd &p, std::vector<FactorMat *> &vMat) {
  uint MatNum = p.length;
  uint rankSize = vMat[0]->n_row;
  double ans = 0;
  double *temp = (double *)malloc(rankSize * sizeof(double));
  memset(temp, 1, rankSize * sizeof(double));
  for (uint r = 0; r < rankSize; ++r) {
    temp[r] = vMat[0]->GetElement(r, p.index[0]);
  }
  for (uint n = 1; n < MatNum; ++n) {
    for (uint r = 0; r < rankSize; ++r) {
      temp[r] *= vMat[n]->GetElement(p.index[n], r);
    }
  }
  for (uint i = 0; i < rankSize; ++i) {
    ans += temp[i];
  }
  free(temp);
  return ans;
}

void doInsertReverse(double p, std::list<double> &listTop) {
  std::list<double>::iterator itr = listTop.begin();
  if (p < listTop.front()) {
    listTop.push_front(p);
    listTop.pop_back();
    return;
  }
  itr++;
  for (; itr != listTop.end(); ++itr) {
    if (p < (*itr)) {
      listTop.insert(itr, p);
      listTop.pop_back();
      return;
    }
  }
}
void doInsert(double p, std::list<double> &listTop) {
  std::list<double>::iterator itr = listTop.begin();
  if (p > listTop.front()) {
    listTop.push_front(p);
    listTop.pop_back();
    return;
  }
  itr++;
  for (; itr != listTop.end(); ++itr) {
    if (p > (*itr)) {
      listTop.insert(itr, p);
      listTop.pop_back();
      return;
    }
  }
}
void doInsert(double p, std::list<double> &listTop, Point3d &coord,
              std::list<Point3d> &listIdx) {
  auto itr = listTop.begin();
  auto itr2 = listIdx.begin();
  if (p > listTop.front()) {
    listTop.push_front(p);
    listTop.pop_back();
    listIdx.push_front(Point3d(coord.x, coord.y, coord.z));
    listIdx.pop_back();
    return;
  }
  itr++;
  itr2++;
  for (; itr != listTop.end(); ++itr, ++itr2) {
    if (p > (*itr)) {
      listTop.insert(itr, p);
      listTop.pop_back();
      listIdx.insert(itr2, Point3d(coord.x, coord.y, coord.z));
      listIdx.pop_back();
      return;
    }
  }
}
