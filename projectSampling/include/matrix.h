#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <algorithm>

typedef unsigned int uint;

const uint MATRIX_NONE_SUM = 0;
const uint MATRIX_COL_SUM  = 1;
const uint MATRIX_ROW_SUM  = 2;
const uint MATRIX_FULL_SUM = 3;
////////////////////////////////
//	      DECALARATIONS	      //
////////////////////////////////

double sgn(double x);
double sgn_foo(double x);
class point2D;
class point3D;
class pointND;
class SubIndex;

void doInsertReverse(double p, std::list<double> &listTop);
void doInsert(double p, std::list<double> &listTop);
void doInsert(double p, std::list<double> &listTop, point3D &coord, std::list<point3D> &listIdx);

typedef std::pair<point2D,double> pidx2d;
typedef std::pair<point3D,double> pidx3d;
typedef std::pair<pointND,double> pidxNd;

template <typename Tpair> 
uint compgt(const Tpair &v1,const Tpair &v2){ return (v1.second > v2.second); }
template <typename Tpair>
uint complt(const Tpair &v1,const Tpair &v2){ return (v1.second < v2.second); }

class Matrix;
double MatrixRowMul(const point2D &coord, Matrix &A, Matrix &B);
double MatrixRowMul(const point3D &coord, Matrix &A, Matrix &B, Matrix &C);
double MatrixColMul(const point2D &coord, Matrix &A, Matrix &B);
double MatrixColMul(const point3D &coord, Matrix &A, Matrix &B, Matrix &C);
double MatrixColMul(Matrix &A, Matrix &B, uint i, uint j);
double MatrixColMul(Matrix &A, Matrix &B, Matrix &C, uint i, uint j, uint k);
double vectors_mul(const pointND &p,std::vector<Matrix*> &vMat);
double vectors_mul(const point2D &coord, Matrix &A, Matrix &B);
double vectors_mul(const point3D &coord, Matrix &A, Matrix &B, Matrix &C);
void binary_search(size_t s, uint *dst, uint n, double *pdf);
uint binary_search_once(double *a, uint ub, double s);
void sort_sample(size_t s, uint*i, uint*r, size_t *freq, uint m, uint n, double*pdf, double sum_pdf);
void sort_sample(size_t s, uint*dst, uint n, double*p, double sum);
void vose_alias(size_t s, uint *dst, uint n, double *pdf,double sum_pdf);
////////////////////////////////
//	     IMPLEMENTATIONS      //
////////////////////////////////
class point2D
{
public:
	point2D(uint i, uint j);
	~point2D(){};
	bool operator<(const point2D &toCmp)const;
	bool operator==(const point2D &toCmp)const;
	bool operator>(const point2D &toCmp)const;
	uint x;
	uint y;
};

class point3D
{
public:
	point3D(uint i, uint j, uint k);
	~point3D(){};
	bool operator<(const point3D &toCmp)const;
	bool operator==(const point3D &toCmp)const;
	bool operator>(const point3D &toCmp)const;
	uint x;
	uint y;
	uint z;
};
struct hashFunc{
    size_t operator()(const point3D &k) const{
    size_t h1 = std::hash<uint>()(k.x);
    size_t h2 = std::hash<uint>()(k.y);
    size_t h3 = std::hash<uint>()(k.z);
    return (h1 ^ (h2 << 1)) ^ h3;
    }
};
struct equalsFunc{
  bool operator()( const point3D& lhs, const point3D& rhs ) const{
    return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
  }
};
typedef std::unordered_map<point3D, double, hashFunc, equalsFunc> TPoint3DMap;
class pointND
{
public:
	pointND(uint *p, uint n);
	~pointND(){};
	bool operator<(const pointND &toCmp)const;
	uint num;
	uint *coord;
};

class Matrix{
public:
	uint row;
	uint col;
	double* element;
	double* SumofCol;
	double* SumofRow;
	Matrix(uint r, uint c);
	Matrix(uint r, uint c, double*pr);
	Matrix(uint r, uint c, double*pr, uint TYPE);
	~Matrix();
	double GetElement(uint i, uint j);
	double GetColSum(uint column);
	double& operator()(uint i, uint j) const { return element[j*row +i]; }
	// return m with probability abs(M_{m,n})/SumofCol(n)
	uint randRow(uint n);
	// return n with probability abs(M_{m,n})/SumofRow(m)
	uint randCol(uint m);
private:
	uint _SUMTYPE;
	bool _SELFVALUE;
};

class SubIndex{
public:
	SubIndex(uint n, uint *max);
	~SubIndex();
	bool isDone(){return doneFlag;};
	bool reset();
	SubIndex& operator++();
	const uint *getIdx(){return curIdx;};
private:
	uint idxSize;
	bool doneFlag;
	uint *curIdx;
	const uint *maxIdx;
};

#endif
