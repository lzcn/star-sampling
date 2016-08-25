#ifndef __MATRIX_H__
#define __MATRIX_H__


#include <vector>
#include <list>
#include <map>
#include <algorithm>
typedef unsigned int uint;
const uint MATRIX_NONE_SUM = 0;
const uint MATRIX_COL_SUM  = 1;
const uint MATRIX_ROW_SUM  = 2;
const uint MATRIX_FULL_SUM = 3;
/*
	class for point2D
	(x,y)
*/
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

/*
	class for point3D
	(x,y,z)
*/
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

/*

	class for pointND
	coord: coordinate
	num: length of coordinate
*/
class pointND
{
public:
	pointND(uint *p, uint n);
	~pointND(){};
	bool operator<(const pointND &toCmp)const;
	uint num;
	uint *coord;
};
typedef std::pair<point2D,double> pidx2d;
typedef std::pair<point3D,double> pidx3d;
typedef std::pair<pointND,double> pidxNd;
template <typename Tpair>
uint compgt(const Tpair &v1,const Tpair &v2){
	return (v1.second > v2.second);
}
/*
	class for Matrix: column major order
			that is a matrix A will be stored in 
			continuous memory as :
			a[0][0] a[1][0] a[2][0] ... a[0][1] a[1][1] ...
	element: the elements of matrix
			GetElement(m,n) = element[n * row + m]
	SumofCol: the absolute sum of every column
	SumofRow: the absolute sum of every row
	ranRow(n): given a column index, sample the row index
	ranCol(n): given a row index, sample the column index
*/

class Matrix{
public:
	uint row;
	uint col;
	double* element;
	double* SumofCol;
	double* SumofRow;
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
};

// sign functions
uint sgn_foo(double x);
/*
	Euclidean distance of two cloumns
*/
double EuclideanMetric(const point2D&, const Matrix &A, const Matrix &B);
double EuclideanMetricRow(const point2D&, const Matrix &A, const Matrix &B);
/*
	Cosine similarity distance of two cloumns
*/
double CosineMetric(const point2D&, const Matrix &A, const Matrix &B);
double CosineMetricRow(const point2D&, const Matrix &A, const Matrix &B);
/*
	Inner product of two columns
*/
double MatrixColMul(const Matrix &A, const Matrix &B, \
					uint m, uint n);
double MatrixColMul(const Matrix &A, \
					const Matrix &B, \
					const Matrix &C, \
					uint m, uint n, uint p);
double MatrixColMul(const point2D &coord, \
				   Matrix &A, \
				   Matrix &B);
double MatrixColMul(const point3D &coord, \
				   Matrix &A, \
				   Matrix &B, \
				   Matrix &C);
/*
	Inner product of two rows
*/
double MatrixRowMul(const point2D &coord, \
				   Matrix &A, \
				   Matrix &B);
double MatrixRowMul(const point3D &coord, \
				   Matrix &A, \
				   Matrix &B, \
				   Matrix &C);

double vectors_mul(const point2D &coord, \
				   Matrix &A, \
				   Matrix &B);
double vectors_mul(const point3D &coord, \
				   Matrix &A, \
				   Matrix &B, \
				   Matrix &C);

double vectors_mul(const pointND &p, std::vector<Matrix*> &vMat);
/*
	list is sorted in ascending order;
	insert p to the list;
	length: size of list;
*/
void doInsertReverse(double p, std::list<double> &listTop);
void doInsert(double p, std::list<double> &listTop);
void doInsert(double p, std::list<double> &listTop, point3D &coord, std::list<point3D> &listIdx);

/* 
	Vose's alias method for sample;
	It is used for the situation when 
		L is not much bigger than S;
	A method with constant time per sample.
*/
uint vose_alias(size_t s, uint *dst, \
			   uint n, double *pdf,double sum_pdf);

/* 
	Sample the pair(k, i)
	Suppose the size of weight/ pdf is (m, n);
	It requires the weight/ pdf to be stored like that
		weight[index] = weight[k * n + i]:
	S: number of samples needed to be sampled;
	index:  the result which index = k * col + i,
		so that it is sorted according to k then i;
		and it will be convenience for the next stage;
	freq_k: will return the number of each k has been sampled;
	m and n : the shape of weight;
		the dimension of feature vector.
	sum_pdf: the sum of the pdf, which is not one, for the decease
		of computation cost.
*/


uint sort_sample(size_t s, uint*i, uint*r, size_t *freq, uint m, uint n, double*pdf, double sum_pdf);
/* 
	SubIndex is used for loop
	data: 
		index_cur: indicate the current index for 
		index_max: each element is the number vector
				of correspongding maxtirx
	method: 
		+ :add t to current index
*/

class SubIndex{
public:
	SubIndex(uint n, uint *max);
	~SubIndex();
	bool isDone(){return doneFlag;};
	bool reset();
	SubIndex& operator+(const uint step);
	SubIndex& operator++();
	const uint *getIdx(){return curIdx;};
private:
	uint idxSize;
	bool doneFlag;
	uint *curIdx;
	const uint *maxIdx;
};

#endif
