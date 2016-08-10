#ifndef __MATRIX_H__
#define __MATRIX_H__


#include <vector>
#include <list>
#include <map>
#include <algorithm>

/*
	class for point2D
	(x,y)
*/
class point2D
{
public:
	point2D(size_t i, size_t j);
	~point2D(){};
	bool operator<(const point2D &toCmp)const;
	bool operator==(const point2D &toCmp)const;
	bool operator>(const point2D &toCmp)const;
	size_t x;
	size_t y;
};

/*
	class for point3D
	(x,y,z)
*/
class point3D
{
public:
	point3D(size_t i, size_t j, size_t k);
	~point3D(){};
	bool operator<(const point3D &toCmp)const;
	bool operator==(const point3D &toCmp)const;
	bool operator>(const point3D &toCmp)const;
	size_t x;
	size_t y;
	size_t z;
};

/*

	class for pointND
	coord: coordinate
	num: length of coordinate
*/
class pointND
{
public:
	pointND(size_t *p, size_t n);
	~pointND(){};
	bool operator < (const pointND &toCmp)const;
	size_t num;
	size_t *coord;
};

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
	size_t row;
	size_t col;
	double* element;
	double* SumofCol;
	double* SumofRow;
	Matrix(size_t r, size_t c, double*pr);
	~Matrix();
	double GetElement(size_t i, size_t j);
	double GetColSum(size_t column);
	// return m with probability abs(M_{m,n})/SumofCol(n)
	size_t randRow(size_t n);
	// return n with probability abs(M_{m,n})/SumofRow(m)
	size_t randCol(size_t m);
};

// sign functions
int sgn_foo(double x);
/*
	Euclidean distance of two cloumns
*/
double EuclideanMetric(const point2D, const Matrix &A, const Matrix &B);
/*
	Cosine similarity distance of two cloumns
*/
double CosineMetric(const point2D, const Matrix &A, const Matrix &B);
/*
	Inner product of two columns
*/
double MatrixColMul(const Matrix &A, const Matrix &B, \
					size_t m, size_t n);
double MatrixColMul(const Matrix &A, \
					const Matrix &B, \
					const Matrix &C, \
					size_t m, size_t n, size_t p);
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
void doInsert(double p, std::list<double> &listTop);
void doInsert(double p, std::list<double> &listTop, point3D &coord, std::list<point3D> &listIdx);

/* 
	Vose's alias method for sample;
	It is used for the situation when 
		L is not much bigger than S;
	A method with constant time per sample.
*/
int vose_alias(size_t s, size_t *dst, \
			   size_t n, double *pdf,double sum_pdf);

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

int sample_index(size_t S, size_t *index, \
				 size_t *IndforI, size_t *IndforK, \
				 size_t *freq_k, \
				 size_t m, size_t n, \
				 double*pdf, double sum_pdf);
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
	SubIndex(int n, size_t *max);
	~SubIndex();
	bool isDone(){return doneFlag;};
	bool reset();
	SubIndex& operator+(const size_t step);
	SubIndex& operator++();
	const size_t *getIdx(){return curIdx;};
private:
	int idxSize;
	bool doneFlag;
	size_t *curIdx;
	const size_t *maxIdx;
};

#endif
