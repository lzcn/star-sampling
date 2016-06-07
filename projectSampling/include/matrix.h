#ifndef __MATRIX_H__
#define __MATRIX_H__


#include <vector>
#include <list>
#include <map>
#include <algorithm>

/*
	class for point2D
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

typedef	std::vector<size_t> pointnD;
/*
	class for Matrix: column major order
	that is a matrix A will be stored in 
	continuous memory as :
	a[0][0] a[1][0] a[2][0] ... a[0][1] a[1][1] ...
	element: the elements of matrix
		a[m][n] = element[n * row + m]
	SumofCol: the absolute sum of every column
*/

class Matrix{
public:
	size_t row;
	size_t col;
	double* element;
	double* SumofCol;
	Matrix(size_t r, size_t c, double*pr);
	~Matrix();
	double GetEmelent(size_t i, size_t j);
	double GetColSum(size_t column);
	// given a column index n sample a row index m 
	// with probability abs(M_{m,n})/SumofCol(n)
	size_t randRow(size_t n);
};
/*
	compute the dot product of two matrices' column
	ans = A(:,m)'*B(:,n)
*/
double MatrixColMul(const Matrix &A, const Matrix &B, \
					size_t m, size_t n);

double MatrixColMul(const Matrix &A, \
					const Matrix &B, \
					const Matrix &C, \
					size_t m, size_t n, size_t p);
/*
	list is sorted in ascending order;
	insert p to the list;
	length: size of list;
*/
void doInsert(double p, std::list<double> &listTop);

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
#endif