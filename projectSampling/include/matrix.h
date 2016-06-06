#ifndef __MATRIX_H__
#define __MATRIX_H__


#include <vector>

/*
	class for point2D
*/
class point2D
{
public:
	point2D(size_t i, size_t j);
	~point2D();
	bool operator<(const point2D &toCmp);
	bool operator==(const point2D &toCmp);
	bool operator>(const point2D &toCmp);
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
	~point3D();
	bool operator<(const point3D &toCmp);
	bool operator==(const point3D &toCmp);
	bool operator>(const point3D &toCmp);
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


#endif