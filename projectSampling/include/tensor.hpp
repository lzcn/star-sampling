#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <map>
#include <vector>
#include "mex.h"
/*
	sample s indexes with probability (pdf/sum_pdf)
	saved in dst;
	1.vose_alias method;
	2.regular method;
*/

int vose_alias(size_t s, size_t *dst, size_t n, double *pdf,double sum_pdf);
int sample_index(size_t s, size_t *dst, size_t L_HEAD,std::map<size_t, size_t> &freq_k,size_t n, double*pdf, double sum_pdf);

class FactorMatrix{
public:
	FactorMatrix(const mxArray *matrix);
	~FactorMatrix();
	/* compute the absolute sum of m-th row */
	double get_row_sum(const size_t m);
	/* compute the absoulte sum of n-th column */
	double get_col_sum(const size_t n);
	/* compute the absoulte sum of all elements */
	double get_sum();
	/* return the (m,n)-th element */
	double get_element(const size_t m, const size_t n);
	/* sample index of row with n-th column be the probability */
	size_t sample_row_index(const size_t n);
	
	size_t get_row(){return row;};
	size_t get_col(){return col;};
	/* absoulte sum of every column*/
	double* get_pr(){return pr;};
	double* sum_of_col(){return pr_col;};

	void disp();
private:
	size_t row;//number of rows
	size_t col;//number of columns
	double *pr;//vector of elements
	double *pr_col;
};

#endif