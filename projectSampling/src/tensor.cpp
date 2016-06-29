#include <algorithm>
#include "mex.h"
#include "tensor.hpp"


FactorMatrix::FactorMatrix(const mxArray *matrix){
	row = mxGetM(matrix);
	col = mxGetN(matrix);
	pr = mxGetPr(matrix);
	pr_col = new double[col];
	for (size_t j = 0; j < col; ++j){
		pr_col[j] = get_col_sum(j);
	}
}

FactorMatrix::~FactorMatrix(){
	delete []pr_col;
}

double FactorMatrix::get_element(size_t m, size_t n){
	return pr[m + n*row];
}

double FactorMatrix::get_col_sum(const size_t n){
	double sum = 0;
	for (size_t i = 0; i < row; ++i){
		sum += abs(pr[i+n*row]);
	}
	return sum;
}

double FactorMatrix::get_row_sum(const size_t m){
	double sum = 0;
	for (size_t i = 0; i < col; ++i){
		sum += abs(pr[m + i*row]);
	}
	return sum;
}

/* Given the n-th column, sample the row index of this column */
size_t FactorMatrix::sample_row_index(const size_t n){
	double x,sum,temp;
	sum = pr_col[n];
	x = sum*((double)rand()/(double)RAND_MAX);
	temp = 0;
	for (size_t i = 0; i < row; ++i){
		temp += abs(pr[i + n*row]);
		if(x <= temp){ 
			return i;
		}
	}
	return (row-1);
}

void FactorMatrix::disp(){
	for(size_t i = 0; i < row; ++i){
		for (size_t j = 0; j < col; ++j)
		{
			printf("%.2f ,", pr[i+j*row]);
		}
		printf("\n\n\n");
	}
	for (size_t i = 0; i < col; ++i)
	{
		printf("%.2f ,", pr_col[i]);
	}
	printf("\n");
}
/* 
	sample an index:
	pdf: the vector of weight;
	sum_pdf: sum of vector pdf;
	Range: length of vector
	n: number of samples;
	index: result vector of indexes.
*/

int sample_index(size_t n, size_t *index, size_t L_HEAD, std::map<size_t, size_t> &freq_k,\
			size_t Range, double*pdf, double sum_pdf){
	/* get n random numbers*/
	std::vector<double> ind_u;
	for (size_t i = 0; i < n; ++i){
		ind_u.push_back(sum_pdf*((double)rand()/(double)RAND_MAX));
	}
	/* sort the random values*/
	sort(ind_u.begin(),ind_u.end());
	size_t k = 0;
	double sum_prob = pdf[0];
	for (size_t i = 0; i < n; ++i){
		while((ind_u[i] >= sum_prob) && (k < (Range-1))){
			sum_prob += pdf[++k];
		}
		index[i] = k;
		freq_k[k / L_HEAD] ++;
	}
	return 1;
}

/*Vose's alias method for sample*/
int vose_alias(size_t s, size_t *dst, size_t n, double *pdf,double sum_pdf){
	double *scaled_prob = new double[n];
	double *table_prob = new double[n];
	size_t *table_alias = new size_t[n];
	size_t *table_small = new size_t[n];
	size_t *table_large = new size_t[n];
	size_t small_index = 0;
	size_t large_index = 0;
	//std::vector<double> scaled_prob;
	//std::stack<size_t> small;
	//std::stack<size_t> large;
	/* stage 1: init */
	for (size_t i = 0; i < n; ++i){
		//scaled_prob[i] = pdf[i] * (double)n / sum_pdf;
		scaled_prob[i] = abs(pdf[i]) * n;
		if ( scaled_prob[i] < sum_pdf ){
			//small.push(i);
			table_small[small_index] = i;
			++small_index;
		}else{
			//large.push(i);
			table_large[large_index] = i;
			++large_index;
		}
	}
	size_t l,g;
	//while(!small.empty() && !large.empty()){
	while(small_index != 0 && large_index != 0){
		//l = small.top();
		//small.pop();
		//g = large.top();
		//large.pop();
		small_index -= 1;
		large_index -= 1;
		l = table_small[small_index];
		g = table_large[large_index];
		table_prob[l] = scaled_prob[l];
		table_alias[l] = g;
		scaled_prob[g] = (scaled_prob[g] + scaled_prob[l]) - sum_pdf;

		if (scaled_prob[g] < sum_pdf){
			//small.push(g);
			table_small[small_index] = g;
			++small_index;
		}else{
			//large.push(g);
			table_large[large_index] = g;
			++large_index;
		}
	}
	//while(!large.empty()){
	while(large_index != 0){
		//table_prob[large.top()] = sum_pdf;
		//large.pop();
		large_index -= 1;
		table_prob[table_large[large_index]] = sum_pdf;
	}
	//while(!small.empty()){
	while(small_index != 0){
		//table_prob[small.top()] = sum_pdf;
		//small.pop();
		small_index -= 1;
		table_prob[table_small[small_index]] = sum_pdf;
	}
	/* stage 2:rand */
	double u;
	size_t fair_die;
	for (size_t i = 0; i < s; ++i ){
		fair_die = rand() % n;
		u = sum_pdf*(double)rand()/(double)RAND_MAX;
		if (table_prob[fair_die] >= u){
			dst[i] = fair_die;
		}else{
			dst[i] = table_alias[fair_die];
		}
	}
	delete []table_prob;
	delete []table_alias;
	delete []scaled_prob;
	delete []table_small;
	delete []table_large;
	return 1;
}

