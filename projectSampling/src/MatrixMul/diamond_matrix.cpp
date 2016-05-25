/*
	Diamond Sampling for matrix multiplication
	Author: Zhi Lu
	Reference:"Diamond Sampling for Approximate Maximum 
			All-pairs Dot-product(MAD) Search"
*/

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdio.h>
#include "mex.h"
#include "tensor.hpp"
typedef struct 
{
	size_t row;
	size_t col;
	double* element;
	double* SumofCol;
	Matrix(size_t r, size_t c, double*pr){
		row = r;
		col = c;
		element = pr;
		double*SumofCol = (double*)malloc(col*sizeof(double));
		memcpy(SumofCol, 0, col*sizeof(double));
		//get the absolute sum of each columns
		double temp = 0;
		for(size_t i = 0; i < col; ++i){
			temp = 0;
			for(size_t j = 0; j < row; ++j){
				temp += abs(element(i*row + j));
			}
			SumofCol[i] = tmp;
		}
	}
	~Matrix(){
		free(SumofCol);
	}
	double GetEmelent(size_t i, size_t j){
		return element[i*row + j];
	}
	double GetColSum(size_t column){
		return SumofCol[column];
	}
	size_t randRow(size_t n){
		double x,sum,temp;
		sum = SumofCol[i];
		x = sum*((double)rand()/(double)RAND_MAX);
		temp = 0;
		for (size_t i = 0; i < row; ++i){
			temp += abs(element[i + n*row]);
			if(x <= temp){ 
				return i;
			}
		}
		return (row-1);
	}
}Matrix;

typedef std::vector<size_t> coordinate;
typedef std::pair<coordinate,double> PAIR;

int cmp(const PAIR &x,const PAIR&y){
	return x.second > y.second;
}

int sgn_foo(double x){
	return x<0? -1:1;
}
/*
	give an coord(i_0,i_1,i_2,...,i_(N-1))
	compute the value of U_{(i_0,i_1,i_2,...,i_(N-1))}
*/
double vectors_mul(const coordinate &coord,\
 				std::vector<FactorMatrix*> &vMatrixs,\
 				size_t nrhs,size_t rank_size){
    double ans = 0;
    double* temp = new double[rank_size];
    /*initialize to the column of vMatrixs[0] */
    for (size_t i = 0; i < rank_size; ++i){
        temp[i] = vMatrixs[0]->get_element(i,coord[0]);
    }
    for (size_t i = 1; i < nrhs; ++i){
        for(size_t j = 0; j < rank_size; ++j){
            temp[j] *= vMatrixs[i]->get_element(coord[i],j);
        }
    }
    for (size_t i = 0; i < rank_size; ++i){
        ans += temp[i];
    }
    delete []temp;
    return ans;
}
void display(std::vector<size_t> v){
	std::vector<size_t>::iterator itr;
	for (itr = v.begin(); itr != v.end(); ++itr){
		printf("%d, ",(*itr));
	}
}
void display_map(std::map<std::vector<size_t>,double> v){
	std::map<std::vector<size_t>,double> ::const_iterator itr;
	for (itr = v.begin(); itr != v.end(); ++itr){
		display(itr->first);
		printf("%f\n",itr->second);
	}
}

/* 
	sample the pair(k, i)
	S: times needed to be sampled;
	index: will return the result index = i * row + k,
		it is sorted according to i then k;
	dim: the length of freq_k;
	freq_k: will return the number of each k has been sampled;
	L: the length of the pdf;
	sum_pdf: the sum of the pdf, which is not one, for the decease
		of computation cost.
*/

int sample_index(size_t S, size_t *index, \
				 size_t dim, &freq_k,\
				 size_t L, double*pdf, double sum_pdf);

/* 
	Vose's alias method for sample;
	It is used for the situation when 
		L is not much bigger than S;
	A method with constant time per sample.
*/
int vose_alias(size_t s, size_t *dst, \
		`	   size_t n, double *pdf,double sum_pdf);
/*
	C = AB;
	A has d rows and m columns,
	B has n rows and d columns;
	It is needed for data structure;
	Matlab function:
		[result, index] = function_name(A,B,,top-t,sample_number);
		result: the desired top-t maximums;
		index: the corresponding instances from A and B;
		sample_number = S: control the numbers to do sampling .

*/

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	start = clock();
	//--------------------
	// Initialization
	//--------------------
	Matirx MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matirx MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	const int top_t = static_cast(int)mxGetPr(prhs[2]);
	const size_t NumSample = static_cast(size_t)mxGetPr(prhs[3]);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during initialization\n",duration);

	//-------------------------------------
	// Compute weight
	//-------------------------------------

	double SumofW = 0;
	double *weight = (double*)malloc(MatA.row*MatA.col*szieof(double));
	memcpy(weight, 0, MatA.row*MatA.col*szieof(double));
	double tempW = 0;
	start = clock();
	for (size_t k = 0; k < MatA.row; ++k){
		for(size_t i = 0; i < MatA.col; ++i){
			tempW = 1;
			tempW *= abs(MatA.GetEmelent(k,i));
			tempW *= MatA.SumofCol[i];
			tempW *= MatB.SumofCol[k];
			weight[i*MatA.col + k] = tempW;
			SumofW += tempW;
		}
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computing weight\n",duration);

	//-------------------------
	// Do Sampling
	//-------------------------

	start = clock();
	size_t *WeightInd = (size_t *)malloc(NumSample*sizeof(size_t));
	memcpy(WeightInd, 0, NumSample*sizeof(size_t));
	int freq_k[MatA.row];
	// sample S pairs (k, i) ,
	sample_index(NumSample, WeightInd, \
				 MatA.row, freq_k,\
				 MatA.row*MatA.col, weight, SumofW);
	// k, i, j, k'
	size_t *IndforI = (size_t*)malloc(NumSample*sizeof(size_t));
	memcpy(IndforI, 0, NumSample*sizeof(size_t))
	size_t *IndforJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memcpy(IndforJ, 0, NumSample*sizeof(size_t))
	size_t *IndforKp = (size_t*)malloc(NumSample*sizeof(size_t));
	memcpy(IndforKp, 0, NumSample*sizeof(size_t))
	// record all i = index / row; index = i * row + k;
	// and sample k';
	for (int s = 0; s < NumSample; ++s){
		IndforI[s] = WeightInd[s] / MatA.row;
		IndforKp[s] = MatA.randRow(WeightInd[s] / MatA.row);
	}
	// sample j;
	size_t offset = 0;
	for (int k = 0; k < MatA.row; ++k){
		vose_alias( freq_k[k], (IndforJ + offset), \
					MatB.row, \
					(MatB.element + offset*MatB.row), \
					MatB.SumofCol(k));
		offset += freq_k[k];
	}
	// compute update value
	double valueSampled = 1.0;
	size_t indi,indj,indk,indkp;
	for (int s = 0; s < NumSample ; ++s){
		// sample k'
		indk = WeightInd[s] / MatA.row;
		indkp = IndforKp[s];
		indi = IndforI[s];
		indj = IndforJ[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(MatA.GetEmelent(indk,indi));
		valueSampled *= sgn_foo(MatB.GetEmelent(indj,indk));
		valueSampled *= sgn_foo(MatA.GetEmelent(indkp,indi));
		valueSampled *= MatB.GetEmelent(indj,indkp);
		// Update the element in coordinate
		Tensor[ coords[s] ] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during sampling\n",duration);
	/* sort value in tensor */
	start = clock();
	std::vector<PAIR> pair_vec;
	std::map<coordinate,double>::iterator map_itr;
	double true_value = 0;

	for (map_itr = Tensor.begin(); map_itr != Tensor.end(); ++map_itr){

		true_value = vectors_mul((map_itr->first),vMatrixs,nrhs,RANK);
		pair_vec.push_back(make_pair(map_itr->first,true_value));
		//pair_vec.push_back(make_pair(map_itr->first,map_itr->second));
	}
	sort(pair_vec.begin(),pair_vec.end(),cmp);


	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computer and sorting tenosor \n",duration);

	start = clock();
	size_t phls_row = pair_vec.size();
	plhs[0] = mxCreateNumericMatrix(phls_row, nrhs, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(phls_row, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[1]);
	for(size_t m = 0; m < pair_vec.size(); ++m){
		plhs_result[m] = pair_vec[m].second;
		//plhs_result[m] = vectors_mul(pair_vec[m].first,vMatrixs,nrhs,RANK);
		for(size_t n = 0; n < nrhs; ++n){
			/*index start with 1*/
			plhs_pr[m+n*phls_row] = ((pair_vec[m].first[n])+1);
		}
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during converting \n",duration);

	//display_map(Tensor);
	delete []weight;
	delete []WeightInd;
	std::vector<FactorMatrix*>::iterator itr;
	for (itr = vMatrixs.begin(); itr != vMatrixs.end() ; ++itr){
		delete *itr;
	}
}



int sample_index(size_t n, size_t *index, \
				 size_t L_HEAD, &freq_k,\
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

/* 
	Vose's alias method for sample
*/
int vose_alias(size_t s, size_t *dst, \
		`	   size_t n, double *pdf,double sum_pdf){
	double *scaled_prob = new double[n];
	double *table_prob = new double[n];
	size_t *table_alias = new size_t[n];
	size_t *table_small = new size_t[n];
	size_t *table_large = new size_t[n];
	size_t small_index = 0;
	size_t large_index = 0;
	/* stage 1: initialization */
	for (size_t i = 0; i < n; ++i){
		scaled_prob[i] = abs(pdf[i]) * n;
		if ( scaled_prob[i] < sum_pdf ){
			table_small[small_index] = i;
			++small_index;
		}else{
			table_large[large_index] = i;
			++large_index;
		}
	}
	size_t l,g;
	while(small_index != 0 && large_index != 0){
		small_index -= 1;
		large_index -= 1;
		l = table_small[small_index];
		g = table_large[large_index];
		table_prob[l] = scaled_prob[l];
		table_alias[l] = g;
		scaled_prob[g] = (scaled_prob[g] + scaled_prob[l]) - sum_pdf;

		if (scaled_prob[g] < sum_pdf){
			table_small[small_index] = g;
			++small_index;
		}else{
			table_large[large_index] = g;
			++large_index;
		}
	}
	while(large_index != 0){
		large_index -= 1;
		table_prob[table_large[large_index]] = sum_pdf;
	}
	while(small_index != 0){
		small_index -= 1;
		table_prob[table_small[small_index]] = sum_pdf;
	}
	/* stage 2: random sampling */
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

