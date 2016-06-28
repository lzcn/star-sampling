/*
	Diamond Sampling for matrix multiplication
	Author: Zhi Lu
	Reference:"Diamond Sampling for Approximate Maximum 
			All-pairs Dot-product(MAD) Search"
*/
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>
#include "mex.h"

struct indexIJ
{	
	size_t indI;
	size_t indJ;
	indexIJ(size_t i,size_t j){
		indI = i;
		indJ = j;
	}
	~indexIJ(){}
	bool operator<(const struct indexIJ &toCmp)const{
		if(indI < toCmp.indI){
			return true;
		}else if(indI == toCmp.indI){
			if(indJ < toCmp.indJ)
				return true;
		}
		return false;
	}
	bool operator==(const struct indexIJ &toCmp){
		if(indI == toCmp.indI && indJ == toCmp.indJ)
			return true;
		return false;
	}

	bool operator > (const struct indexIJ &toCmp){
		if(indI > toCmp.indI){
			return true;
		}else if(indI == toCmp.indI){
			if(indJ > toCmp.indJ)
				return true;
		}
		return false;
	}
};

typedef std::pair<struct indexIJ,double> indValue;

struct Matrix
{
	size_t row;
	size_t col;
	double* element;
	double* SumofCol;
	Matrix(size_t r, size_t c, double*pr){
		row = r;
		col = c;
		element = pr;
		SumofCol = (double*)malloc(col*sizeof(double));
		memset(SumofCol, 0, col*sizeof(double));
		//get the absolute sum of each columns
		double temp = 0.0;
		for(size_t i = 0; i < col; ++i){
			temp = 0.0;
			for(size_t j = 0; j < row; ++j){
				temp += abs(element[i*row + j]);
			}
			SumofCol[i] = temp;
		}
	}
	~Matrix(){
		free(SumofCol);
	}
	double GetEmelent(size_t i, size_t j){
		return element[j*row + i];
	}
	double GetColSum(size_t column){
		return SumofCol[column];
	}
	size_t randRow(size_t n){
		double x,temp;
		x = SumofCol[n]*((double)rand()/(double)RAND_MAX);
		temp = 0;
		for (size_t i = 0; i < row; ++i){
			temp += abs(element[i + n*row]);
			if(x <= temp){ 
				return i;
			}
		}
		return (row-1);
	}
};



int cmp(const indValue &x,const indValue&y){
	return x.second > y.second;
}

int sgn_foo(double x){
	return x<0? -1:1;
}

/*
	give an pair(i, j)
	compute the value of c_ij;
*/
double vectors_mul(const struct indexIJ &coord, struct Matrix &A, struct Matrix &B){
    double ans = 0;
    size_t indI = coord.indI;
    size_t indJ = coord.indJ;
    for (size_t k = 0; k < A.row; ++k){
        ans += A.GetEmelent(k,indI) * B.GetEmelent(indJ,k);
    }
    return ans;
}

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
	Vose's alias method for sample;
	It is used for the situation when 
		L is not much bigger than S;
	A method with constant time per sample.
*/
int vose_alias(size_t s, size_t *dst, \
			   size_t n, double *pdf,double sum_pdf);
/*
	c = ab;
	a has d rows and m columns,
	b has n rows and d columns;
	it is needed for data structure;
	matlab function:
		[result, index] = function_name(a,b,,top-t,sample_number);
		result: the desired top-t maximums;
		index: the corresponding instances from a and b;
		sample_number = s: control the numbers to do sampling .

*/

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	start = clock();
	struct Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	struct Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	const int top_t = (int)mxGetPr(prhs[2])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during initialization\n",duration);

	//-------------------------------------
	// Compute weight
	//-------------------------------------

	start = clock();
	double SumofW = 0;
	//weight has the same size of A
	double *weight = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(weight, 0, MatA.row*MatA.col*sizeof(double));
	double tempW = 0;
	// weight[k * MatA.col + i] : i-th column k-th row
	for (size_t k = 0; k < MatA.row; ++k){
		for(size_t i = 0; i < MatA.col; ++i){
			//w_{ki} = |a_{ki}|*||a_{*i}||_1*||b_{*k}||_1
			tempW = 1;
			tempW *= abs(MatA.GetEmelent(k,i));
			tempW *= MatA.SumofCol[i];
			tempW *= MatB.SumofCol[k];
			weight[k*MatA.col + i] = tempW;
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
	// sampled index  for weight
	size_t *WeightInd = (size_t *)malloc(NumSample*sizeof(size_t));
	memset(WeightInd, 0, NumSample*sizeof(size_t));
	// sampled k, i, j, k'
	size_t *IndforK = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforK, 0, NumSample*sizeof(size_t));	
	size_t *IndforI = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforI, 0, NumSample*sizeof(size_t));
	size_t *IndforJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforJ, 0, NumSample*sizeof(size_t));
	size_t *IndforKp = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforKp, 0, NumSample*sizeof(size_t));
	// sampled k's frequency 
	size_t *freq_k = (size_t*)malloc(MatA.row*sizeof(size_t));
	memset(freq_k, 0, MatA.row*sizeof(size_t));
	// Do sample S pairs (k, i) ,
	sample_index(NumSample, WeightInd, \
				 IndforI, IndforK, \
				 freq_k, \
				 MatA.row, MatA.col, \
				 weight, SumofW);
	// sample k';
	for (int s = 0; s < NumSample; ++s){
		IndforKp[s] = MatA.randRow(IndforI[s]);
	}
	// sample j;
	size_t offset = 0;
	for (int k = 0; k < MatA.row; ++k){
		vose_alias( freq_k[k], (IndforJ + offset), \
					MatB.row, \
					(MatB.element + k*MatB.row), \
					MatB.SumofCol[k]);
		offset += freq_k[k];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t indi,indj,indk,indkp;
	std::map<struct indexIJ, double> IrJc;
	for (int s = 0; s < NumSample ; ++s){
		indk = IndforK[s];
		indkp = IndforKp[s];
		indi = IndforI[s];
		indj = IndforJ[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(MatA.GetEmelent(indk,indi));
		valueSampled *= sgn_foo(MatB.GetEmelent(indj,indk));
		valueSampled *= sgn_foo(MatA.GetEmelent(indkp,indi));
		valueSampled *= MatB.GetEmelent(indj,indkp);
		// Update the element in coordinate
		IrJc[struct indexIJ(indi,indj)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during sampling\n",duration);

	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------

	start = clock();
	std::vector<indValue> sortVec;
	std::map<struct indexIJ, double>::iterator mapItr;
	double true_value = 0;

	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		true_value =  vectors_mul(mapItr->first,MatA, MatB);
		sortVec.push_back(std::make_pair(mapItr->first,true_value));
		//sortVec.push_back(make_pair(mapItr->first,mapItr->second));
	}
	sort(sortVec.begin(),sortVec.end(),cmp);

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computer and sorting tensor \n",duration);
 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	start = clock();
	size_t phls_row = sortVec.size();
	// pair
	plhs[0] = mxCreateNumericMatrix(phls_row, 2, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[0]);
	// value
	plhs[1] = mxCreateDoubleMatrix(phls_row, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[1]);
	for(size_t m = 0; m < sortVec.size(); ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//i
		plhs_pr[m] = (sortVec[m].first.indI + 1);
		//j
		plhs_pr[m + phls_row] = (sortVec[m].first.indJ + 1);
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during converting \n",duration);
	
	//---------------
	// free
	//---------------
	free(weight);
	free(WeightInd);
	free(IndforI);
	free(IndforJ);
	free(IndforK);
	free(IndforKp);
	free(freq_k);

}

int sample_index(size_t S, size_t *index, \
				 size_t *IndforI, size_t *IndforK, \
				 size_t *freq_k, \
				 size_t m, size_t n, \
				 double*pdf, double sum_pdf){
	// pdf has size (m, n) the sample 
	// and the sampled index = k * n + i;
	// First stage : get S uniform random numbers
	std::vector<double> rand_u;
	for (size_t i = 0; i < S; ++i){
		rand_u.push_back(sum_pdf*((double)rand()/(double)RAND_MAX));
	}
	// Sort the random values
	// It will be sorted according to k then i;
	sort(rand_u.begin(),rand_u.end());
	size_t ind = 0;
	size_t range = m * n;
	double sum_prob = pdf[0];
	for (size_t i = 0; i < S; ++i){
		while((rand_u[i] >= sum_prob) && (ind < (range-1))){
			sum_prob += pdf[++ind];
		}
		index[i] = ind;
		IndforI[i] = ind % n;
		IndforK[i] = ind / n;
		freq_k[IndforK[i]] ++;
	}
	return 1;
}

/* 
	Vose's alias method for sample
*/
int vose_alias(size_t s, size_t *dst, \
			   size_t n, double *pdf,double sum_pdf){
	double *scaled_prob = new double[n];
	double *table_prob = new double[n];
	size_t *table_alias = new size_t[n];
	size_t *table_small = new size_t[n];
	size_t *table_large = new size_t[n];
	size_t small_index = 0;
	size_t large_index = 0;
	/* stage 1: initialization */
	for (size_t i = 0; i < n; ++i){
		scaled_prob[i] = abs(*(pdf+i)) * n;
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
			*(dst + i) = fair_die;
		}else{
			*(dst + i) = table_alias[fair_die];
		}
	}
	delete []table_prob;
	delete []table_alias;
	delete []scaled_prob;
	delete []table_small;
	delete []table_large;
	return 1;
}