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
		double temp = 0;
		for(size_t i = 0; i < col; ++i){
			temp = 0;
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
		double x,sum,temp;
		sum = SumofCol[n];
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
				 size_t dim, int *freq_k,\
				 size_t L, double*pdf, double sum_pdf);

/* 
	Vose's alias method for sample;
	It is used for the situation when 
		L is not much bigger than S;
	A method with constant time per sample.
*/
int vose_alias(size_t s, size_t *dst, \
			   size_t n, double *pdf,double sum_pdf);
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

	double SumofW = 0;
	double *weight = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(weight, 0, MatA.row*MatA.col*sizeof(double));
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
	memset(WeightInd, 0, NumSample*sizeof(size_t));
	int *freq_k = (int*)malloc(MatA.row*sizeof(int));
	memset(freq_k, 0, MatA.row*sizeof(int));
	// sample S pairs (k, i) ,
	sample_index(NumSample, WeightInd, \
				 MatA.row, freq_k,\
				 MatA.row*MatA.col, weight, SumofW);
	// k, i, j, k'
	size_t *IndforI = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforI, 0, NumSample*sizeof(size_t));
	size_t *IndforJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforJ, 0, NumSample*sizeof(size_t));
	size_t *IndforKp = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforKp, 0, NumSample*sizeof(size_t));
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
					(MatB.element + k*MatB.row), \
					MatB.SumofCol[k]);
		offset += freq_k[k];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t indi,indj,indk,indkp;
	std::map<struct indexIJ, double> IrJc;
	for (int s = 0; s < NumSample ; ++s){
		// sample k'
		indk = WeightInd[s] % MatA.row;
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
	free(IndforKp);
	free(freq_k);

}

int sample_index(size_t n, size_t *index, \
				 size_t row, int *freq_k,\
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
		freq_k[k % row] ++;
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