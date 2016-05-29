/*
	Diamond Sampling for matrix multiplication
	Author: Zhi Lu
	Reference:"Diamond Sampling for Approximate Maximum 
			All-pairs Dot-product(MAD) Search"
*/
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

struct graph 
{
  int V;
  int E;  
  mwIndex *ind; 
  mwIndex *ptr;
};
// struct for sparse matrix
struct Matrix
{
	size_t row;
	size_t col;
	size_t numofElement;
	double *element;
	// sum of every column or degree of every node
	double *SumofCol;
	// Jc is the cumulative number of  nnz of every column
	mwIndex *Jc;
	// Ir is the row index of every non zero element
	mwIndex *Ir;
	Matrix(const mxArray *pIn){
		row = mxGetM(pIn);
		col = mxGetN(pIn);
		element = mxGetPr(pIn);
		SumofCol = (double*)malloc(col*sizeof(double));
		memset(SumofCol, 0, col*sizeof(double));
		numofElement = mxGetNzmax(pIn);
		Jc = mxGetJc(pIn);
		Ir = mxGetIr(pIn);
		for(size_t i = 0; i < col; ++i){
			SumofCol[i] = (double)(Jc[i+1] - Jc[i]);
		}
	}
	~Matrix(){
		free(SumofCol);
	}
	bool isNone(size_t m, size_t n){
		size_t nnz_n = Jc[n+1] - Jc[n];
  		// if the n-th column has no element 
  		// then matrix(m, n) = 0
  		if (nnz_n == 0){
  			return true;
  		}else{
  			for(size_t i = Jc[n]; i < Jc[n+1]; ++i){
  		    	if(Ir[i] == m) return (false);
  		  	}
  		}
  		return true;
	}
	double GetEmelent(size_t m, size_t n){
  		size_t nnz_n = Jc[n+1] - Jc[n];
  		// if the n-th column has no element 
  		// then matrix(m, n) = 0
  		if (nnz_n == 0){
  			return 0.0;
  		}else{
  			for(size_t i = Jc[n]; i < Jc[n+1]; ++i){
  		    	if(Ir[i] == m) return (element[i]);
  		  	}
  		}
  		return 0.0;
	}
	size_t randRow(size_t n){
		// if the n-th colunm has no element
		if((Jc[n+1] - Jc[n]) == 0)
			return (size_t)(rand() % row);
		else{
			// the n-th column has (Jc[n+1] - Jc[n]) elements
			size_t x = (size_t)(rand() % (Jc[n+1] - Jc[n]));
			return (Ir[Jc[n] + x]);
		}
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
double vectors_mul(const struct indexIJ &coord, \
				struct Matrix &A, struct Matrix &B){
    double ans = 0;
    size_t indI = coord.indI;
    size_t indJ = coord.indJ;
    size_t m = A.Jc[indI];
    size_t n = A.Jc[indJ];
    if(indI == indJ){
    	return (A.Jc[indI+1] - A.Jc[indI]);
    }
    while(m < A.Jc[indI+1] && n < A.Jc[indJ+1]){
    	if(A.Ir[m] == A.Ir[n]){
    		ans += 1;
    	}else if(A.Ir[m] < A.Ir[n]){
    		m++;
    	}else if(A.Ir[m] < A.Ir[n]){
    		n++;
    	}
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
int sample_index(size_t S, size_t *index, \
				 size_t numofElement, \
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
	Find the index x in which column of matrix
	If Jc[i-1] <= x  < Jc[i] 
	then x is the (i-1)-th column element
*/   

size_t find_col(mwIndex *Jc, size_t col, size_t x)
{ 
  size_t i;
  for( i = 0; x >= Jc[i+1] && i < col;++i);
  if (i < col)
  {
    return(i);
  }
  return(col-1);
}

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

	clock_t start, finish;
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	start = clock();
	struct Matrix MatA(prhs[0]);
	struct Matrix MatB(prhs[1]);
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
	double *weight = (double*)malloc(MatA.numofElement*sizeof(double));
	memset(weight, 0, MatA.numofElement*sizeof(double));
	double tempW = 0;
	// only compute the nonzero element of A
	for(size_t i = 0; i < MatA.col; ++i){
		// the i-the column 's index
		// is start from MatA.Jc[i] to  MatA.Jc[i+1]-1
		for(size_t nnz = MatA.Jc[i]; nnz < MatA.Jc[i+1]; ++nnz){
			// if A(k,i) is the element
			// then A(k,i) = element[nnz]
			// k = MatA.Ir[nnz]
			// tempW *= abs(MatA.element[nnz]);
			tempW = 1;
			tempW *= MatA.SumofCol[i];
			tempW *= MatB.SumofCol[MatA.Ir[nnz]];
			weight[nnz] = tempW;
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
	// sampled index for  k, i, j, k'
	size_t *IndforK = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforK, 0, NumSample*sizeof(size_t));	
	size_t *IndforI = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforI, 0, NumSample*sizeof(size_t));
	size_t *IndforJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforJ, 0, NumSample*sizeof(size_t));
	size_t *IndforKp = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforKp, 0, NumSample*sizeof(size_t));
	// Do sample S pairs (k, i)
	sample_index(NumSample, WeightInd, MatA.numofElement, weight, SumofW);

	for(size_t i = 0; i < NumSample; ++i){
		// get the column of A which is i
		IndforI[i] = find_col(MatA.Jc, MatA.col, WeightInd[i]);
		// get the row of A which is k
		IndforK[i] = MatA.Ir[WeightInd[i]];
		// given i sample the k' from the i-th column of A
		IndforKp[i] = MatA.randRow(IndforI[i]);
		// given k, sample the j from the k-tn column of B
		IndforJ[i] = MatB.randRow(IndforK[i]);
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t indi,indj,indk,indkp;
	std::map<struct indexIJ, double> IrJc;
	for(size_t s = 0; s < NumSample ; ++s){
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
		true_value =  vectors_mul(mapItr->first, MatA, MatB);
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
	
	//----------------
	// free and delete
	//----------------
	free(weight);
	free(WeightInd);
	free(IndforI);
	free(IndforJ);
	free(IndforK);
	free(IndforKp);

}

int sample_index(size_t S, size_t *index, \
				 size_t numofElement, \
				 double*pdf, double sum_pdf){
	std::vector<double> rand_u;
	for (size_t i = 0; i < S; ++i){
		rand_u.push_back(sum_pdf*((double)rand()/(double)RAND_MAX));
	}
	// Sort the random values
	sort(rand_u.begin(),rand_u.end());
	size_t ind = 0;
	double sum_prob = pdf[0];
	for (size_t i = 0; i < S; ++i){
		while((rand_u[i] >= sum_prob) && (ind < (numofElement-1))){
			sum_prob += pdf[++ind];
		}
		index[i] = ind;
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