/*
	Equality Sampling with N factor matrices
	Author: Zhi Lu
*/

#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "../../include/matrix.h"

typedef std::pair<point3D,double> indValue;

int cmp(const indValue &x,const indValue&y){
	return (x.second > y.second);
}

/*
	suppose the dimension of feature is d;
	same column size

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
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	const int top_t = (int)mxGetPr(prhs[3])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	size_t rankSize = mxGetN(prhs[0]);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during initialization\n",duration);

	//-------------------------------------
	// Compute weight
	//-------------------------------------

	start = clock();
	double SumofW = 0;
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0, rankSize*sizeof(double));
	double tempW = 0;
	for (int r = 0; r < rankSize; ++r){
		weight[r] = MatA.SumofCol[r];
		weight[r] *= MatA.SumofCol[r];
		weight[r] *= MatA.SumofCol[r];
		SumofW += weight[r]; 
	}
	size_t *freq_r = (size_t *)malloc(rankSize*sizeof(size_t));
	memset( freq_r, 0, rankSize*sizeof(size_t));
	for (size_t r = 0; r < rankSize; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		// NumSample*q'_r*/|q'|_1
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = ceil(c);
		else
			freq_r[r] = floor(c);
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computing weight\n",duration);

	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	size_t *IdxI = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxI, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxJ = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxJ, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxK = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxK, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxR = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxR, 0, (NumSample + rankSize)*sizeof(size_t));

	// sample indexes
	size_t offset = 0;
	for (int r = 0; r < rankSize; ++r){
		// sample i
		vose_alias( freq_r[r], (IdxI + offset), \
					MatA.row, \
					(MatA.element + r*MatA.row), \
					MatA.SumofCol[r]);
		// sample j
		vose_alias( freq_r[r], (IdxJ + offset), \
					MatB.row, \
					(MatB.element + r*MatB.row), \
					MatB.SumofCol[r]);	
		// sample k
		vose_alias( freq_r[r], (IdxK + offset), \
					MatC.row, \
					(MatC.element + r*MatC.row), \
					MatC.SumofCol[r]);						
		offset += freq_r[r];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t idxi, idxj, idxk;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	offset = 0;
	for(int r = 0; r < rankSize; ++r){
		for(int s = 0; s < freq_r[r]; ++s){
			idxi = IdxI[offset];
			idxj = IdxJ[offset];
			idxk = IdxK[offset];
			IrJc[point3D(idxi, idxj, idxk)] += 1;
			++offset;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during sampling\n",duration);

	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------

	start = clock();
	std::vector<indValue> sortVec;
	std::map<point3D, double>::iterator mapItr;
	double true_value = 0;
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		true_value = vectors_mul(mapItr->first, MatA, MatB, MatC);
		sortVec.push_back(std::make_pair(mapItr->first,true_value));
		//sortVec.push_back(std::make_pair(mapItr->first,mapItr->second));
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
	plhs[0] = mxCreateNumericMatrix(phls_row, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[0]);
	// value
	plhs[1] = mxCreateDoubleMatrix(phls_row, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[1]);
	for(size_t m = 0; m < sortVec.size(); ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//m
		plhs_pr[m] = (sortVec[m].first.x + 1);
		//n
		plhs_pr[m + phls_row] = (sortVec[m].first.y + 1);
		//p
		plhs_pr[m + phls_row + phls_row] = (sortVec[m].first.z + 1);
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during converting \n",duration);
	//---------------
	// free
	//---------------
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(IdxR);
	free(freq_r);

}