/*
	Diamond Sampling with N-1 factor matrices and a query matrix
	It will return a sparse tensor stored 
	the sampled result
*/

#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "matrix.h"

typedef std::pair<point3D,double> indValue;

int cmp(const indValue &x,const indValue&y){
	return (x.second > y.second);
}

/*
	suppose the dimension of feature is d;
	The first matrix has d rows;
	The left matrices have d columns;

*/

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	mexPrintf("Initialization >>>>\n");
	// MatA is a set of queries
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	// MatB and MatC are two factor matrices
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	// budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// sample number
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// kNN
	const size_t knn = (size_t)mxGetPr(prhs[5])[0];
	// sampling time for each query
	const size_t NumQueries = mxGetN(prhs[0]);
	// result values for each query
	plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
	double *knnValue = mxGetPr(plhs[0]);
	// sampling time for each query
	plhs[1] = mxCreateDoubleMatrix(NumQueries, 1, mxREAL);	
	double *SamplingTime = mxGetPr(plhs[1]);
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	mexPrintf("Initialization Complete!\n");

	//-------------------------------------
	// Compute weight
	//-------------------------------------
	mexPrintf("Start Computing weight!\n");
	double *weight = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(weight, 0, MatA.row*MatA.col*sizeof(double));
	double *SumofW = (double*)malloc(MatA.col*sizeof(double));
	memset(SumofW, 0, MatA.col*sizeof(double));
	int *isZero = (int*)malloc(MatA.col*sizeof(int));
	memset(isZero, 0, MatA.col*sizeof(int));
	double tempW = 0;
	//each query's weight is q'_r = |q_r|*||b_{*r}||_1||c_{*r}||_1
	for(size_t i = 0; i < MatA.col; ++i){
		start = clock();
		SumofW[i] = 0;
		for (size_t r = 0; r < MatA.row; ++r){
			// for each query do
			// w_{ri} = |a_{ri}|*||b_{*r}||_1||c_{*r}||_1
			tempW = 1;
			tempW *= abs(MatA.GetElement(r,i));
			tempW *= MatB.SumofCol[r];
			tempW *= MatC.SumofCol[r];
			weight[i*MatA.row + r] = tempW;
			SumofW[i] += tempW;
			if(SumofW[i] == 0){
				isZero[i] = 1;
			}
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
	}
	mexPrintf("Computing weight complete!\n");
	//-------------------------------
	// Compute c_r for each query
	//-------------------------------
	mexPrintf("Start computing c_r!\n");
	size_t *freq_r = (size_t*)malloc(MatA.row*MatA.col*sizeof(size_t));
	memset(freq_r, 0, MatA.row*MatA.col*sizeof(size_t));
	// c_r has the expectation NumSample*q'_r*/|q'|_1
	for(size_t i = 0; i < MatA.col; ++i){
		if(isZero[i] == 1){
			continue;
		}
		start = clock();
		for (size_t r = 0; r < MatA.row; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			// NumSample*q'_r*/|q'|_1
			double c = (double)NumSample*weight[i*MatA.row + r]/SumofW[i];
			if(u < (c - floor(c)))
				freq_r[i*MatA.row + r] = ceil(c);
			else
				freq_r[i*MatA.row + r] = floor(c);
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
	}
	//-------------------------
	// Do Sampling
	//-------------------------
	// list for sub walk
	mexPrintf("Start Sampling!\n");
	std::vector<std::vector<point2D>> subWalk(MatA.row);
	size_t *idxRp = (size_t*)malloc((NumSample + MatA.row)*sizeof(size_t));
	for(int i = 0; i < NumQueries; ++i){
		if(isZero[i] == 1){
			continue;
		}
		start = clock();
		// sample r' for this query
		memset(idxRp, 0, (NumSample + MatA.row)*sizeof(size_t));
		vose_alias((NumSample + MatA.row), idxRp, \
					MatA.row, \
					(MatA.element + i*MatA.row), \
					MatA.SumofCol[i]);
		// use map IrJc to save the sampled values
		std::map<point3D, double> IrJc;
		size_t offset = 0;
		for(size_t r = 0; r < MatA.row; ++r){
			// Check the list length for each query
			if(freq_r[i*MatA.row + r] > subWalk[r].size()){
				int remain = freq_r[i*MatA.row + r] - subWalk[r].size();
				size_t *IdxJ = (size_t*)malloc(remain*sizeof(size_t));
				size_t *IdxK = (size_t*)malloc(remain*sizeof(size_t));
				memset(IdxJ, 0, remain*sizeof(size_t));
				memset(IdxK, 0, remain*sizeof(size_t));
				vose_alias(remain, IdxJ, \
					MatB.row, \
					(MatB.element + r*MatB.row), \
					MatB.SumofCol[r]);
				vose_alias(remain, IdxK, \
					MatC.row, \
					(MatC.element + r*MatC.row), \
					MatC.SumofCol[r]);
				for(int p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
				free(IdxJ);
				free(IdxK);
			}
			for(int m = 0; m < freq_r[i*MatA.row + r]; ++m){
				// repeat c_r times to sample indexes j, k
				size_t rp = idxRp[offset];
				size_t idxJ = (subWalk[r])[m].x;
				size_t idxK = (subWalk[r])[m].y;
				double valueSampled = 1.0;
				valueSampled *= sgn_foo(MatA.GetElement(r,i));
				valueSampled *= sgn_foo(MatB.GetElement(idxJ,r));
				valueSampled *= sgn_foo(MatC.GetElement(idxK,r));
				valueSampled *= sgn_foo(MatA.GetElement(rp,i));
				valueSampled *= MatB.GetElement(idxJ,rp);
				valueSampled *= MatC.GetElement(idxK,rp);
				IrJc[point3D(i,idxJ,idxK)] += valueSampled;
				++offset;
			}
		}

		// compute the score for each query
		std::vector<indValue> tempSortedVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); mapItr++) {
			tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
		}
		sort(tempSortedVec.begin(),tempSortedVec.end(),cmp);
		
		std::vector<indValue> sortVec;
		std::map<point3D, double>::iterator mapItr;
		
		double true_value = 0;
		for(size_t i = 0; i < tempSortedVec.size() && i < budget; ++i){
			true_value = vectors_mul(tempSortedVec[i].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[i].first,true_value));
		}
		sort(sortVec.begin(),sortVec.end(),cmp);
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		SamplingTime[i] /= CLOCKS_PER_SEC;		
		if(sortVec.size() <= knn)
			printf("Warning:The size of sampled %s result is less then K!\n", sortVec.size());
		for(int s = 0; s < sortVec.size() && s < knn;++s){
			knnValue[i*knn + s] = sortVec[s].second;
		}
	}
	//---------------
	// free
	//---------------
	free(freq_r);
	free(idxRp);
	free(SumofW);
}
