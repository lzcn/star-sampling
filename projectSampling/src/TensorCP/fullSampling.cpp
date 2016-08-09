#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "matrix.h"
double getValue(const point3D &coord, \
				   Matrix &A, \
				   Matrix &B, \
				   Matrix &C){
	double ans = 0;
    for (size_t k = 0; k < A.col; ++k){
        ans += A.GetElement(coord.x,k) * \
        	   B.GetElement(coord.y,k) * \
        	   C.GetElement(coord.z,k);
    }
    return ans;
}
typedef std::pair<point3D,double> indValue;

int cmp(const indValue &x,const indValue&y){
	return (x.second > y.second);
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	start = clock();
	// input
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	const size_t top_t = (size_t)mxGetPr(prhs[5])[0];
	// output
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);	
	//-------------------------------------
	// Compute A, B, C
	//-------------------------------------
	size_t rankSize = MatA.col;
	double *A = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(A, 0, MatA.row*MatA.col*sizeof(double));
	double *B = (double*)malloc(MatB.row*MatB.col*sizeof(double));
	memset(B, 0, MatB.row*MatB.col*sizeof(double));
	double *C = (double*)malloc(MatC.row*MatC.col*sizeof(double));
	memset(C, 0, MatC.row*MatC.col*sizeof(double));
	
	for(size_t i = 0; i < MatA.row; ++i){
		for (size_t r = 0; r < MatA.col; ++r){
			double tempW = 1.0;
			tempW *= abs(MatA.GetElement(i,r));
			tempW *= MatA.SumofRow[i];
			tempW *= MatB.SumofCol[r];
			tempW *= MatC.SumofCol[r];
			A[r*MatA.row + i] = tempW;
		}
	}
	for(size_t j = 0; j < MatB.row; ++j){
		for (size_t r = 0; r < MatB.col; ++r){
			double tempW = 1.0;
			tempW *= abs(MatB.GetElement(j,r));
			tempW *= MatB.SumofRow[j];
			tempW *= MatA.SumofCol[r];
			tempW *= MatC.SumofCol[r];
			B[r*MatB.row + j] = tempW;
		}
	}
	for(size_t k = 0; k < MatA.row; ++k){
		for (size_t r = 0; r < MatA.col; ++r){
			double tempW = 1.0;
			tempW *= abs(MatC.GetElement(k,r));
			tempW *= MatC.SumofRow[k];
			tempW *= MatA.SumofCol[r];
			tempW *= MatB.SumofCol[r];
			C[r*MatC.row + k] = tempW;
		}
	}
	Matrix Aex(mxGetM(prhs[0]), mxGetN(prhs[0]), A);
	Matrix Bex(mxGetM(prhs[1]), mxGetN(prhs[1]), B);
	Matrix Cex(mxGetM(prhs[2]), mxGetN(prhs[2]), C);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec = duration;
	mexPrintf("%f seconds during initialization\n",duration);
	//-------------------
	// Sample index r
	//-------------------
	start = clock();
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0, rankSize*sizeof(double));
	double SumofW = 0.0;
	for(size_t r = 0; r < rankSize; ++r){
		weight[r] = Aex.SumofCol[r];
		weight[r] *= Bex.SumofCol[r];
		weight[r] *= Cex.SumofCol[r];
		SumofW += weight[r]; 
	}
	size_t *IdxR = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxR, 0, NumSample*sizeof(size_t));
	vose_alias(NumSample, IdxR, rankSize, weight, SumofW);
	//---------------------------------
	// Sample i,j,k,r1,r2,r3
	//----------------------------------
	size_t *IdxI = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxI, 0, NumSample*sizeof(size_t));
	size_t *IdxJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxJ, 0, NumSample*sizeof(size_t));
	size_t *IdxK = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxK, 0, NumSample*sizeof(size_t));
	size_t *IdxRi = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxRi, 0, NumSample*sizeof(size_t));
	size_t *IdxRj = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxRj, 0, NumSample*sizeof(size_t));
	size_t *IdxRk = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxRk, 0, NumSample*sizeof(size_t));
	for (size_t s = 0; s < NumSample; ++s){
		size_t r = IdxR[s];
		IdxI[s] = Aex.randRow(r);
		IdxRi[s] = Aex.randCol(IdxI[s]);
		IdxJ[s] = Bex.randRow(r);
		IdxRj[s] = Bex.randCol(IdxJ[s]);
		IdxK[s] = Cex.randRow(r);
		IdxRk[s] = Cex.randCol(IdxK[s]);
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f seconds during Sampling\n",duration);
	//----------------------
	// Compute Scores
	//---------------------
	start = clock();
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	for (int s = 0; s < NumSample ; ++s){
		size_t i = IdxI[s];
		size_t j = IdxJ[s];
		size_t k = IdxK[s];
		size_t r = IdxR[s];
		size_t ri = IdxRi[s];
		size_t rj = IdxRj[s];
		size_t rk = IdxRk[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(MatA.GetElement(i,r));
		valueSampled *= sgn_foo(MatB.GetElement(j,r));
		valueSampled *= sgn_foo(MatC.GetElement(k,r));
		valueSampled *= sgn_foo(MatA.GetElement(i,ri));
		valueSampled *= MatB.GetElement(j,ri);
		valueSampled *= MatC.GetElement(k,ri);
		valueSampled *= MatA.GetElement(i,rj);
		valueSampled *= sgn_foo(MatB.GetElement(j,rj));
		valueSampled *= MatC.GetElement(k,rj);
		valueSampled *= MatA.GetElement(i,rk);
		valueSampled *= MatB.GetElement(j,rk);
		valueSampled *= sgn_foo(MatC.GetElement(k,rk));
		// Update the element in coordinate
		IrJc[point3D(i, j, k)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f seconds during Computing\n",duration);

	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	std::vector<indValue> tempSortedVec;
	std::vector<indValue> sortVec;
	std::map<point3D, double>::iterator mapItr;
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	start = clock();
	sort(tempSortedVec.begin(), tempSortedVec.end(), cmp);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f seconds during pre-sorting\n",duration);

	start = clock();
	double true_value = 0;
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		true_value = MatrixRowMul(tempSortedVec[m].first, MatA, MatB, MatC);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	sort(sortVec.begin(), sortVec.end(), cmp);

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f seconds during computing and sorting\n",duration);
 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//i
		plhs_pr[m] = (sortVec[m].first.x + 1);
		//j
		plhs_pr[m + top_t] = (sortVec[m].first.y + 1);
		//k
		plhs_pr[m + top_t + top_t] = (sortVec[m].first.z + 1);
	}
	//---------------
	// free
	//---------------
	free(A);
	free(B);
	free(C);
	free(weight);
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(IdxR);
	free(IdxRi);
	free(IdxRj);
	free(IdxRk);
}
