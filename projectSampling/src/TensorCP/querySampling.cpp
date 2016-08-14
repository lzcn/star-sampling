#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "matrix.h"

double DiamondScore(size_t i, size_t,j ,size_t k, size_t r, size_t rp, Matrix &A, Matrix &B, Matrix &C){
	double valueSampled;
	valueSampled = sgn_foo(A.GetElement(i,r));
	valueSampled *= sgn_foo(B.GetElement(j,r));
	valueSampled *= sgn_foo(C.GetElement(k,r));
	valueSampled *= sgn_foo(A.GetElement(i,rp));
	valueSampled *= B.GetElement(j,rp);
	valueSampled *= C.GetElement(k,rp);
	return valueSampled;
}
double CentralScore(size_t i, size_t,j ,size_t k, size_t r, size_t rp, Matrix &A, Matrix &B, Matrix &C){
	double valueSampled = sgn_foo(A.GetElement(i,r));
	valueSampled *= sgn_foo(B.GetElement(j,r));
	valueSampled *= sgn_foo(C.GetElement(k,r));
	return valueSampled;
}
double ExtensionScore(size_t i, size_t,j ,size_t k, size_t r, size_t rp, Matrix &A, Matrix &B, Matrix &C){
	double valueSampled = sgn_foo(A.GetElement(i,r));
	valueSampled *= sgn_foo(B.GetElement(j,r));
	valueSampled *= sgn_foo(C.GetElement(k,r));
	retrun valueSampled;
}
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	// number of users
	const size_t NumQueries = mxGetM(prhs[0]);
	// the rank size
	size_t rankSize = mxGetN(prhs[0]);
	// score functions
	double (*score)(size_t i, size_t,j ,size_t k, size_t r, size_t rp, Matrix &A, Matrix &B, Matrix &C);
	// pointer to factor matrices
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
	// budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// sample number
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// kNN
	const size_t knn = (size_t)mxGetPr(prhs[5])[0];
	// method type
	char *type = mxArrayToString(prhs[6]);
    if(!strcmp(type,"Central")){
        score = CentralScore;
		rankSize = mxGetN(prhs[0]);
        mexPrintf("Central Sampling for Queries");
    }else if(!strcmp(type,"Diamond")){
        score = DiamondScore;
		rankSize = mxGetN(prhs[0]);
        mexPrintf("Diamond Sampling for Queries");
    }if(!strcmp(type,"Extension")){
		score = ExtensionScore;
		rankSize = mxGetN(prhs[0])*mxGetN(prhs[0]);
        mexPrintf("Extension Sampling for Queries");
	}else{
        mexPrintf("No Match Methods...\n");
        return;
    }
	// extension of matrices
	double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSize*sizeof(double));
	double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSize*sizeof(double));
	double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSize*sizeof(double));
	memset(Aex, 0, mxGetM(prhs[0])*rankSize*sizeof(double));
	memset(Bex, 0, mxGetM(prhs[1])*rankSize*sizeof(double));
	memset(Cex, 0, mxGetM(prhs[2])*rankSize*sizeof(double));
	// get the extension of factor matrices and poiter to them
	if(!strcmp(type,"Extension")){
		size_t La = mxGetM(prhs[0]);
		size_t Lb = mxGetM(prhs[1]);
		size_t Lc = mxGetM(prhs[2]);
		size_t R = mxGetN(prhs[0]);
		for (size_t m = 0; m < R; ++m){
			for (size_t n = 0; n < R; ++n){
				// extension for matrix A
				for(size_t i = 0; i < La; ++i)
					Aex[(m*R+n)*La+i] = A[m*La+i] * A[n*La+i];
				// extension for matrix B
				for(size_t j = 0; j < Lb; ++j)
					Bex[(m*R+n)*Lb+i] = B[m*Lb+i] * B[n*Lb+i];
				// extension for matrix C
				for(size_t k = 0; k < Lc; ++k)
					Cex[(m*R+n)*Lc+i] = C[m*Lc+i] * C[n*Lc+i];
			}
		}
		A = Aex;
		B = Bex;
		C = Cex;
	}
	// the matrices
	Matrix MatA(mxGetM(prhs[0]),rankSize, A);
	Matrix MatB(mxGetM(prhs[1]),rankSize, B);
	Matrix MatC(mxGetM(prhs[2]),rankSize, C);
	// output
	plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
	double *knnValue = mxGetPr(plhs[0]);
	// sampling time for each query
	plhs[1] = mxCreateDoubleMatrix(NumQueries, 1, mxREAL);	
	double *SamplingTime = mxGetPr(plhs[1]);
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	mexPrintf("- Top-%d ",top_t);
	mexPrintf("- Samples:%d ",NumSample);
	mexPrintf("- knn:%d ",knn);
	mexPrintf("......");
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double *weight = (double*)malloc(NumQueries*rankSize*sizeof(double));
	memset(weight, 0, NumQueries*rankSize*sizeof(double));
	double *SumofW = (double*)malloc(rankSize*sizeof(double));
	memset(SumofW, 0, rankSize*sizeof(double));
	int *isZero = (int*)malloc(rankSize*sizeof(int));
	memset(isZero, 0, rankSize*sizeof(int));
	for(size_t i = 0; i < rankSize; ++i){
		start = clock();
		for (size_t r = 0; r < rankSize; ++r){
			weight[i*rankSize + r] = abs(MatA.GetElement(i,r)) \
									* MatB.SumofCol[r] \
									* MatC.SumofCol[r];
			SumofW[i] += weight[i*rankSize + r];
		}
		if(SumofW[i] == 0){
			isZero[i] = 1;
		}
		finish = clock();
		SamplingTime[i] = (double)(finish-start);
	}
	//-------------------------------
	// Sampling for each query
	//-------------------------------
	size_t *freq_r = (size_t*)malloc(NumQueries*rankSize*sizeof(size_t));
	memset(freq_r, 0, NumQueries*rankSize*sizeof(size_t));
	double *pdf = (double*)malloc(rankSize*sizeof(double));
	memset(pdf, 0, rankSize*sizeof(double));
	size_t *idxRp = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(idxRp, 0, (NumSample + rankSize)*sizeof(size_t));
	// get c_r
	for(size_t i = 0; i < NumQueries; ++i){
		if(isZero[i] == 1){
			continue;
		}
		start = clock();
		for (size_t r = 0; r < rankSize; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			double c = (double)NumSample*weight[i*rankSize + r]/SumofW[i];
			if(u < (c - floor(c)))
				freq_r[i*rankSize + r] = (size_t)ceil(c);
			else
				freq_r[i*rankSize + r] = (size_t)floor(c);
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
	}
	//-------------------------
	// Do sub-path Sampling
	//-------------------------
	// list for sub walk
	std::vector<std::vector<point2D> > subWalk(rankSize);
	for(size_t i = 0; i < NumQueries; ++i){
		if(isZero[i] == 1){
			continue;
		}
		start = clock();
		// sample r' for this query if useing diaond sampling
		if(!strcmp(type,"Diamond"){
			// get the vector of i-th user
			for(size_t r = 0; r < rankSize; ++r){
				pdf[r] = MatA.GetElement(i,r);
			}
			vose_alias((NumSample + rankSize), idxRp, rankSize, pdf, MatA.SumofRow[i]);
		}
		// use map IrJc to save the sampled values
		std::map<point3D, double> IrJc;
		// save the sampled values
		for(size_t r = 0; r < NumQueries; ++r){
			// Check the list length for each query
			if(freq_r[i*NumQueries + r] > subWalk[r].size()){
				size_t remain = freq_r[i*NumQueries + r] - subWalk[r].size();
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
				for(size_t p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
				free(IdxJ);
				free(IdxK);
			}
			// use the pool of indexes to compute the sampled value
			for(size_t m = 0,offset = 0; m < freq_r[i*rankSize + r]; ++m){
				size_t rp = idxRp[offset++];
				size_t j = (subWalk[r])[m].x;
				size_t k = (subWalk[r])[m].y;
				IrJc[point3D(i,idxJ,idxK)] += score(i,j,k,r,rp,MatA,MatB,MatC);
			}
		}
		// pre-sorting the scores
		std::vector<pidx3d> tempSortedVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); mapItr++){
			tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
		}
		sort(tempSortedVec.begin(),tempSortedVec.end(),compgt<pidx3d>);
		std::vector<pidx3d> sortVec;
		
		// compute the actual value for top-t' indexes
		for(size_t t = 0; t < tempSortedVec.size() && t < budget; ++t){
			double true_value = MatrixRowMul(tempSortedVec[t].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[t].first, true_value));
		}
		sort(sortVec.begin(),sortVec.end(),compgt<pidx3d>);
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		SamplingTime[i] /= CLOCKS_PER_SEC;
		for(size_t s = 0; s < sortVec.size() && s < knn; ++s){
			knnValue[i*knn + s] = sortVec[s].second;
		}
	}
	mexPrintf("Done!\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(freq_r);
	free(idxRp);
	free(pdf);
	free(SumofW);
	free(isZero);
}
