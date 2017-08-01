#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "utilmex.h"
#include "matrix.h"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	uint L_a = (uint)mxGetM(prhs[0]);
	uint L_b = (uint)mxGetM(prhs[1]);
	uint L_c = (uint)mxGetM(prhs[2]);
	uint rankSize = (uint)mxGetN(prhs[0]);
	uint rankSizeExt = rankSize * rankSize;
	// normal matrix
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint knn = (uint)mxGetPr(prhs[5])[0];
	// number of queries
	const uint NumQueries = (uint)mxGetM(prhs[0]);
	// result values for each query
	plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
	double *knnValue = mxGetPr(plhs[0]);
	// sampling time for each query
	plhs[1] = mxCreateDoubleMatrix(NumQueries, 1, mxREAL);	
	double *SamplingTime = mxGetPr(plhs[1]);
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	mexPrintf("Core^2 Sampling for MultiUsers:");
	mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d,Number of Queries:%d\n",knn,(int)log10(NumSample),(int)log10(budget),NumQueries);
	mexEvalString("drawnow");
	progressbar(0);
	//-------------------------
	// extension for matrices
	//-------------------------
	start = clock();
	Matrix AT(rankSize,L_a);
	Matrix BT(rankSize,L_b);
	Matrix CT(rankSize,L_c);
	AT.transpose(A);
	BT.transpose(B);
	CT.transpose(C);
	double *Aex = (double*)malloc(L_a*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(L_b*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(L_c*rankSizeExt*sizeof(double));
	memset(Aex, 0, L_a*rankSizeExt*sizeof(double));
	memset(Bex, 0, L_b*rankSizeExt*sizeof(double));
	memset(Cex, 0, L_c*rankSizeExt*sizeof(double));
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			size_t r = m * rankSize + n;
			for(uint i = 0; i < L_a; ++i){
				Aex[r * L_a + i] = abs(A[m * L_a + i] * A[n * L_a + i]);
			}
			double sum = 0;
			for(uint j = 0; j < L_b; ++j){
				sum += abs(B[m * L_b + j] * B[n * L_b + j]);
				Bex[r * L_b + j] = sum;
			}
			sum = 0;
			for(uint k = 0; k < L_c; ++k){
				sum += abs(C[m * L_c + k] * C[n * L_c + k]);
				Cex[r * L_c + k] = sum;
			}
		}
	}
	Matrix MatAex(L_a, rankSizeExt, Aex, MATRIX_NONE_SUM);
	Matrix MatBex(L_b, rankSizeExt, Bex, MATRIX_NONE_SUM);
	Matrix MatCex(L_c, rankSizeExt, Cex, MATRIX_NONE_SUM);
	finish = clock();
	for (uint i = 0; i < NumQueries; i++) {
		SamplingTime[i] = (double)(finish - start)/(CLOCKS_PER_SEC*NumQueries);
	}
	// extension matrices
	//-------------------------
	// Do Sampling
	//-------------------------
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	size_t *freq_r = (size_t *)malloc(NumQueries*rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, NumQueries*rankSizeExt*sizeof(size_t));
	uint *IdxJ = (uint*)malloc(NumSample*sizeof(uint));
	uint *IdxK = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxJ, 0, NumSample*sizeof(uint));
	memset(IdxK, 0, NumSample*sizeof(uint));
	//-----------------------
	// Sampling the Indexes
	//-----------------------
	std::vector<std::vector<point2D>> subWalk(rankSizeExt);
	double SumofW = 0.0;
	for(uint i = 0; i < NumQueries; ++i){
		clearprogressbar();
		progressbar((double)i/NumQueries);
		start = clock();
        SumofW = 0.0;
		for (uint r = 0; r < rankSizeExt; ++r){
			weight[r] = MatAex(i,r);
			weight[r] *= MatBex(L_b-1,r);
			weight[r] *= MatCex(L_c-1,r);
			SumofW += weight[r]; 
		}
		finish = clock();
		SamplingTime[i] += (double)(finish - start)/CLOCKS_PER_SEC;
		if(SumofW == 0){
			continue;
		}
		// sample freq_r[r]
		start = clock();
		for (uint r = 0; r < rankSizeExt; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			double c = (double)NumSample*weight[r]/SumofW;
			if(u < (c - floor(c)))
				freq_r[r] = (size_t)ceil(c);
			else
				freq_r[r] = (size_t)floor(c);
		}
		TPoint3DMap IrJc;
        for(uint m = 0; m < rankSize; ++m){
    		for(uint n = 0; n < rankSize; ++n){
                size_t r = m * rankSize + n;
			    if(freq_r[r] > subWalk[r].size()){
                    size_t remain = freq_r[r] - subWalk[r].size();
				    // sample indexes
				    binary_search(remain, IdxJ, L_b, (MatBex.element + r*L_b));
				    binary_search(remain, IdxK, L_c, (MatCex.element + r*L_c));
			        for(uint p = 0; p < remain; ++p){
					   subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
			        }
			    }
			    for(size_t p = 0; p < freq_r[r]; ++p){
				    uint idxJ = (subWalk[r])[p].x;
				    uint idxK = (subWalk[r])[p].y;
                    int score = sgn(AT(i,m))*sgn(AT(i,n));
                    score *= sgn(BT(idxJ,m))*sgn(BT(idxJ,n));
                    score *= sgn(CT(idxK,m))*sgn(CT(idxK,n));
				    IrJc[point3D(i,idxJ,idxK)] += (double)score;
			    }
            }
		}
		finish = clock();
		SamplingTime[i] += (double)(finish - start)/CLOCKS_PER_SEC;
		//-----------------------------------
		//sort the values have been sampled
		//-----------------------------------
		std::vector<pidx3d> tempSortedVec;
		std::vector<pidx3d> sortVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
			tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
		}
		start = clock();
		sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
		for(size_t t = 0; t < tempSortedVec.size() && t < budget; ++t){
			double true_value = MatrixColMul(tempSortedVec[t].first, AT, BT, CT);
			sortVec.push_back(std::make_pair(tempSortedVec[t].first,true_value));
		}
		sort(sortVec.begin(),sortVec.end(),compgt<pidx3d>);
		finish = clock();
		SamplingTime[i] += (double)(finish-start)/CLOCKS_PER_SEC;
		for(size_t s = 0; s < sortVec.size() && s < knn; ++s){
			knnValue[i*knn + s] = sortVec[s].second;
		}
	}
	clearprogressbar();
	progressbar(1);
	mexPrintf("\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(freq_r);
	free(Aex);
	free(Bex);
	free(Cex);
	free(IdxJ);
	free(IdxK);
}
