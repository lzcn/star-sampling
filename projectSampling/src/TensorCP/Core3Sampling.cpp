#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "utilmex.h"
#include "matrix.h"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	uint rankSize = (uint)mxGetN(prhs[0]);
	uint rankSizeExt = rankSize * rankSize * rankSize;
	uint Arow = (uint)mxGetM(prhs[0]);
	uint Brow = (uint)mxGetM(prhs[1]);
	uint Crow = (uint)mxGetM(prhs[2]);
	Timer sTime;
	// original matrices
	start = clock();
	Matrix MatA(Arow, rankSize, mxGetPr(prhs[0]), MATRIX_NONE_SUM);
	Matrix MatB(Brow, rankSize, mxGetPr(prhs[1]), MATRIX_NONE_SUM);
	Matrix MatC(Crow, rankSize, mxGetPr(prhs[2]), MATRIX_NONE_SUM);
	Matrix AT(rankSize,Arow);
	Matrix BT(rankSize,Brow);
	Matrix CT(rankSize,Crow);
	AT.transpose(mxGetPr(prhs[0]));
	BT.transpose(mxGetPr(prhs[1]));
	CT.transpose(mxGetPr(prhs[2]));
	sTime.r_init(start);
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint top_t = (uint)mxGetPr(prhs[5])[0];
	// result of sampling
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *values = mxGetPr(plhs[0]);
	// time duration sampling
	plhs[1] = mxCreateDoubleMatrix(1, 4, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* indexes = (uint64_T*)mxGetData(plhs[2]);
	mexPrintf("Starting Core^3 Sampling:");
	mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n",top_t,(int)log10(NumSample),(int)log10(budget));
	mexEvalString("drawnow");
	// compute the extension for matrices
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	start = clock();
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			for (uint h = 0; h < rankSize; ++h){
					double p = 0.0;
					for(uint i = 0; i < Arow; ++i){
						p += abs(MatA(i,m)*MatA(i,n)*MatA(i,h));
					}
					double q = 0.0;
					for(uint j = 0; j < Brow; ++j){
						q += abs(MatB(j,m)*MatB(j,n)*MatB(j,h));
					}
					double t = 0.0;
					for(uint k = 0; k < Crow; ++k){
						t += abs(MatC(k,m)*MatC(k,n)*MatC(k,h));
					}
					size_t r = m*rankSize*rankSize + n*rankSize + h;
					weight[r] = p*q*t;
					SumofW += p*q*t; 
			}
		}
	}
	sTime.r_init(start);
	mexPrintf("|-%f during the initialization phase.\n",sTime.initialization);
	mexEvalString("drawnow");
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	size_t TotalS = 0;
	size_t *freq_r = (size_t *)malloc(rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, rankSizeExt*sizeof(size_t));
	for (uint r = 0; r < rankSizeExt; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (size_t)ceil(c);
		else
			freq_r[r] = (size_t)floor(c);
		TotalS += freq_r[r];
	}
	
	uint *IdxI = (uint*)malloc(TotalS*sizeof(uint));
	memset(IdxI, 0, TotalS*sizeof(uint));
	uint *IdxJ = (uint*)malloc(TotalS*sizeof(uint));
	memset(IdxJ, 0, TotalS*sizeof(uint));
	uint *IdxK = (uint*)malloc(TotalS*sizeof(uint));
	memset(IdxK, 0, TotalS*sizeof(uint));
	// sample indexes
	double *pa = (double *)malloc(Arow*sizeof(double));
	double *pb = (double *)malloc(Brow*sizeof(double));
	double *pc = (double *)malloc(Crow*sizeof(double));
	memset( pa, 0, Arow*sizeof(double));
	memset( pb, 0, Brow*sizeof(double));
	memset( pc, 0, Crow*sizeof(double));
	size_t offset = 0;
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			for (uint h = 0; h < rankSize; ++h){
				double sum_a = 0.0;
				for(uint i = 0; i < Arow; ++i){
					sum_a += abs(MatA(i,m)*MatA(i,n)*MatA(i,h));
					pa[i] = sum_a;
				}
				double sum_b = 0.0;
				for(uint j = 0; j < Brow; ++j){
					sum_b += abs(MatB(j,m)*MatB(j,n)*MatB(j,h));
					pb[j] = sum_b;
				}
				// extension for matrix C
				double sum_c = 0.0;
				for(uint k = 0; k < Crow; ++k){
					sum_c += abs(MatC(k,m)*MatC(k,n)*MatC(k,h));
					pc[k] = sum_c;
				}
				size_t r = m*rankSize*rankSize + n*rankSize + h;
				binary_search(freq_r[r], (IdxI + offset), Arow, pa);
				binary_search(freq_r[r], (IdxJ + offset), Brow, pb);
				binary_search(freq_r[r], (IdxK + offset), Crow, pc);
				offset += freq_r[r];
			}
		}
	}
	sTime.r_samp(start);
	mexPrintf("|-%f during the sampling phase.\n",sTime.sampling);
	mexEvalString("drawnow");
	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	// std::map<point3D, double> IrJc;
	TPoint3DMap IrJc;
	offset = 0;
	start = clock();
	if(budget >= NumSample){
		for(size_t i = 0; i < TotalS; ++i){
			IrJc[point3D(IdxI[i], IdxJ[i], IdxK[i])] = 1;
		}
	}else{
		for (uint m = 0; m < rankSize; ++m){
			for (uint n = 0; n < rankSize; ++n){
				for (uint h = 0; h < rankSize; ++h){
					size_t r = m*rankSize*rankSize + n*rankSize + h;
					for(size_t s = 0; s < freq_r[r]; ++s){
						uint idxi = IdxI[offset];
						uint idxj = IdxJ[offset];
						uint idxk = IdxK[offset];
						double score = sgn(MatA(idxi,m))*sgn(MatA(idxi,n))*sgn(MatA(idxi,h));
						score *= sgn(MatB(idxj,m))*sgn(MatB(idxj,n))*sgn(MatB(idxj,h));
						score *= sgn(MatC(idxk,m))*sgn(MatC(idxk,n))*sgn(MatC(idxk,h));
						IrJc[point3D(idxi, idxj, idxk)] += score;
						++offset;
					}
				}
			}
		}
	}
	sTime.r_score(start);
	mexPrintf("|-%f during the scoring phase.\n",sTime.scoring);
	mexEvalString("drawnow");
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// for pre-sort
	start = clock();
	std::vector<pidx3d> tempSortedVec;
	std::vector<pidx3d> sortVec;
	// push the value into a vector for sorting
	for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		double true_value = MatrixColMul(tempSortedVec[m].first, AT, BT, CT);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), compgt<pidx3d>);
	sTime.r_filter(start);
 	mexPrintf("|-%f during the sorting phase.\n",sTime.filtering);
	mexEvalString("drawnow");
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	// value
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		values[m] = sortVec[m].second;
		//indexes
		indexes[m] = (sortVec[m].first.x + 1);
		indexes[m + top_t] = (sortVec[m].first.y + 1);
		indexes[m + top_t + top_t] = (sortVec[m].first.z + 1);
	}
	tsec[0] = sTime.initialization;
	tsec[1] = sTime.sampling;
	tsec[2] = sTime.scoring;
	tsec[3] = sTime.filtering;
	mexPrintf("Done!\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(freq_r);
	free(pa);
	free(pb);
	free(pc);
}
