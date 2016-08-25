/*
	Diamond Sapmling with N factor matrixes
	It will return a sparse tansor stored 
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

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	//---------------------
	// Initialization
	//---------------------
	//  get the number of samples
	const size_t budget = mxGetPr(prhs[nrhs-3])[0];
	const size_t NumSample = mxGetPr(prhs[nrhs-2])[0];
	const uint top_t = mxGetPr(prhs[nrhs-1])[0];
	const uint NumMat = (nrhs - 3);
	// vector to save factor matrices
	std::vector<Matrix*> Mats;
	start = clock();
	for (uint i = 0; i < NumMat; ++i){
		Mats.push_back(new Matrix(mxGetM(prhs[i]),mxGetN(prhs[i]),mxGetPr(prhs[i])));
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	// value
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	// result for time
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	*tsec = duration;
	// pair
	plhs[2] = mxCreateNumericMatrix(top_t, NumMat, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);
	mexPrintf("Starting Dimaond Sampling:");
	mexPrintf("- Top-%d ",top_t);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("......");mexEvalString("drawnow");
	//--------------------
	// Compute the weight
	//--------------------
	start = clock();
	uint rankSize = mxGetM(prhs[0]);
	uint numCol = mxGetN(prhs[0]);
	// Get the weight matrix
	double SumofW = 0;
	double *weight = (double*)malloc(rankSize*numCol*sizeof(double));
	memset(weight, 0, rankSize*numCol*sizeof(double));
	start = clock();
	for (uint r = 0; r < rankSize; ++r){
		for(uint i = 0; i < numCol; ++i){
			double tempW = abs(Mats[0]->GetElement(r,i));
			tempW *= Mats[0]->SumofCol[i];
			for (uint n = 1; n < NumMat; ++n){
				tempW *= Mats[n]->SumofCol[r];
			}
			weight[r*numCol + i] = tempW;
			SumofW += tempW;
		}
	}
	for (uint r = 0; r < rankSize; ++r){
		for(uint i = 0; i < numCol; ++i){
			weight[r*numCol + i] /= SumofW;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//--------------------
	// Sampling
	//--------------------

	// map to save the sampling results
	std::map<pointND,double> IrJc;
	// sampled index for vertexes
	std::vector<uint*> idxes(NumMat);
	for(uint i = 0; i < NumMat; ++i){
		idxes[i] = (uint*)malloc(NumSample*sizeof(uint));
		memset(idxes[i], 0, NumSample*sizeof(uint));
	}
	uint *IdxR = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxR, 0, NumSample*sizeof(uint));	
	// freqency of r
	size_t *freq_r = (size_t*)malloc(rankSize*sizeof(size_t));
	memset(freq_r, 0, rankSize*sizeof(size_t));
	// sample pairs
	binary_sample(NumSample, \
				 idxes[0], IdxR, \
				 freq_r, \
				 rankSize, numCol, \
				 weight, 1.0);
	// sample sub-paths
	for(uint n = 1; n < NumMat; ++n){
		uint offset = 0;	
		for (uint r = 0; r < rankSize; ++r){
			vose_alias( freq_r[r], (idxes[n] + offset), \
						Mats[n]->row, \
						(Mats[n]->element + r*Mats[n]->row), \
						Mats[n]->SumofCol[r]);
			offset += freq_r[r];
		}
	}
	// sample r' and compute socres
	std::vector<uint*> coordinates(NumSample);
	for (size_t s = 0; s < NumSample; ++s){
		coordinates[s] = (uint*)malloc(NumMat*sizeof(uint));
		memset(coordinates[s], 0, NumMat*sizeof(uint));
		uint r = IdxR[s];
		uint rp = Mats[0]->randRow(idxes[0][s]);
		double valueSampled = 1.0;
		valueSampled *= sgn_foo(Mats[0]->GetElement(r,idxes[0][s]));
		valueSampled *= sgn_foo(Mats[0]->GetElement(rp,idxes[0][s]));
		coordinates[s][0] = idxes[0][s];
		for(uint n = 1; n < NumMat; ++n){
			coordinates[s][n] = idxes[n][s];
			valueSampled *= sgn_foo(Mats[n]->GetElement(idxes[n][s],r)) \
							* Mats[n]->GetElement(idxes[n][s],rp);
		}
		IrJc[pointND(coordinates[s],NumMat)] += valueSampled;
	}
	start = clock();
	std::vector<pidxNd> sortVec;
	std::vector<pidxNd> tempSortedVec;
	for (auto mapItr = IrJc.begin(); mapItr != IrJc.end() ; ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	start = clock();
	sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidxNd>);
	finish = clock();
	*tsec += duration;
	// compute the actual of top-t'(budget)
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		double true_value = vectors_mul(tempSortedVec[m].first, Mats);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	sort(sortVec.begin(),sortVec.end(),compgt<pidxNd>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration; 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		for(uint n = 0; n < NumMat; ++n){
			plhs_pr[m + n * top_t] = sortVec[m].first.coord[n] + 1;
		}
	}
	mexPrintf("Done!\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(IdxR);
	free(freq_r);	
	for(auto itr = Mats.begin(); itr != Mats.end() ; ++itr){
		delete *itr;
	}
	for(auto itr = idxes.begin(); itr != idxes.end() ; ++itr){
		free(*itr);
	}
	for(auto itr = coordinates.begin(); itr != coordinates.end() ; ++itr){
		free(*itr);
	}
}
