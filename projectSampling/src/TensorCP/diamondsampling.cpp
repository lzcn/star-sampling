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
	start = clock();
	//  get the number of samples
	const size_t budget = mxGetPr(prhs[nrhs-3])[0];
	const size_t NumSample = mxGetPr(prhs[nrhs-2])[0];
	const size_t top_t = mxGetPr(prhs[nrhs-1])[0];
	const int MatNum = (nrhs - 3);
	// vector to save factor matrices
	std::vector<Matrix*> vMat;
	for (int i = 0; i < MatNum; ++i){
		vMat.push_back(new Matrix(mxGetM(prhs[i]), \
								mxGetN(prhs[i]), \
								mxGetPr(prhs[i])));
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during initialization\n",duration);	
	//--------------------
	// Compute the weight
	//--------------------
	start = clock();
	size_t rankSize = mxGetM(prhs[0]);
	size_t numCol = mxGetN(prhs[0]);
	// Get the weight matrix
	double SumofW = 0;
	double *weight = (double*)malloc(rankSize*numCol*sizeof(double));
	memset(weight, 0, rankSize*numCol*sizeof(double));
	double tempW = 0.0;
	start = clock();
	for (size_t k = 0; k < rankSize; ++k){
		for(size_t i = 0; i < numCol; ++i){
			tempW = 1;
			tempW *= abs(vMat[0]->GetElement(k,i));
			tempW *= vMat[0]->SumofCol[i];
			for (size_t j = 1; j < MatNum; ++j){
				tempW *= vMat[j]->SumofCol[k];
			}
			weight[k*numCol + i] = tempW;
			SumofW += tempW;
		}
	}
	for (size_t k = 0; k < rankSize; ++k){
		for(size_t i = 0; i < numCol; ++i){
			weight[k*numCol + i] /= SumofW;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computing weight\n",duration);

	//--------------------
	// Sampling
	//--------------------

	// map to save the sampling results
	std::map<pointND,double> IrJc;
	std::map<pointND,double> Tensor;
	// sampled index for vertexes
	size_t *WeightInd = (size_t *)malloc(NumSample*sizeof(size_t));
	memset(WeightInd, 0, NumSample*sizeof(size_t));
	size_t *IndforK = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforK, 0, NumSample*sizeof(size_t));	
	size_t *IndforKp = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforKp, 0, NumSample*sizeof(size_t));	
	std::vector<size_t*> coords(MatNum);
	for(int i = 0; i < MatNum; ++i){
		coords[i] = (size_t*)malloc(NumSample*sizeof(size_t));
		memset(coords[i], 0, NumSample*sizeof(size_t));
	}
	// freqency of k
	size_t *freq_k = (size_t*)malloc(rankSize*sizeof(size_t));
	memset(freq_k, 0, rankSize*sizeof(size_t));
	// sample pairs
	sample_index(NumSample, WeightInd, \
				 coords[0], IndforK, \
				 freq_k, \
				 rankSize, numCol, \
				 weight, 1.0);
	// sample k';
	for (int s = 0; s < NumSample; ++s){
		IndforKp[s] = vMat[0]->randRow(coords[0][s]);
	}
	// sample sub-path
	size_t offset = 0;	
	for(int n = 1; n < MatNum; ++n){
		offset = 0;
		for (int k = 0; k < rankSize; ++k){
			vose_alias( freq_k[k], (coords[n] + offset), \
						vMat[n]->row, \
						(vMat[n]->element + k*vMat[n]->row), \
						vMat[n]->SumofCol[k]);
			offset += freq_k[k];
		}		
	}
	double valueSampled = 1.0;
	size_t idxk, idkp;
	for (size_t s = 0; s < NumSample; ++s){
		idxk = IndforK[s];
		idkp = IndforKp[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(vMat[0]->GetElement(idxk,coords[0][s]));
		valueSampled *= sgn_foo(vMat[0]->GetElement(idkp,coords[0][s]));
		for(int i = 1; i < MatNum; ++i){
			valueSampled *= sgn_foo(vMat[i]->GetElement(coords[i][s],idxk))*\
			vMat[i]->GetElement(coords[i][s],idkp);
		}
		IrJc[pointND(coords[s],MatNum)] += valueSampled;
	}
	start = clock();
	std::vector<pidxNd> sortVec;
	std::map<pointND, double>::iterator mapItr;
	double true_value = 0;
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		true_value = vectors_mul(mapItr->first, vMat);
		sortVec.push_back(std::make_pair(mapItr->first,true_value));
		//sortVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(sortVec.begin(),sortVec.end(),compgt<pidxNd>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computer and sorting tensor \n",duration);
 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	start = clock();
	size_t phls_row = sortVec.size();
	// value
	plhs[0] = mxCreateDoubleMatrix(phls_row, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	for(size_t m = 0; m < sortVec.size(); ++m){
		//value
		plhs_result[m] = sortVec[m].second;
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during converting \n",duration);
	//---------------
	// free
	//---------------
	free(weight);
	free(WeightInd);
	free(IndforK);
	free(IndforKp);
	free(freq_k);	
	std::vector<Matrix*>::iterator itr;
	for (itr = vMat.begin(); itr != vMat.end() ; ++itr){
		delete *itr;
	}
	std::vector<size_t *>::iterator idxitr;
	for (itr = vMat.begin(); itr != vMat.end() ; ++itr){
		free(*itr);
	}
}
