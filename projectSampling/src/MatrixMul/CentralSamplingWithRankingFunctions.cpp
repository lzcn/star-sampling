#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstring>
#include "mex.h"
#include "matrix.h"

int EuclideanComp(const pidx2d &x, const pidx2d &y){
	return (x.second < y.second);
}
int CosineComp(const pidx2d &x, const pidx2d &y){
	return (x.second > y.second);
}
double EuclideanScore(size_t i, size_t j, size_t r, const Matrix &A, const Matrix &B){
	size_t row = A.row;
	double a = A.element[i * row + r];
	double b = B.element[j * row + r];
	return ((a-b)*(a-b)/abs(a*b));
	
}
double ConsineScore(size_t i, size_t j, size_t r, const Matrix &A, const Matrix &B){
	size_t row = A.row;
	double a = A.element[i * row + r];
	double b = B.element[j * row + r];
	double normA = 0.0;
	double normB = 0.0;
	for(size_t t = 0; t < row; ++t){
		normA += A.element[i * row + r] * \
				 A.element[i * row + r];
		normB += B.element[j * row + r] * \
				 B.element[j * row + r];
	}
	return (sgn_foo(a*b)/(sqrt(normA)*sqrt(normB)));
}

void mexFunction (size_t nlhs, mxArray *plhs[], size_t nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	double (*metric)(const point2D&, const Matrix&B, const Matrix&A);
	double (*score)(size_t, size_t, size_t, const Matrix&A, const Matrix&B);
	int (*comp)(const pidx2d &x, const pidx2d &y);
	//--------------------
	// Initialization
	//--------------------
	start = clock();
	// get matrices
	size_t rankSize = mxGetN(prhs[0]);
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	// get the budget
	const size_t budget = (size_t)mxGetPr(prhs[2])[0];
	// get number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
	// get the top-t 
	const size_t top_t = (size_t)mxGetPr(prhs[4])[0];
	// get the type of ranking functions
	char *type = mxArrayToString(prhs[5]);
	if(!strcmp(type,"Euclidean")){
		metric = EuclideanMetricRow;
		score = EuclideanScore;
		comp = EuclideanComp;
		mexPrintf("Using Ranking Function:Euclidean Metric...\n");
	}else if(!strcmp(type,"Cosine")){
		metric = CosineMetricRow;
		score = ConsineScore;
		comp = CosineComp;
		mexPrintf("Using Ranking Function:Cosine Similarity...\n");
	}else{
		mexPrintf("No Match Ranking Functions...\n");
		return;
	}
	// result of sampling
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	
	// time duration for sampling
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	*tsec = duration;
	
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 2, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);
	
	//-------------------------------------
	// Compute weight for indexes r
	//-------------------------------------
	start = clock();
	double SumofW = 0;
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0, rankSize*sizeof(double));
	double tempW = 0;
	for (size_t r = 0; r < rankSize; ++r){
		weight[r] = MatA.SumofCol[r];
		weight[r] *= MatB.SumofCol[r];
		SumofW += weight[r]; 
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	size_t *freq_r = (size_t *)malloc(rankSize*sizeof(size_t));
	memset( freq_r, 0, rankSize*sizeof(size_t));
	for (size_t r = 0; r < rankSize; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (size_t)ceil(c);
		else
			freq_r[r] = (size_t)floor(c);
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	// sample the idx i,j
	size_t *IdxI = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxI, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxJ = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxJ, 0, (NumSample + rankSize)*sizeof(size_t));
	// sample indexes
	start = clock();
	for (size_t r = 0,offset = 0; r < rankSize; ++r){
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
		offset += freq_r[r];
	}
	// compute update value and saved in map<pair, value>
	size_t idxi, idxj;
	// use map IrJc to save the sampled values
	std::map<point2D, double> IrJc;
	for(size_t r = 0,offset = 0; r < rankSize; ++r){
		for(size_t s = 0; s < freq_r[r]; ++s,++offset){
			idxi = IdxI[offset];
			idxj = IdxJ[offset];
			IrJc[point2D(idxi, idxj)] += score(idxi,idxj,r,MatA,MatB);
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// for pre sort
	std::vector<pidx2d> tempSortedVec;
	// sort by actual value
	std::vector<pidx2d> sortVec;
	// push the value into a vector for sorting
	std::map<point2D, double>::iterator mapItr;
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	start = clock();
	sort(tempSortedVec.begin(), tempSortedVec.end(), comp);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;

	start = clock();
	double true_value = 0;
	// compute the top-t' (budget) actual value
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		true_value = metric(tempSortedVec[m].first, MatA, MatB);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), comp);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration; 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	// value
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//i
		plhs_pr[m] = (sortVec[m].first.x + 1);
		//j
		plhs_pr[m + top_t] = (sortVec[m].first.y + 1);
	}
	
	//---------------
	// free
	//---------------
	free(IdxI);
	free(IdxJ);
	free(freq_r);
	free(weight);
}
