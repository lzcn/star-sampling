#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include "mex.h"
#include "matrix.h"

double getAbsValue(const point3D &coord, \
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
/*
    [p1, p2, p3] = getProbability(A,B,C,topIndexes)
*/
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//--------------------
	// Initialization
	//--------------------
    
    // rank size 
	size_t rankSize = mxGetN(prhs[0]);
	size_t rankSizeExt = rankSize * rankSize;
    //-----------------------------------
    // Get Matrices for each approaches
    //-----------------------------------
	// normal matrix
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
    Matrix MatA(mxGetM(prhs[0]), rankSize, A);
    Matrix MatB(mxGetM(prhs[1]), rankSize, B);
    Matrix MatC(mxGetM(prhs[2]), rankSize, C);
    // A transpose
    double *Atrs = (double*)malloc(mxGetM(prhs[0])*mxGetN(prhs[0])*sizeof(double));
    memset(Atrs, 0, mxGetM(prhs[0])*mxGetN(prhs[0])*sizeof(double));
    for(size_t m = 0; m < mxGetM(prhs[0]); ++m){
        for(size_t n = 0; n < mxGetN(prhs[0]); ++n){
            Atrs[m*rankSize + n] = MatA.GetElement(m,n);
        }
    }
    Matrix MatAtrs(rankSize, mxGetM(prhs[0]), Atrs);
    // extension for matrices
    double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSizeExt*sizeof(double));
    double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSizeExt*sizeof(double));
    double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSizeExt*sizeof(double));
    memset(Aex, 0, mxGetM(prhs[0])*rankSizeExt*sizeof(double));
    memset(Bex, 0, mxGetM(prhs[1])*rankSizeExt*sizeof(double));
    memset(Cex, 0, mxGetM(prhs[2])*rankSizeExt*sizeof(double));
    for (size_t m = 0; m < rankSize; ++m){
        for (size_t n = 0; n < rankSize; ++n){
            // extension for matrix A
            for(size_t i = 0; i < mxGetM(prhs[0]); ++i){
                Aex[(m*rankSize + n) * MatA.row + i] = A[m * MatA.row + i] * A[n * MatA.row + i];
            }
            // extension for matrix B
            for(size_t j = 0; j < mxGetM(prhs[1]); ++j){
                Bex[(m*rankSize + n) * MatB.row + j] = B[m * MatB.row + j] * B[n * MatB.row + j];
            }
            // extension for matrix C
            for(size_t k = 0; k < mxGetM(prhs[2]); ++k){
                Cex[(m*rankSize + n) * MatC.row + k] = C[m * MatC.row + k] * C[n * MatC.row + k];
            }
        }
    }
    // extension matrices
    Matrix MatAex(mxGetM(prhs[0]), rankSizeExt, Aex);
    Matrix MatBex(mxGetM(prhs[1]), rankSizeExt, Bex);
    Matrix MatCex(mxGetM(prhs[2]), rankSizeExt, Cex);
    //--------------------------------
    // Get the top indexes
    //--------------------------------
    std::vector<point3D> Idx;
    const size_t top_t = mxGetM(prhs[3]);
    double *topIndexes = mxGetPr(prhs[3]);
    for(size_t m = 0; m < top_t; ++m){
            size_t i = (size_t)topIndexes[m] - 1;
            size_t j = (size_t)topIndexes[top_t + m] - 1;
            size_t k = (size_t)topIndexes[top_t + top_t + m] - 1;
            Idx.push_back(point3D(i,j,k));
    }
    //-------------------------------------------
    // Get the probability of diamond sampling
    //-------------------------------------------
    double SumofDiamondWeight = 0;
	for (size_t r = 0; r < MatAtrs.row; ++r){
		for(size_t i = 0; i < MatAtrs.col; ++i){
			double tempW = 1;
			tempW *= abs(MatAtrs.GetElement(r,i));
			tempW *= MatAtrs.SumofCol[i];
			tempW *= MatB.SumofCol[r];
			tempW *= MatC.SumofCol[r];
			SumofDiamondWeight += tempW;
		}
	}
    //-------------------------------------------
    // Get the probability of equality sampling
    //-------------------------------------------
    double SumofEqualityW = 0;
	for (size_t r = 0; r < rankSize; ++r){
        double temp;
		temp = MatA.SumofCol[r];
		temp *= MatB.SumofCol[r];
		temp *= MatC.SumofCol[r];
		SumofEqualityW += temp; 
	}
    //-------------------------------------------
    // Get the probability of extension sampling
    //-------------------------------------------
    double SumofExtensionW = 0;
    for (size_t r = 0; r < rankSizeExt; ++r){
        double temp;
        temp = MatAex.SumofCol[r];
        temp *= MatBex.SumofCol[r];
        temp *= MatCex.SumofCol[r];
        SumofExtensionW += temp; 
    }
    //----------------------------------------
    // Compute the absoult value
    //----------------------------------------
    double *value = (double*)malloc(top_t*sizeof(double));
    memset(value, 0, top_t*sizeof(double));
    for (size_t i = 0; i < top_t; ++i) {
        value[i] = getAbsValue(Idx[i], MatA, MatB, MatC);
    }
    //----------------------------------------
    // Compute the probability
    //----------------------------------------
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
    double *p0 = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
    double *p1 = mxGetPr(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *p2 = mxGetPr(plhs[2]);
    for (size_t n = 0; n < top_t; ++n) {
        size_t i = Idx[n].x;
        p0[n] = (value[n]*MatAtrs.SumofCol[i])/SumofDiamondWeight;
        p1[n] = value[n]/SumofEqualityW;
        p2[n] = (value[n]*value[n])/SumofExtensionW;
    }
    //-------
    // Free
    //-------
    free(Aex);
    free(Bex);
    free(Cex);
    free(Atrs);
    free(value);
}
