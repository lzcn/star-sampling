#ifndef __DMATRIX_H__
#define __DMATRIX_H__


#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include "random.h"
#include "util.h"
const uint INIT_RAND_U = 1;
const uint INIT_RAND_N = 3;
const uint INIT_ALL_ZEROS = 4;
const uint INIT_ALL_ONES = 5;

class DMatrix;
class DVector;

struct SEntry {
    uint id;
    double value;
};
	
struct SVector {
	SEntry *data;
	uint size;
};

class DMatrix{
public:    
    DMatrix();
    DMatrix(uint r_dim, uint c_dim);
    DMatrix(uint r_dim, uint c_dim, double *ptr);
	~DMatrix();
    double& operator()(uint i, uint j);
	double operator()(uint i, uint j)const;
	uint randRow(uint n, double sum);
	uint randCol(uint m, double sum);
    void init(uint TYPE, double mean, double stdev);
    double* get_abs_row_sum();
    double* get_abs_col_sum();
public:
	uint row, col;
	double *value;
private:
    bool SELFINITVALUE;
    
};

class DVector{
public:
    DVector();
    DVector(uint v_dim);
    DVector(uint v_dim, double*ptr);
    void init(uint TYPE, double mean, double stdev);
	~DVector();
    double& operator()(uint n);
	double operator()(uint n)const;
public:
	uint dim;
	double* value;
private:
    bool SELFINITVALUE;
    
};

DMatrix::~DMatrix(){
    if(SELFINITVALUE)
        free(value);
}

DMatrix::DMatrix(){
    row = 0;
    col = 0;
    value = nullptr;
    SELFINITVALUE = false;
}
DMatrix::DMatrix(uint r_dim, uint c_dim){
    row = r_dim;
    col = c_dim;
    value = nullptr;
    SELFINITVALUE = false;
}
DMatrix::DMatrix(uint r_dim, uint c_dim, double *ptr){
    row = r_dim;
    col = c_dim;
    value = ptr;
    SELFINITVALUE = false;
}
void DMatrix::init(uint TYPE, double mean, double stdev){
    if(value == nullptr){
        value = (double*)malloc(col*row*sizeof(double));
        SELFINITVALUE = true;
    }
    switch (TYPE) {
        case INIT_RAND_N:
            for(uint i = 0; i < col*row; ++i){
                // normal distribution with \mu = mean,\sigma^2 = stdev
                value[i] = rnd::randn(mean, stdev);
            }break;
        case INIT_RAND_U:
            for(uint i = 0; i < col*row; ++i)
                value[i] = rnd::randu();
            break;
        case INIT_ALL_ONES:
            memset(value, 1, col*row*sizeof(double));
            break;
        case INIT_ALL_ZEROS:
            memset(value, 0, col*row*sizeof(double));
            break;
        default: break;
    }
}
double& DMatrix::operator()(uint i, uint j){ 
    //assert((i >= row) && (j >= col));
    return value[j*row +i];
}
double DMatrix::operator()(uint i, uint j) const {
    return value[j*row +i];
}
uint DMatrix::randRow(uint n, double sum){
    double x,temp;
    x = sum*((double)rand()/(double)RAND_MAX);
    temp = 0;
    for (uint i = 0; i < row; ++i){
        temp += abs(value[i + n*row]);
        if(x <= temp){ 
            return i;
        }
    }
    return (row-1);
}
uint DMatrix::randCol(uint m, double sum){
    double x,temp;
    x = sum*((double)rand()/(double)RAND_MAX);
    temp = 0;
    for (uint j = 0; j < col; ++j){
        temp += abs(value[m + j*row]);
        if(x <= temp){ 
            return j;
        }
    }
    return (col-1);
}

DVector::DVector(){
    dim = 0;
    value = nullptr;
    SELFINITVALUE = false;
}
DVector::DVector(uint v_dim){
    dim = v_dim;
    value = nullptr;
    SELFINITVALUE = false;
}
DVector::DVector(uint v_dim, double *ptr){
    dim = v_dim;
    value = ptr;
    SELFINITVALUE = false;
}
double& DVector::operator()(uint n){ 
    //assert((i >= row) && (j >= col));
    return value[n];
}
double DVector::operator()(uint n) const {
    return value[n];
}

void DVector::init(uint TYPE, double mean, double stdev){
    if(value == nullptr){
        value = (double*)malloc(dim*sizeof(double));
        SELFINITVALUE = true;
    }
    switch (TYPE) {
        case INIT_RAND_N:
            for(uint i = 0; i < dim; ++i)
                // normal distribution with \mu = mean,\sigma^2 = stdev
                value[i] = rnd::randn(mean, stdev);
            break;
        case INIT_RAND_U:
            for(uint i = 0; i < dim; ++i)
                value[i] = rnd::randu();
            break;
        case INIT_ALL_ONES:
            memset(value, 1, dim*sizeof(double));
            break;
        case INIT_ALL_ZEROS:
            memset(value, 0, dim*sizeof(double));
            break;
        default: break;
    }
}

DVector::~DVector(){
    if(SELFINITVALUE)
        free(value);
}
#endif
