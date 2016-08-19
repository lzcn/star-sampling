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
    DMatrix(size_t r_dim, size_t c_dim);
    DMatrix(size_t r_dim, size_t c_dim, double *ptr);
	~DMatrix();
    double& operator()(size_t i, size_t j);
	double operator()(size_t i, size_t j)const;
	size_t randRow(size_t n, double sum);
	size_t randCol(size_t m, double sum);
    void init(uint TYPE, double mean, double stdev);
    double* get_abs_row_sum();
    double* get_abs_col_sum();
public:
	size_t row, col;
	double *value;
private:
    bool SELFINITVALUE;
    
};

class DVector{
public:
    DVector();
    DVector(size_t v_dim);
    DVector(size_t v_dim, double*ptr);
    void init(uint TYPE, double mean, double stdev);
	~DVector();
    double& operator()(size_t n);
	double operator()(size_t n)const;
public:
	size_t dim;
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
DMatrix::DMatrix(size_t r_dim, size_t c_dim){
    row = r_dim;
    col = c_dim;
    value = nullptr;
    SELFINITVALUE = false;
}
DMatrix::DMatrix(size_t r_dim, size_t c_dim, double *ptr){
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
            for(size_t i = 0; i < col*row; ++i){
                // normal distribution with \mu = mean,\sigma^2 = stdev
                value[i] = rnd::randn(mean, stdev);
            }break;
        case INIT_RAND_U:
            for(size_t i = 0; i < col*row; ++i)
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
double& DMatrix::operator()(size_t i, size_t j){ 
    //assert((i >= row) && (j >= col));
    return value[j*row +i];
}
double DMatrix::operator()(size_t i, size_t j) const {
    return value[j*row +i];
}
size_t DMatrix::randRow(size_t n, double sum){
    double x,temp;
    x = sum*((double)rand()/(double)RAND_MAX);
    temp = 0;
    for (size_t i = 0; i < row; ++i){
        temp += abs(value[i + n*row]);
        if(x <= temp){ 
            return i;
        }
    }
    return (row-1);
}
size_t DMatrix::randCol(size_t m, double sum){
    double x,temp;
    x = sum*((double)rand()/(double)RAND_MAX);
    temp = 0;
    for (size_t j = 0; j < col; ++j){
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
DVector::DVector(size_t v_dim){
    dim = v_dim;
    value = nullptr;
    SELFINITVALUE = false;
}
DVector::DVector(size_t v_dim, double *ptr){
    dim = v_dim;
    value = ptr;
    SELFINITVALUE = false;
}
double& DVector::operator()(size_t n){ 
    //assert((i >= row) && (j >= col));
    return value[n];
}
double DVector::operator()(size_t n) const {
    return value[n];
}

void DVector::init(uint TYPE, double mean, double stdev){
    if(value == nullptr){
        value = (double*)malloc(dim*sizeof(double));
        SELFINITVALUE = true;
    }
    switch (TYPE) {
        case INIT_RAND_N:
            for(size_t i = 0; i < dim; ++i)
                // normal distribution with \mu = mean,\sigma^2 = stdev
                value[i] = rnd::randn(mean, stdev);
            break;
        case INIT_RAND_U:
            for(size_t i = 0; i < dim; ++i)
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
