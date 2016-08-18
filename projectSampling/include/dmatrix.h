#ifndef __DMATRIX_H__
#define __DMATRIX_H__


#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include "random.h"
const uint INIT_RAND_U = 1;
const uint INIT_RAND_N = 3;
const uint INIT_ALL_ZEROS = 4;
const uint INIT_ALL_ONES = 5;
class DMatrix{
public:    
    DMatrix();
    DMatrix(size_t r_dim, size_t c_dim);
    DMatrix(size_t r_dim, size_t c_dim, double *ptr);
	~DMatrix();
    double& operator()(size_t i, size_t j);
	double operator()(size_t i, size_t j);
	size_t randRow(size_t n);
	size_t randCol(size_t m);
    DMatirx& init(uint TYPE);
    DMatirx& get_abs_row_sum(double *dst);
    DMatirx& get_abs_col_sum(double *dst);
public:
	size_t row, col;
	double *value;
private:
    bool SELFINITVALUE;
    
};

class DVector{
public:
    DVector();
    DVector(size_t dim);
    DVector(size_t dim, double*ptr);
	~DVector();
    double& operator()(size_t n);
	double operator()(size_t n);
	size_t randIdx(size_t n);
public:
	size_t dim;
	double* value;
private:
    bool SELFINITVALUE;
    
};

struct SEntry {
    uint id;
    double value;
};
	
struct SVector {
	SEntry *data;
	uint size;
};

DMatrix::~DMatrix(){
    if(SELFINITVALUE)
        free(value);
}

DMatrix::DMatrix(){
    row = 0;
    col = 0;
    value = nullptr;
}
DMatrix::DMatrix(size_t r_dim, size_t c_dim){
    row = r_dim;
    col = c_dim;
    value = nullptr;
}
DMatrix(size_t r_dim, size_t c_dim, double *ptr){
    row = r_dim;
    col = c_dim;
    value = ptr;
    SELFINITVALUE = false;
}
DMatirx& DMatirx::init(uint TYPE){
    if(!value){
        printf("Matrix already initialized\n");
        return *this;
    }
    switch (TYPE) {
        case INIT_RAND_N:
            for(size_t i; i < col*row; ++i)
                value[i] = rnd::randn();
            break;
        case INIT_RAND_U:
            for(size_t i; i < col*row; ++i)
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
    SELFINITVALUE = true;
    return *this;
}
double& operator()(size_t i, size_t j){ 
    //assert((i >= row) && (j >= col));
    return value[j*row +i];
}
double operator()(size_t i, size_t j) const {
    return value[j*row +i];
}
#endif
