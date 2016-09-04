#ifndef __DMATRIX_H__
#define __DMATRIX_H__


#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include "random.h"
#include "util.h"
const uint INIT_RAND_U = 1;
const uint INIT_RAND_N = 3;
const uint INIT_ALL_ZEROS = 4;
const uint INIT_ALL_ONES = 5;

class DMatrix;

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
    void DMatrix::save(std::string filename);
    double* get_abs_row_sum();
    double* get_abs_col_sum();
public:
	uint row, col;
	double *value;
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
void DMatrix::save(std::string filename) {
    std::ofstream out_file (filename.c_str());
    if (out_file.is_open())	{
        out_file << row << "\t" << col << std::endl;
        for (uint c = 0; c < col; c++){
            for (uint r = 0; r < row; r++){
                out_file << value[c*row + r] << "\t";
            }
            out_file << std::endl;
        }
        out_file.close();
    } else {
        std::cout << "Unable to open file " << filename;
    }   			
}
#endif
