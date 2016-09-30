#include <cstdio>
#include <cmath>
#include "matrix.h"


double sgn(double x){ return (x < 0 ? -1:1); }
double sgn_foo(double x){ return (x < 0 ? -1:1); }
/*
	class for point2D
*/
point2D::point2D(uint i, uint j){
	x = i;
	y = j;
}

bool point2D::operator<(const point2D &toCmp)const{
	if(x < toCmp.x){
		return true;
	}else if(x == toCmp.x){
		if(y < toCmp.y)
			return true;
	}
	return false;	
}
bool point2D::operator==(const point2D &toCmp)const{
	if(x == toCmp.x && y == toCmp.y)
		return true;
	return false;
}
bool point2D::operator > (const point2D &toCmp)const{
	if(x > toCmp.x){
		return true;
	}else if(x == toCmp.x){
		if(y > toCmp.y)
			return true;
	}
	return false;
}

/*
	class for point3D
*/
point3D::point3D(uint i, uint j, uint k){
	x = i;
	y = j;
	z = k;
}

bool point3D::operator<(const point3D &toCmp)const{
	if(x < toCmp.x){
		return true;
	}
	if(x == toCmp.x){
		if(y < toCmp.y)
			return true;
	}
	if(x == toCmp.x && y == toCmp.y){
		if(z < toCmp.z)
			return true;
	}
	return false;	
}
bool point3D::operator==(const point3D &toCmp)const{
	if(x == toCmp.x && y == toCmp.y && z == toCmp.z)
		return true;
	return false;
}
bool point3D::operator > (const point3D &toCmp)const{
	if(x > toCmp.x){
		return true;
	}
	if(x == toCmp.x){
		if(y > toCmp.y)
			return true;
	}
	if(x == toCmp.x && y == toCmp.y){
		if(z > toCmp.z)
			return true;
	}
	return false;	
}
/*
	class for pointND
*/
pointND::pointND(uint *p, uint n){
	coord = p;
	num = n;
}
bool pointND::operator < (const pointND &toCmp)const{
	for(uint i = 0; i < num; ++i){
		if(coord[i] < toCmp.coord[i]){
			return true;
		}else if(coord[i] > toCmp.coord[i]){
			return false;
		}
	}
	if(coord[num-1] < toCmp.coord[num-1])
		return true;
	else
		return false;
}
/*
	class for matrix
*/
Matrix::Matrix(uint r, uint c){
	row = r;
	col = c;
	element = (double*)malloc(row*col*sizeof(double));
	_SELFVALUE = true;
	_SUMTYPE = MATRIX_NONE_SUM;
}
Matrix::Matrix(uint r, uint c, double*pr){
	row = r;
	col = c;
	element = pr;
	_SELFVALUE = false;
	_SUMTYPE = MATRIX_FULL_SUM;
	SumofCol = (double*)malloc(col*sizeof(double));
	memset(SumofCol, 0, col*sizeof(double));
	SumofRow = (double*)malloc(row*sizeof(double));
	memset(SumofRow, 0, row*sizeof(double));
	//get the absolute sum of each columns
	for(uint i = 0; i < col; ++i){
		for(uint j = 0; j < row; ++j){
			SumofCol[i] += abs(element[i*row + j]);
			SumofRow[j] += abs(element[i*row + j]);
		}	
	}
}
Matrix::Matrix(uint r, uint c, double*pr, uint TYPE){
	row = r;
	col = c;
	element = pr;
	_SELFVALUE = false;
	_SUMTYPE = TYPE;
	switch (TYPE) {
		case MATRIX_FULL_SUM:
			SumofCol = (double*)malloc(col*sizeof(double));
			memset(SumofCol, 0, col*sizeof(double));
			SumofRow = (double*)malloc(row*sizeof(double));
			memset(SumofRow, 0, row*sizeof(double));
			for(uint i = 0; i < col; ++i){
				for(uint j = 0; j < row; ++j){
					SumofCol[i] += abs(element[i*row + j]);
					SumofRow[j] += abs(element[i*row + j]);
				}	
			}
			break;
		case MATRIX_NONE_SUM:
			SumofCol = nullptr;
			SumofRow = nullptr;
			break;
		case MATRIX_COL_SUM:
			SumofCol = (double*)malloc(col*sizeof(double));
			memset(SumofCol, 0, col*sizeof(double));
			SumofRow = nullptr;
			for(uint i = 0; i < col; ++i){
				for(uint j = 0; j < row; ++j){
					SumofCol[i] += abs(element[i*row + j]);
				}
			}
			break;
		case MATRIX_ROW_SUM:
			SumofCol = nullptr;
			SumofRow = (double*)malloc(row*sizeof(double));
			memset(SumofRow, 0, row*sizeof(double));
			for(uint i = 0; i < col; ++i){
				for(uint j = 0; j < row; ++j){
					SumofRow[j] += abs(element[i*row + j]);
				}
			}	
			break;
		default: break;
	}
}
Matrix::~Matrix(){
	switch (_SUMTYPE) {
		case MATRIX_COL_SUM:
			free(SumofCol);break;
		case MATRIX_ROW_SUM:
			free(SumofRow);break;
		case MATRIX_FULL_SUM:
			free(SumofCol);
			free(SumofRow);
			break;
		default:break;
	}
	if(_SELFVALUE){
		free(element);
	}
}

double Matrix::GetElement(uint i, uint j){
	return element[j*row + i];
}

double Matrix::GetColSum(uint column){
	return SumofCol[column];
}
void Matrix::transpose(double *pr){
	for(uint r = 0 ; r < row; ++r){
		for (uint c = 0; c < col; ++c){
			element[c*row + r] = pr[r*col + c];
		}
	}
}
void Matrix::accumulation(double *pr){
	for (uint c = 0; c < col; ++c){
		double sum = 0.0;
		size_t offset = c*row;
		for(uint r = 0 ; r < row; ++r){
			sum += abs(pr[offset +r]);
			element[c*row + r] = sum;
		}
	}
}
uint Matrix::randRow(uint n){
	double x,temp;
	x = SumofCol[n]*((double)rand()/(double)RAND_MAX);
	temp = 0;
	for (uint i = 0; i < row; ++i){
		temp += abs(element[i + n*row]);
		if(x <= temp){ 
			return i;
		}
	}
	return (row-1);
}
uint Matrix::randCol(uint m){
	double x,temp;
	x = SumofRow[m]*((double)rand()/(double)RAND_MAX);
	temp = 0;
	for (uint j = 0; j < col; ++j){
		temp += abs(element[m + j*row]);
		if(x <= temp){ 
			return j;
		}
	}
	return (col-1);
}


double MatrixRowMul(const point2D &coord, Matrix &A, Matrix &B){
	double ans = 0.0;
	uint r1 = coord.x;
	uint r2 = coord.y;
	for(uint c = 0; c < A.col; ++c){
		ans += A(r1,c) * B(r2,c);
	}
	return ans;
}
double MatrixRowMul(const point3D &coord, Matrix &A, Matrix &B, Matrix &C){
	double ans = 0.0;
	uint r1 = coord.x;
	uint r2 = coord.y;
	uint r3 = coord.z;
	for(uint c = 0; c < A.col; ++c){
		ans += A(r1,c) * B(r2,c) * C(r3,c);
	}
	return ans;
}
double MatrixColMul(const point2D &coord, Matrix &A, Matrix &B){
	double ans = 0.0;
	uint c1 = coord.x;
	uint c2 = coord.y;
	for(uint r = 0; r < A.row; ++r){
		ans += A(r,c1) * B(r,c2);
	}
	return ans;
}

double MatrixColMul(const point3D &coord, Matrix &A, Matrix &B, Matrix &C){
	double ans = 0.0;
	uint c1 = coord.x;
	uint c2 = coord.y;
	uint c3 = coord.z;
	for(uint r = 0; r < A.row; ++r){
		ans += A(r,c1) * B(r,c2) * C(r,c3);
	}
	return ans;
}

double MatrixColMul(Matrix &A, Matrix &B, uint c1, uint c2){
	double ans = 0.0;
	for(uint r = 0; r < A.row; ++r){
		ans += A(r,c1) * B(r,c2);
	}
	return ans;
}

double MatrixColMul(Matrix &A, Matrix &B, Matrix &C, uint c1, uint c2, uint c3){
	double ans = 0.0;
	for(uint r = 0; r < A.row; ++r){
		ans += A(r,c1) * B(r,c2) * C(r,c3);
	}
	return ans;
}
double vectors_mul(const point2D &coord, Matrix &A, Matrix &B){
	double ans = 0;
	uint i = coord.x;
	uint j = coord.y;
    for (uint r = 0; r < A.row; ++r){
        ans += A(r,i) * B(j,r);
    }
    return ans;
}
double vectors_mul(const point3D &coord, Matrix &A, Matrix &B, Matrix &C){
	double ans = 0;
	uint i = coord.x;
	uint j = coord.y;
	uint k = coord.z;
    for (uint r = 0; r < A.row; ++r){
        ans += A(r,i) * B(j,r) * C(k,r);
    }
    return ans;
}

double vectors_mul(const pointND &p,std::vector<Matrix*> &vMat){
    uint MatNum = p.num;
    uint rankSize = vMat[0]->row;
    double ans = 0;
    double *temp = (double*)malloc(rankSize*sizeof(double));
    memset(temp, 1, rankSize*sizeof(double));
    for (uint r = 0; r < rankSize; ++r){
        temp[r] = vMat[0]->GetElement(r,p.coord[0]);
    }
    for (uint n = 1; n < MatNum; ++n){
        for(uint r = 0; r < rankSize; ++r){
            temp[r] *= vMat[n]->GetElement(p.coord[n],r);
        }
    }
    for (uint i = 0; i < rankSize; ++i){
        ans += temp[i];
    }
    free(temp);
    return ans;
}

void doInsertReverse(double p, std::list<double> &listTop){
    std::list<double>::iterator itr = listTop.begin();
    if(p < listTop.front()){
        listTop.push_front(p);
        listTop.pop_back();
        return;
    }
    itr++;
    for(;itr != listTop.end(); ++itr){
        if(p < (*itr)){
            listTop.insert(itr,p);
            listTop.pop_back(); 
            return;
        }
    }
}
void doInsert(double p, std::list<double> &listTop){
    std::list<double>::iterator itr = listTop.begin();
    if(p > listTop.front()){
        listTop.push_front(p);
        listTop.pop_back();
        return;
    }
    itr++;
    for(;itr != listTop.end(); ++itr){
        if(p > (*itr)){
            listTop.insert(itr,p);
            listTop.pop_back();
            return;
        }
    }
}
void doInsert(double p, std::list<double> &listTop, point3D &coord, std::list<point3D> &listIdx){
    auto itr = listTop.begin();
    auto itr2 = listIdx.begin();
    if(p > listTop.front()){
        listTop.push_front(p);
        listTop.pop_back();
        listIdx.push_front(point3D(coord.x,coord.y,coord.z));
        listIdx.pop_back();
        return;
    }
    itr++;
    itr2++;
    for(;itr != listTop.end(); ++itr,++itr2){
        if(p > (*itr)){
            listTop.insert(itr,p);
            listTop.pop_back();
            listIdx.insert(itr2,point3D(coord.x,coord.y,coord.z));
            listIdx.pop_back();
            return;
        }
    }	
}
void binary_search(size_t s, uint *dst, uint n, double *pdf){
	double sum = pdf[n-1];
	for (size_t i = 0; i < s; ++i){
		double u = sum*((double)rand()/(double)RAND_MAX);
		*(dst + i) = binary_search_once(pdf,n-1,u);
	}
}
uint binary_search_once(double *a, uint ub, double s){
	uint m;
	uint lb = 0;
	if(s < a[0])
	return 0;
	while (lb < ub-1){
		m = (lb + ub )/2;
		if (s < a[m])
			ub = m;
		else
			lb = m;
	}
	return (ub);
}
void vose_alias(size_t s, uint *dst, uint n, double *pdf,double sum_pdf){
	double *scaled_prob = new double[n];
	double *table_prob = new double[n];
	uint *table_alias = new uint[n];
	uint *table_small = new uint[n];
	uint *table_large = new uint[n];
	uint small_index = 0;
	uint large_index = 0;
	/* stage 1: initialization */
	for (uint i = 0; i < n; ++i){
		scaled_prob[i] = abs(*(pdf+i)) * n;
		if ( scaled_prob[i] < sum_pdf ){
			table_small[small_index] = i;
			++small_index;
		}else{
			table_large[large_index] = i;
			++large_index;
		}
	}
	uint l,g;
	while(small_index != 0 && large_index != 0){
		small_index -= 1;
		large_index -= 1;
		l = table_small[small_index];
		g = table_large[large_index];
		table_prob[l] = scaled_prob[l];
		table_alias[l] = g;
		scaled_prob[g] = (scaled_prob[g] + scaled_prob[l]) - sum_pdf;

		if (scaled_prob[g] < sum_pdf){
			table_small[small_index] = g;
			++small_index;
		}else{
			table_large[large_index] = g;
			++large_index;
		}
	}
	while(large_index != 0){
		large_index -= 1;
		table_prob[table_large[large_index]] = sum_pdf;
	}
	while(small_index != 0){
		small_index -= 1;
		table_prob[table_small[small_index]] = sum_pdf;
	}
	/* stage 2: random sampling */
	double u;
	uint fair_die;
	for (size_t i = 0; i < s; ++i ){
		fair_die = rand() % n;
		u = sum_pdf*(double)rand()/(double)RAND_MAX;
		if (table_prob[fair_die] >= u){
			*(dst + i) = fair_die;
		}else{
			*(dst + i) = table_alias[fair_die];
		}
	}
	delete []table_prob;
	delete []table_alias;
	delete []scaled_prob;
	delete []table_small;
	delete []table_large;
}


void sort_sample(size_t s, uint*dst, uint n, double*p, double sum){
	std::vector<double> rand_u;
	for (size_t i = 0; i < s; ++i){
		rand_u.push_back(sum*((double)rand()/(double)RAND_MAX));
	}
	sort(rand_u.begin(),rand_u.end());
	uint ind = 0;
	double prob_accum = abs(p[0]);
	for (size_t i = 0; i < s; ++i){
		while((rand_u[i] >= prob_accum) && (ind < (n-1))){
			prob_accum += abs(p[++ind]);
		}
		dst[i] = ind;
	}
}
void sort_sample(size_t s, uint*idxI, uint*idxR, size_t *freq,
				  uint m, uint n, \
				  double*pdf, double sum_pdf){
	std::vector<double> rand_u;
	for (size_t i = 0; i < s; ++i){
		rand_u.push_back(sum_pdf*((double)rand()/(double)RAND_MAX));
	}
	// Sort the random values
	// It will be sorted according to k then i;
	sort(rand_u.begin(),rand_u.end());
	uint ind = 0;
	uint range = m * n;
	double sum_prob = abs(pdf[0]);
	for (size_t i = 0; i < s; ++i){
		while((rand_u[i] >= sum_prob) && (ind < (range-1))){
			sum_prob += abs(pdf[++ind]);
		}
		idxI[i] = ind % n;
		idxR[i] = ind / n;
		++freq[idxR[i]];
	}
}
SubIndex::SubIndex(uint n, uint *max){
	idxSize = n;
	maxIdx = max;
	doneFlag = false;
	curIdx = (uint*)malloc((n + 1)*sizeof(uint));
	memset(curIdx, 0, (n + 1)*sizeof(uint));
}
SubIndex::~SubIndex(){
	free(curIdx);
}

bool SubIndex::reset(){
	doneFlag = false;
	memset(curIdx, 0, (idxSize + 1)*sizeof(uint));
	return true;
}

SubIndex& SubIndex::operator++(){
	++curIdx[0];
	for(uint i = 0; i < idxSize; ++i){
		if(curIdx[i] < maxIdx[i]){
			return *this;
		}else{
			curIdx[i] = 0;
			++curIdx[i + 1];
		}
	}
	if(curIdx[idxSize] >= 1){
		doneFlag = true;
	}
	return *this;
}
