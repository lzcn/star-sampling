#include "../matrix.h"


/*
	class for point2D
*/
point2D::point2D(size_t i, size_t j){
	x = i;
	y = j;
}

bool point2D::operator<(const point2D &toCmp){
	if(x < toCmp.x){
		return true;
	}else if(x == toCmp.x){
		if(y < toCmp.y)
			return true;
	}
	return false;	
}
bool point2D::operator==(const point2D &toCmp){
	if(x == toCmp.x && y == toCmp.y)
		return true;
	return false;
}
bool point2D::operator > (const point2D &toCmp){
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
point3D::point3D(size_t i, size_t j, size_t k){
	x = i;
	y = j;
	z = k;
}

bool point3D::operator<(const point3D &toCmp){
	if(x < toCmp.x){
		return true;
	}else if(x == toCmp.x){
		if(y < toCmp.y)
			return true;
	}else if(y == toCmp.y){
		if(z < toCmp.z)
			return true;
	}
	return false;	
}
bool point3D::operator==(const point3D &toCmp){
	if(x == toCmp.x && y == toCmp.y && z == toCmp.z)
		return true;
	return false;
}
bool point3D::operator > (const point3D &toCmp){
	if(x > toCmp.x){
		return true;
	}else if(x == toCmp.x){
		if(y > toCmp.y)
			return true;
	}else if(y == toCmp.y){
		if(z > toCmp.z)
			return true;
	}
	return false;
}

/*
	class for matrix
*/
Matrix::Matrix(size_t r, size_t c, double*pr){
	row = r;
	col = c;
	element = pr;
	SumofCol = (double*)malloc(col*sizeof(double));
	memset(SumofCol, 0, col*sizeof(double));
	//get the absolute sum of each columns
	double temp = 0.0;
	for(size_t i = 0; i < col; ++i){
		temp = 0.0;
		for(size_t j = 0; j < row; ++j){
			temp += abs(element[i*row + j]);
		}
		SumofCol[i] = temp;
		}
	}
Matrix::~Matrix(){
	free(SumofCol);
}

double Matrix::GetEmelent(size_t i, size_t j){
	return element[j*row + i];
}

double Martrix::GetColSum(size_t column){
	return SumofCol[column];
}

size_t Martrix::randRow(size_t n){
	double x,temp;
	x = SumofCol[n]*((double)rand()/(double)RAND_MAX);
	temp = 0;
	for (size_t i = 0; i < row; ++i){
		temp += abs(element[i + n*row]);
		if(x <= temp){ 
			return i;
		}
	}
	return (row-1);
}


double MatrixColMul(const Matrix &A, const Matrix &B, \
					size_t m, size_t n){
	if(A.row != B.row){
		printf("<TrackBack:MatrixColMul>row size doesn't match\n");
		return 0;
	}
	size_t row = A.row;
	double temp = 0.0;
	for(size_t i = 0; i < row; ++i){
		temp += A[m * row + i]*B[n * row + i];
	}
	return temp;
}

double MatrixColMul(const Matrix &A, \
					const Matrix &B, \
					const Matrix &C, \
					size_t m, size_t n, size_t p){
		if(A.row != B.row || B.row != C.row){
		printf("<TrackBack:MatrixColMul>row size doesn't match\n");
		return 0;
	}
	size_t row = A.row;
	double temp = 0.0;
	for(size_t i = 0; i < row; ++i){
		temp += A[m * row + i]*B[n * row + i]*C[p * row + i];
	}
	return temp;
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