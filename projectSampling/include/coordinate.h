#ifndef __COORDINATE_H__
#define __COORDINATE_H__

#include "util.h"

class point2D;
class point3D;
class pointND;
class SubIndex;

typedef std::pair<point2D,double> pidx2d;
typedef std::pair<point3D,double> pidx3d;
typedef std::pair<pointND,double> pidxNd;

template <typename Tpair>
int compgt(const Tpair &v1,const Tpair &v2){
	return (v1.second > v2.second);
}

template <typename Tpair>
int complt(const Tpair &v1,const Tpair &v2){
	return (v1.second < v2.second);
}

class point2D{
public:
    size_t x;
    size_t y;
    
	point2D(size_t i, size_t j);
	~point2D(){};
	bool operator<(const point2D &toCmp)const;
	bool operator==(const point2D &toCmp)const;
	bool operator>(const point2D &toCmp)const;
};

class point3D{
public:
    size_t x;
    size_t y;
    size_t z;
        
	point3D(size_t i, size_t j, size_t k);
	~point3D(){};
	bool operator<(const point3D &toCmp)const;
	bool operator==(const point3D &toCmp)const;
	bool operator>(const point3D &toCmp)const;

};

class pointND
{
public:
	pointND(size_t *p, size_t n);
	~pointND(){};
	bool operator<(const pointND &toCmp)const;
	size_t num;
	size_t *coord;
};

point2D::point2D(size_t i, size_t j){
	x = i;
	y = j;
}

class SubIndex{
public:
	SubIndex(int n, size_t *max);
	~SubIndex();
	bool isDone(){return doneFlag;};
	bool reset();
	SubIndex& operator+(const size_t step);
	SubIndex& operator++();
	const size_t *getIdx(){return curIdx;};
private:
	int idxSize;
	bool doneFlag;
	size_t *curIdx;
	const size_t *maxIdx;
};

////////////////
//  point2D 
///////////////

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

////////////////
//  point3D 
///////////////

point3D::point3D(size_t i, size_t j, size_t k){
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

////////////////
//  pointND 
///////////////

pointND::pointND(size_t *p, size_t n){
	coord = p;
	num = n;
}
bool pointND::operator < (const pointND &toCmp)const{
	for(size_t i = 0; i < num; ++i){
		if(coord[i] < toCmp.coord[i]){
			return true;
		}else if(coord[i] > toCmp.coord[i]){
			return false;
		}
	}
	if(coord[num-1] == toCmp.coord[num-1])
		return false;
}
////////////////
// SubIndex
////////////////
SubIndex::SubIndex(int n, size_t *max){
	idxSize = n;
	maxIdx = max;
	doneFlag = false;
	curIdx = (size_t*)malloc((n + 1)*sizeof(size_t));
	memset(curIdx, 0, (n + 1)*sizeof(size_t));
}
SubIndex::~SubIndex(){
	free(curIdx);
}

bool SubIndex::reset(){
	doneFlag = false;
	memset(curIdx, 0, (idxSize + 1)*sizeof(size_t));
	return true;
}
SubIndex& SubIndex::operator+(const size_t step){
	size_t a, b;
	a = step;
	b= 0;
	size_t *temp = (size_t *)malloc(idxSize*sizeof(size_t));
	memset(temp, 0, idxSize*sizeof(size_t));
	for(size_t i = 0; i< idxSize;++i){
		b = a % maxIdx[i];
		a = a / maxIdx[i];
		temp[i] = b;
		curIdx[i] += b;
		while(curIdx[i] >= maxIdx[i]){
			curIdx[i] -= maxIdx[i];
			++curIdx[i + 1];
		}
		if(a > 0){
			curIdx[idxSize] += a;
		}
		if(curIdx[idxSize] > 0){
			doneFlag = true;
		}
		free(temp);
		return *this;
	}
}
SubIndex& SubIndex::operator++(){
	curIdx[0]++;
	for(size_t i = 0; i < idxSize; ++i){
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

#endif/*__COORDINATE_H__*/
