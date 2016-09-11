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
    uint x;
    uint y;
    
	point2D(uint i, uint j);
	~point2D(){};
	bool operator<(const point2D &toCmp)const;
	bool operator==(const point2D &toCmp)const;
	bool operator>(const point2D &toCmp)const;
};

class point3D{
public:
    uint x;
    uint y;
    uint z;
        
	point3D(uint i, uint j, uint k);
	~point3D(){};
	bool operator<(const point3D &toCmp)const;
	bool operator==(const point3D &toCmp)const;
	bool operator>(const point3D &toCmp)const;

};

class pointND
{
public:
	pointND(uint *p, uint n);
	~pointND(){};
	bool operator<(const pointND &toCmp)const;
	uint num;
	uint *coord;
};

point2D::point2D(uint i, uint j){
	x = i;
	y = j;
}

class SubIndex{
public:
	SubIndex(uint n, uint *max);
	~SubIndex();
	bool isDone(){return doneFlag;};
	bool reset();
	SubIndex& operator++();
	const uint *getIdx(){return curIdx;};
private:
	uint idxSize;
	bool doneFlag;
	uint *curIdx;
	const uint *maxIdx;
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

////////////////
//  pointND 
///////////////

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
	if(coord[num-1] == toCmp.coord[num-1])
		return false;
	else 
		return true;
}
////////////////
// SubIndex
////////////////
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
	curIdx[0]++;
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

#endif/*__COORDINATE_H__*/
