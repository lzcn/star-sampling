/** @file 
 *	@author Zhi Lu
 *  @brief Class for Matrices
 */ 
#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <algorithm>

typedef unsigned int uint; ///< uint : unsigned int

const uint MATRIX_NONE_SUM = 0; ///< no column or row's one norm will be computed
const uint MATRIX_COL_SUM  = 1; ///< compute the one norm of each column
const uint MATRIX_ROW_SUM  = 2; ///< compute the one norm of each row asum
const uint MATRIX_FULL_SUM = 3; ///< compute all one norm

/**
 * @biref compute the sign of x
 *
 */
inline int sgn(const double &x){return (x < 0 ? -1:1);}
inline int sgn_foo(const double &x){return (x < 0 ? -1:1);}
class point2D;
class point3D;
class pointND;
class SubIndex;
/** 
 *  @brief insert p into a ascending sequence
 *				 if p less than the last element
 */
void doInsertReverse(double p, std::list<double> &listTop);
/** 
 *  @brief insert p into a descending sequence
 *				 if p is greater than the last element
 */
void doInsert(double p, std::list<double> &listTop);
/** 
 *	@param[in] listTop : the value sequence
 *	@param[in] p : value p to insert 
 *	@param[in] listIdx : the coordinate sequence corresponding
 * 											 to the value sequence
 *	@param[in] coord : coordiante attribute for p
 *
 *  @brief insert value p into a descending sequence
 *				 if p is greater than the last element
 *         and coord is the attribute for p
 *
 *  if p is not lager than the last element in value sequence, do not insert.
 */
void doInsert(double p, std::list<double> &listTop, point3D &coord, std::list<point3D> &listIdx);

typedef std::pair<point2D,double> pidx2d;
typedef std::pair<point3D,double> pidx3d;
typedef std::pair<pointND,double> pidxNd;

template <typename Tpair> 
bool compgt(const Tpair &v1,const Tpair &v2){ return (v1.second > v2.second); }
template <typename Tpair>
bool complt(const Tpair &v1,const Tpair &v2){ return (v1.second < v2.second); }

class Matrix;
/** 
 *  @brief compute \f$\sum_{r}{a_{x,r}*b_{y,r}}\f$
 */
double MatrixRowMul(const point2D &coord, 
                    Matrix &A, 
                    Matrix &B);
/** 
 *  @brief compute \f$\sum_{r}{a_{x,r}*b_{y,r}*c_{z,r}}\f$
 */
double MatrixRowMul(const point3D &coord, 
                    Matrix &A, 
                    Matrix &B, 
                    Matrix &C);
/** 
*  @brief compute \f$\sum_{r}{a_{r,x}*b_{r,y}}\f$
*/
double MatrixColMul(const point2D &coord, 
										Matrix &A, 
										Matrix &B);
/** 
*  @brief compute \f$\sum_{r}{a_{r,x}*b_{r,y}*c_{r,z}}\f$
*/
double MatrixColMul(const point3D &coord, 
										Matrix &A, 
										Matrix &B, 
										Matrix &C);
/** 
 *  @brief compute \f$\sum_{r}{a_{r,i}*b_{r,j}}\f$
 */
double MatrixColMul(Matrix &A, 
										Matrix &B, 
										uint i, uint j);
/** 
 *  @brief compute \f$\sum_{r}{a_{r,i}*b_{r,j}*c_{r,k}}\f$
 */
double MatrixColMul(Matrix &A, 
										Matrix &B, 
										Matrix &C, 
										uint i, uint j, uint k);
double vectors_mul(const pointND &p,std::vector<Matrix*> &vMat);
double vectors_mul(const point2D &coord, Matrix &A, Matrix &B);
double vectors_mul(const point3D &coord, Matrix &A, Matrix &B, Matrix &C);
/** 
 *  @brief using binary search to get s uint with given distribution
 *  @param[in] s : number of random results
 *  @param[in] *dst : the random numbers
 *  @param[in] *pdf : distribution that sums to 1
 */
void binary_search(size_t s, uint *dst, uint n, double *pdf);
/** 
 *  @brief binary search
 *  @param[in] *a : the search vector
 *  @param[in] ub : the length of a
 *  @param[in] s : search value
 */
uint binary_search_once(double *a, uint ub, double s);
void sort_sample(size_t s, uint*i, uint*r, size_t *freq, uint m, uint n, double*pdf, double sum_pdf);
void sort_sample(size_t s, uint*dst, uint n, double*p, double sum);
void vose_alias(size_t s, uint *dst, uint n, double *pdf,double sum_pdf);
/** 
 * 	@brief point2D for (x,y)
 */
class point2D
{
public:
	point2D(uint i, uint j);
	~point2D(){};
	bool operator<(const point2D &toCmp)const;
	bool operator==(const point2D &toCmp)const;
	bool operator>(const point2D &toCmp)const;
	uint x; 
	uint y;
};
struct hashFunc2d{
    size_t operator()(const point2D &k) const{
    size_t h1 = std::hash<uint>()(k.x);
    size_t h2 = std::hash<uint>()(k.y);
    return (h2 ^ (h1 << 1));
    }
};
struct equalsFunc2d{
  bool operator()( const point2D& lhs, const point2D& rhs ) const{
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
  }
};
typedef std::unordered_map<point2D, double, hashFunc2d, equalsFunc2d> TPoint2DMap;
/** @brief point3D for (x,y,z)
 *
 */
class point3D
{
public:
	point3D(uint i, uint j, uint k);
	~point3D(){};
	bool operator<(const point3D &toCmp)const;
	bool operator==(const point3D &toCmp)const;
	bool operator>(const point3D &toCmp)const;
	uint x;
	uint y;
	uint z;
};
struct hashFunc{
    size_t operator()(const point3D &k) const{
    size_t h1 = std::hash<uint>()(k.x);
    size_t h2 = std::hash<uint>()(k.y);
    size_t h3 = std::hash<uint>()(k.z);
    return (h1 ^ (h2 << 1)) ^ h3;
    }
};
struct equalsFunc{
  bool operator()( const point3D& lhs, const point3D& rhs ) const{
    return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
  }
};
typedef std::unordered_map<point3D, double, hashFunc, equalsFunc> TPoint3DMap;
class pointND
{
public:
	pointND(uint *p, uint n);
	~pointND(){};
	bool operator<(const pointND &toCmp)const;
	uint num;
	uint *coord;
};

class Matrix{
public:
	uint row;
	uint col;
	double* element;
	double* SumofCol;
	double* SumofRow;
	Matrix(uint r, uint c);
	Matrix(uint r, uint c, double*pr);
	Matrix(uint r, uint c, double*pr, uint TYPE);
	~Matrix();
	double GetElement(uint i, uint j);
	double GetColSum(uint column);
	double& operator()(uint i, uint j) const { return element[j*row +i]; }
	void transpose(double *pr);
    /** 
     *  @brief compute the absolute accumulation of each column
     *  
     * \f$ m_{ij} = \sum_{s=0}^{s=j}|p_{is}| \f$
     */
	void accumulation(double *pr);
	uint randRow(uint n);
	/** \biref	return n with probability abs(M_{m,n})/SumofRow(m)
	 *
	 */ 
	uint randCol(uint m);
private:
	uint _SUMTYPE;
	bool _SELFVALUE;
};
/** 
 *  @brief counter for uint array[n]
 *
 *  each position in arry has maximum
 */
class SubIndex{
public:
	SubIndex(uint n, uint *max);
	~SubIndex();
    /** 
     *  @brief if all element meets maximum
     */
	bool isDone(){return doneFlag;};
    /** 
     *  @brief reset to all zeros
     */
	bool reset();
	SubIndex& operator++();
    /** 
     *  @brief get current indexes
     */
	const uint *getIdx(){return curIdx;};
private:
	uint idxSize;
	bool doneFlag;
	uint *curIdx;
	const uint *maxIdx;
};

#endif // __MATRIX_H__
