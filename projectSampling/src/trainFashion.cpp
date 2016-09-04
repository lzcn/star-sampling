#include <map>
#include <set>
#include <cstdio>
#include <ctime>
#include "dmatrix.h"
#include "util.h"
#include "coordinate.h"
#include "random.h"
#include "mex.h"
double diff_y_pred(uint u,point3D &p,point3D &n,DMatrix &U,DMatrix &T,DMatrix &B,DMatrix &S){
	uint rank = U.row;
	double ypos = 0.0;
	double yneg = 0.0;
	for(uint r = 0; r < rank; ++r){
		ypos += U(r,u)*T(r,p.x)*B(r,p.y)*S(r,p.z);
		yneg += U(r,u)*T(r,n.x)*B(r,n.y)*S(r,n.z);
	}
	return (ypos-yneg);
}
double Frobenius(DMatrix &A){
	double ans = 0.0;
	for(uint i = 0; i < A.col*A.row; i++){
		ans += A.value[i]*A.value[i];
	}
	return (ans);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    //--------------------------------------------
    // Remove some tuple untill to be 5-core data
    //--------------------------------------------
	clock_t start;
	uint NumCate = (uint)mxGetM(prhs[0]);
	uint NumofPosEntry = (uint)mxGetN(prhs[0]);
	uint NumofNegEntry = (uint)mxGetN(prhs[1]);
	size_t *PtrPos = (size_t*)malloc(NumCate*NumofPosEntry*sizeof(size_t));
	size_t *PtrNeg = (size_t*)malloc(NumCate*NumofNegEntry*sizeof(size_t));
	double *pos = mxGetPr(prhs[0]);
	double *neg = mxGetPr(prhs[1]);
	for(uint r = 0; r < NumCate; ++r){
		for(uint c1 = 0; c1 < NumofPosEntry; ++c1){
			PtrPos[c1*NumCate + r] = (size_t)pos[c1*NumCate + r];
		}
		for(uint c2 = 0; c2 < NumofNegEntry; ++c2){
			PtrNeg[c2*NumCate + r] = (size_t)neg[c2*NumCate + r];
		}
	}
	DMatrix MatPos(NumCate, NumofPosEntry,pos);
	DMatrix MatNeg(NumCate, NumofNegEntry,neg);
    // record the data to a map, and count user,item,tag
	std::set<pointND> PosTuples;
	std::set<pointND> NegTuples;
	std::map<uint,uint> UserMap;
	std::map<uint,uint> TopMap;
	std::map<uint,uint> BottomMap;
	std::map<uint,uint> ShoeMap;
	for(uint i = 0; i < NumofPosEntry; ++i){
        PosTuples.insert(pointND((PtrPos + NumCate*i), NumCate));
        UserMap[(uint)MatPos(0,i)] = 1;
        TopMap[(uint)MatPos(1,i)] = 1;
        BottomMap[(uint)MatPos(2,i)] = 1;
		ShoeMap[(uint)MatPos(3,i)] = 1;
	}
	for(uint i = 0; i < NumofNegEntry; ++i){
		NegTuples.insert(pointND((PtrNeg + NumCate*i), NumCate));
		//UserMap[(uint)MatNeg(0,i)] = 1;
		//TopMap[(uint)MatNeg(1,i)] = 1;
		//BottomMap[(uint)MatNeg(2,i)] = 1;
		//ShoeMap[(uint)MatNeg(3,i)] = 1;
	}
	// do delete
	
    // comprese the range of user, item, tag from 0 to the size;
	// compression map
	// record the compression map
	uint offset = 0;
	for(auto itr = UserMap.begin(); itr != UserMap.end(); ++itr){
		itr->second = offset++;
	}
	offset = 0;
	for(auto itr = TopMap.begin(); itr != TopMap.end(); ++itr){
		itr->second = offset++;
	}
	offset = 0;
	for(auto itr = BottomMap.begin(); itr != BottomMap.end(); ++itr){
		itr->second = offset++;
	}
	offset = 0;
	for(auto itr = ShoeMap.begin(); itr != ShoeMap.end(); ++itr){
		itr->second = offset++;
	}
    // map to save user-outfit
    std::map<uint, std::vector<point3D> > UNeg;
    std::map<uint, std::vector<point3D> > UPos;
	uint countpos = 0;
	uint countneg = 0;
	for(auto itr = NegTuples.begin(); itr != NegTuples.end(); ++itr){
		if(TopMap.end() == TopMap.find(itr->coord[1])||BottomMap.end() == BottomMap.find(itr->coord[2])||ShoeMap.end() == ShoeMap.find(itr->coord[3])){
			continue;
		}
		++countneg;
		uint u = UserMap[itr->coord[0]];
		uint t  = TopMap[itr->coord[1]];
		uint b  = BottomMap[itr->coord[2]];
		uint s = ShoeMap[itr->coord[3]];
		UNeg[u].push_back(point3D(t,b,s));
	}
	for(auto itr = PosTuples.begin(); itr != PosTuples.end(); ++itr){
		++countpos;
		uint u = UserMap[itr->coord[0]];
		uint t  = TopMap[itr->coord[1]];
		uint b  = BottomMap[itr->coord[2]];
		uint s = ShoeMap[itr->coord[3]];
        UPos[u].push_back(point3D(t,b,s));
		if(UNeg.end() == UNeg.find(u)){
			mexPrintf("User:%d has no negative outfits",u);
			return;
		}
	}
	mexPrintf("Current length of tuple:pos-%d,neg-%d\n",countpos,countneg);mexEvalString("drawnow");
	
    //////////////////////////////
    // SGD for factorization
    //////////////////////////////
    // initialized the factor matrices
    uint userSize = UserMap.size();
    uint topSize = TopMap.size();
    uint bottomSize = BottomMap.size();
    uint shoeSize = ShoeMap.size();
    uint factorSize = 64;
    double mean = 0.0;
    double stdev = 0.1;
    double lambda = 0;
    double alpha = 0.01;
    uint maxiter = (uint)mxGetPr(prhs[2])[0];
	mexPrintf("SGD for factorization ......\n");mexEvalString("drawnow");
	// initialize U,I,T
    DMatrix MatUser(factorSize, userSize);
    MatUser.init(INIT_RAND_N,mean,stdev);
    DMatrix MatTop(factorSize, topSize);
    MatTop.init(INIT_RAND_N, mean, stdev);
	DMatrix MatBottom(factorSize, bottomSize);
	MatBottom.init(INIT_RAND_N, mean, stdev);
    DMatrix MatShoe(factorSize, shoeSize);
    MatShoe.init(INIT_RAND_N, mean, stdev);
	uint flag = 0;
    for(uint l = 0; l < maxiter;++l){
		start = clock();
		double BRPOPT = 0.0;
		//BRPOPT -= Frobenius(MatUser);
		//BRPOPT -= Frobenius(MatTop);
		//BRPOPT -= Frobenius(MatBottom);
		//BRPOPT -= Frobenius(MatShoe);
		//BRPOPT *= lambda;
		for(auto upos = UPos.begin(); upos != UPos.end(); ++upos){
			uint u = upos->first;
			std::vector<point3D> OutfitPosVec = upos->second;
			std::vector<point3D> OutfitNegVec = UNeg.find(u)->second;
			for(auto pos = 0; pos < OutfitPosVec.size(); ++pos){
				point3D OutfitPos = OutfitPosVec[pos];
				for(auto neg = 0; neg < OutfitNegVec.size(); ++neg){
					point3D OutfitNeg = OutfitNegVec[neg];
					double y = diff_y_pred(u,OutfitPos,OutfitNeg,MatUser,MatTop,MatBottom,MatShoe);
					BRPOPT += y;
					double sigma = 1 - sigmoid(y);
					uint tpos = OutfitPos.x;
					uint bpos = OutfitPos.y;
					uint spos = OutfitPos.z;
					uint tneg = OutfitNeg.x;
					uint bneg = OutfitNeg.y;
					uint sneg = OutfitNeg.z;
					for(uint f = 0; f < factorSize; ++f){
						double ut = MatUser(f,u)*MatTop(f,tpos);
						double bs = MatBottom(f,bpos)*MatShoe(f,spos);
						double utn = MatUser(f,u)*MatTop(f,tneg);
						double bsn = MatBottom(f,bneg)*MatShoe(f,sneg);
						MatUser(f,u) += alpha*(sigma*(MatTop(f,tpos)*bs - MatTop(f,tneg)*bsn));
						MatTop(f,tpos) += alpha*(sigma*bs*MatUser(f,u));
						MatTop(f,tneg) += alpha*(-sigma*bsn*MatUser(f,u));
						MatBottom(f,bpos) += alpha*(sigma*ut*MatShoe(f,spos));
						MatBottom(f,bneg) += alpha*(-sigma*utn*MatShoe(f,sneg));
						MatShoe(f,spos) += alpha*(sigma*ut*MatBottom(f,bpos));
						MatShoe(f,sneg) += alpha*(-sigma*utn*MatBottom(f,bneg));
					}
			
				}
			}
		}
		++flag;
		if(flag > 100){
			flag = 0;
			MatUser.save("User.txt");
			MatTop.save("Top.txt");
			MatBottom.save("Bottom.txt");
			MatShoe.save("Shoe.txt");
			//BRPOPT = sigmoid(BRPOPT);
			mexPrintf("Iter:%d,ObjectFuntcionValue:%f, TimeDuration:%f\n",l,BRPOPT,timeDuration(start));mexEvalString("drawnow");
		}
	}
	MatUser.save("User.txt");
	MatTop.save("Top.txt");
	MatBottom.save("Bottom.txt");
	MatShoe.save("Shoe.txt");
}
