#include <cstdio>
#include <ctime>
#include <map>
#include <set>

#include "matrix.h"
#include "mex.h"
#include "random.h"
#include "utils.h"

double diff_y_pred(uint u, Point3d &p, Point3d &n, DenseMat &U, DenseMat &T,
                   DenseMat &B, DenseMat &S) {
  uint rank = U.n_row;
  double ypos = 0.0;
  double yneg = 0.0;
  for (uint r = 0; r < rank; ++r) {
    ypos += U(r, u) * T(r, p.x) * B(r, p.y) * S(r, p.z);
    yneg += U(r, u) * T(r, n.x) * B(r, n.y) * S(r, n.z);
  }
  return (ypos - yneg);
}
double Frobenius(DenseMat &A) {
  double ans = 0.0;
  for (uint i = 0; i < A.n_col * A.n_row; i++) {
    ans += A.value[i] * A.value[i];
  }
  return (ans);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  //--------------------------------------------
  // Remove some tuple untill to be 5-core data
  //--------------------------------------------
  clock_t start;
  // number of catrgries
  uint NumCate = (uint)mxGetM(prhs[0]);
  // number of postive outfits
  uint NumofPosEntry = (uint)mxGetN(prhs[0]);
  // number of negative outfits
  uint NumofNegEntry = (uint)mxGetN(prhs[1]);
  uint *PtrPos = (uint *)malloc(NumCate * NumofPosEntry * sizeof(uint));
  uint *PtrNeg = (uint *)malloc(NumCate * NumofNegEntry * sizeof(uint));
  double *train_pos = mxGetPr(prhs[0]);
  double *trian_neg = mxGetPr(prhs[1]);
  for (uint i = 0; i < NumCate * NumofPosEntry; ++i) {
    PtrPos[i] = (uint)train_pos[i];
  }
  for (uint i = 0; i < NumCate * NumofNegEntry; ++i) {
    PtrNeg[i] = (uint)trian_neg[i];
  }
  // record the data to a map, and count user,item,tag
  std::set<PointNd> PosTuples;
  std::set<PointNd> NegTuples;
  // postive outfit
  for (uint i = 0; i < NumofPosEntry; ++i) {
    PosTuples.insert(PointNd((PtrPos + NumCate * i), NumCate));
  }
  // postive outfit
  for (uint i = 0; i < NumofNegEntry; ++i) {
    NegTuples.insert(PointNd((PtrNeg + NumCate * i), NumCate));
  }
  ///////////////////////
  // Delete the Elements
  // in NegTuples while
  // all has occured in
  // PosTuples
  //////////////////////
  for (; PosTuples.size() != 0;) {
    std::set<uint> User;
    std::set<uint> Top;
    std::set<uint> Bottom;
    std::set<uint> Shoe;
    bool flag = false;
    // all possible items
    for (auto itr = PosTuples.begin(); itr != PosTuples.end(); ++itr) {
      Top.insert(itr->index[1]);
      Bottom.insert(itr->index[2]);
      Shoe.insert(itr->index[3]);
    }
    // if one item of outfit in NegTuples has not ouccured in items, delete
    for (; NegTuples.size() != 0;) {
      mexPrintf("Current NegTuples size %d\n", NegTuples.size());
      mexEvalString("drawnow");
      std::set<PointNd> temp;
      temp = NegTuples;
      bool changed = false;
      for (auto itr = temp.begin(); itr != temp.end(); ++itr) {
        // if the NegTuples's element is not in PosTuples skip this tuple
        if (Top.end() == Top.find(itr->index[1]) ||
            Bottom.end() == Bottom.find(itr->index[2]) ||
            Shoe.end() == Shoe.find(itr->index[3])) {
          NegTuples.erase(*itr);
          changed = true;
          flag = true;
        }
      }
      if (changed)
        continue;
      else
        break;
    }
    mexPrintf("NegTuples size %d\n", NegTuples.size());
    mexEvalString("drawnow");
    // the user with negative outfits
    for (auto itr = NegTuples.begin(); itr != NegTuples.end(); ++itr) {
      User.insert(itr->index[0]);
    }
    // if one user has not negative outfits delete
    for (; PosTuples.size() != 0;) {
      mexPrintf("Current PosTuples size %d\n", PosTuples.size());
      mexEvalString("drawnow");
      std::set<PointNd> temp;
      temp = PosTuples;
      bool changed = false;
      for (auto itr = temp.begin(); itr != temp.end(); ++itr) {
        if (User.end() == User.find(itr->index[0])) {
          PosTuples.erase(*itr);
          changed = true;
          flag = true;
        }
      }
      if (changed)
        continue;
      else
        break;
    }
    mexPrintf("PosTuples size %d\n", PosTuples.size());
    mexEvalString("drawnow");
    if (flag)
      continue;
    else
      break;
  }
  mexPrintf("Final NegTuples size %d\n", NegTuples.size());
  mexEvalString("drawnow");
  mexPrintf("Final PosTuples size %d\n", PosTuples.size());
  mexEvalString("drawnow");

  //////////////////////
  // comprese range
  //////////////////////
  std::map<uint, uint> UserMap;
  std::map<uint, uint> TopMap;
  std::map<uint, uint> BottomMap;
  std::map<uint, uint> ShoeMap;
  for (auto itr = PosTuples.begin(); itr != PosTuples.end(); ++itr) {
    UserMap[itr->index[0]] = 1;
    TopMap[itr->index[1]] = 1;
    BottomMap[itr->index[2]] = 1;
    ShoeMap[itr->index[3]] = 1;
  }
  uint offset = 0;
  for (auto itr = UserMap.begin(); itr != UserMap.end(); ++itr) {
    itr->second = offset++;
  }
  offset = 0;
  for (auto itr = TopMap.begin(); itr != TopMap.end(); ++itr) {
    itr->second = offset++;
  }
  offset = 0;
  for (auto itr = BottomMap.begin(); itr != BottomMap.end(); ++itr) {
    itr->second = offset++;
  }
  offset = 0;
  for (auto itr = ShoeMap.begin(); itr != ShoeMap.end(); ++itr) {
    itr->second = offset++;
  }
  // map to save user-outfit
  std::map<uint, std::vector<Point3d> > UNeg;
  std::map<uint, std::vector<Point3d> > UPos;
  for (auto itr = PosTuples.begin(); itr != PosTuples.end(); ++itr) {
    uint u = UserMap[itr->index[0]];
    uint t = TopMap[itr->index[1]];
    uint b = BottomMap[itr->index[2]];
    uint s = ShoeMap[itr->index[3]];
    UPos[u].push_back(Point3d(t, b, s));
  }
  for (auto itr = NegTuples.begin(); itr != NegTuples.end(); ++itr) {
    if (UserMap.end() == UserMap.find(itr->index[0])) {
      continue;
    }
    uint u = UserMap[itr->index[0]];
    uint t = TopMap[itr->index[1]];
    uint b = BottomMap[itr->index[2]];
    uint s = ShoeMap[itr->index[3]];
    UNeg[u].push_back(Point3d(t, b, s));
  }
  //////////////////////////////
  // SGD for factorization
  //////////////////////////////
  uint userSize = UserMap.size();
  uint topSize = TopMap.size();
  uint bottomSize = BottomMap.size();
  uint shoeSize = ShoeMap.size();
  mexPrintf("User Size:%d, Top Size:%d, Bootom Size:%d,Shoe Size:%d\n",
            userSize, topSize, bottomSize, shoeSize);
  mexEvalString("drawnow");
  uint factorSize = 64;
  double mean = 0.0;
  double stdev = 0.1;
  double lambda = 0;
  double alpha = 0.01;
  uint maxiter = (uint)mxGetPr(prhs[2])[0];
  mexPrintf("SGD for factorization ......\n");
  mexEvalString("drawnow");
  ////////////////////////
  // initialize U,I,T
  ////////////////////////
  DenseMat MatUser(factorSize, userSize);
  DenseMat MatTop(factorSize, topSize);
  DenseMat MatBottom(factorSize, bottomSize);
  DenseMat MatShoe(factorSize, shoeSize);
  char *type;
  type = mxArrayToString(prhs[3]);
  if (!strcmp(type, "Load")) {
    MatUser.load("User.txt");
    MatTop.load("Top.txt");
    MatBottom.load("Bottom.txt");
    MatShoe.load("Shoe.txt");
  } else {
    MatUser.init(MATRIX_INIT_RANDN, mean, stdev);
    MatTop.init(MATRIX_INIT_RANDN, mean, stdev);
    MatBottom.init(MATRIX_INIT_RANDN, mean, stdev);
    MatShoe.init(MATRIX_INIT_RANDN, mean, stdev);
  }
  ///////////////////////////////
  // Start Iteration
  ///////////////////////////////
  double *gradu = (double *)malloc(factorSize * sizeof(double));
  double *gradb = (double *)malloc(factorSize * sizeof(double));
  double *grads = (double *)malloc(factorSize * sizeof(double));
  double *gradt = (double *)malloc(factorSize * sizeof(double));
  for (uint l = 0, saveflag = 0; l < maxiter; ++l, ++saveflag) {
    start = clock();
    double ObjFun = 0.0;
    ObjFun -= Frobenius(MatUser);
    ObjFun -= Frobenius(MatTop);
    ObjFun -= Frobenius(MatBottom);
    ObjFun -= Frobenius(MatShoe);
    ObjFun *= lambda;
    for (auto upos = UPos.begin(); upos != UPos.end(); ++upos) {
      // for all u
      double auc = 0.0;
      uint u = upos->first;
      std::vector<Point3d> OutfitPosVec = upos->second;
      std::vector<Point3d> OutfitNegVec = UNeg.find(u)->second;
      uint poSize = OutfitPosVec.size();
      uint noSize = OutfitNegVec.size();
      double z = poSize * noSize;
      double *wtn = (double *)malloc(poSize * noSize * sizeof(double));
      memset(wtn, 0, poSize * noSize * sizeof(double));
      memset(gradu, 0, factorSize * sizeof(double));
      // compute wtn(o+,o-)  and  gradu
      for (uint pos = 0; pos < poSize; ++pos) {
        uint tpos = OutfitPosVec[pos].x;
        uint bpos = OutfitPosVec[pos].y;
        uint spos = OutfitPosVec[pos].z;
        Point3d OutfitPos(tpos, bpos, spos);
        for (uint neg = 0; neg < noSize; ++neg) {
          uint tneg = OutfitNegVec[neg].x;
          uint bneg = OutfitNegVec[neg].y;
          uint sneg = OutfitNegVec[neg].z;
          Point3d OutfitNeg(tneg, bneg, sneg);
          double y = diff_y_pred(u, OutfitPos, OutfitNeg, MatUser, MatTop,
                                 MatBottom, MatShoe);
          double wtn_temp = sigmoid(y) * (1 - sigmoid(y));
          wtn[pos * noSize + neg] = wtn_temp;
          auc += sigmoid(y);
          for (uint f = 0; f < factorSize; ++f) {
            gradu[f] +=
                wtn_temp *
                (MatTop(f, tpos) * MatBottom(f, bpos) * MatShoe(f, spos) -
                 MatTop(f, tneg) * MatBottom(f, bneg) * MatShoe(f, sneg));
          }
        }
      }
      //---------------------------------
      // update user matrices
      //---------------------------------
      for (uint f = 0; f < factorSize; ++f) {
        MatUser(f, u) += alpha * (gradu[f] / z - lambda * MatUser(f, u));
      }
      ObjFun += auc / z;
      //---------------------------------
      // update postive top,bottom,shoe
      //---------------------------------
      memset(gradb, 0, factorSize * sizeof(double));
      memset(gradt, 0, factorSize * sizeof(double));
      memset(grads, 0, factorSize * sizeof(double));
      for (uint pos = 0; pos < poSize; ++pos) {
        uint tpos = OutfitPosVec[pos].x;
        uint bpos = OutfitPosVec[pos].y;
        uint spos = OutfitPosVec[pos].z;
        for (uint neg = 0; neg < noSize; ++neg) {
          double tempwtn = wtn[pos * noSize + neg];
          for (uint f = 0; f < factorSize; ++f) {
            gradt[f] += tempwtn * MatBottom(f, bpos) * MatShoe(f, spos);
            gradb[f] += tempwtn * MatTop(f, tpos) * MatShoe(f, spos);
            grads[f] += tempwtn * MatTop(f, tpos) * MatBottom(f, bpos);
          }
        }
        for (uint f = 0; f < factorSize; ++f) {
          MatTop(f, pos) +=
              alpha * (gradt[f] * MatUser(f, u) / z - lambda * MatTop(f, pos));
          MatBottom(f, pos) += alpha * (gradb[f] * MatUser(f, u) / z -
                                        lambda * MatBottom(f, pos));
          MatShoe(f, pos) +=
              alpha * (grads[f] * MatUser(f, u) / z - lambda * MatShoe(f, pos));
        }
      }
      //---------------------------------
      // update negative top,bottom,shoe
      //---------------------------------
      memset(gradb, 0, factorSize * sizeof(double));
      memset(gradt, 0, factorSize * sizeof(double));
      memset(grads, 0, factorSize * sizeof(double));
      for (uint neg = 0; neg < noSize; ++neg) {
        uint tneg = OutfitNegVec[neg].x;
        uint bneg = OutfitNegVec[neg].y;
        uint sneg = OutfitNegVec[neg].z;
        for (uint pos = 0; pos < poSize; ++pos) {
          double tempwtn = wtn[pos * noSize + neg];
          for (uint f = 0; f < factorSize; ++f) {
            gradt[f] += tempwtn * MatBottom(f, bneg) * MatShoe(f, sneg);
            gradb[f] += tempwtn * MatTop(f, tneg) * MatShoe(f, sneg);
            grads[f] += tempwtn * MatTop(f, tneg) * MatBottom(f, bneg);
          }
        }
        for (uint f = 0; f < factorSize; ++f) {
          MatTop(f, neg) +=
              alpha * (-gradt[f] * MatUser(f, u) / z - lambda * MatTop(f, neg));
          MatBottom(f, neg) += alpha * (-gradb[f] * MatUser(f, u) / z -
                                        lambda * MatBottom(f, neg));
          MatShoe(f, neg) += alpha * (-grads[f] * MatUser(f, u) / z -
                                      lambda * MatShoe(f, neg));
        }
      }
      free(wtn);
    }
    // end of one iteration
    if (saveflag > 100) {
      saveflag = 0;
      MatUser.save("User.txt");
      MatTop.save("Top.txt");
      MatBottom.save("Bottom.txt");
      MatShoe.save("Shoe.txt");
      mexPrintf("Iter:%d,ObjectFuntcionValue:%f, TimeDuration:%f\n", l, ObjFun,
                timeDuration(start));
      mexEvalString("drawnow");
    }
  }  // end of all iterations
  MatUser.save("User.txt");
  MatTop.save("Top.txt");
  MatBottom.save("Bottom.txt");
  MatShoe.save("Shoe.txt");
  plhs[0] = mxCreateDoubleMatrix(factorSize, userSize, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(factorSize, topSize, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(factorSize, bottomSize, mxREAL);
  plhs[3] = mxCreateDoubleMatrix(factorSize, shoeSize, mxREAL);
  for (uint i = 0; i < userSize * factorSize; ++i) {
    mxGetPr(plhs[0])[i] = MatUser.value[i];
  }
  for (uint i = 0; i < topSize * factorSize; ++i) {
    mxGetPr(plhs[1])[i] = MatTop.value[i];
  }
  for (uint i = 0; i < bottomSize * factorSize; ++i) {
    mxGetPr(plhs[2])[i] = MatBottom.value[i];
  }
  for (uint i = 0; i < shoeSize * factorSize; ++i) {
    mxGetPr(plhs[3])[i] = MatShoe.value[i];
  }
  free(gradu);
  free(gradt);
  free(gradb);
  free(grads);
  free(PtrPos);
  free(PtrNeg);
  mxFree(type);
}
