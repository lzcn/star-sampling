#ifndef __UTIL_H__
#define __UTIL_H__

#include <vector>
#include <list>
#include <cmath>
#include <ctime>

typedef unsigned int uint;

int sgn(double x);
double sqr(double x);
double sigmoid(double x);
double timeDuration(clock_t start);
void doInsertReverse(double p, std::list<double> &listTop);
void doInsert(double p, std::list<double> &listTop);

////////////////////
// Implementation
///////////////////
double sqr(double x) { return x*x; }

double sigmoid(double x) { return (double)1.0/(1.0+exp(-x)); }

int sgn(double x) { return (x < 0 ? -1:1); }

double timeDuration(clock_t start){ 
	return (double)(clock() - start)/(double)CLOCKS_PER_SEC;
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

#endif /*__UTIL_H__*
