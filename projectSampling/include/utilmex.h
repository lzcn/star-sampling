#ifndef __UTILMEX_H__
#define __UTILMEX_H__

#include <ctime>
#include "mex.h"

void progressbar(double progress){
    int barWidth = 40;
    int pos = (int)(barWidth * progress);
    mexPrintf("[");
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)  mexPrintf("=");
        else if (i == pos) mexPrintf(">");
        else mexPrintf(" ");
    }
    mexPrintf("]");
    mexPrintf("%3d%%",int(progress * 100.0));
    mexEvalString("drawnow");
}
void clearprogressbar(){
    int barWidth = 46;
    for (int i = 0; i < barWidth; ++i) mexPrintf("\b");
    mexEvalString("drawnow");
}

class Timer{
public:
    Timer();
    ~Timer(){};
    void r_init(clock_t start){initialization += (double)(clock()-start)/CLOCKS_PER_SEC;}
    void r_samp(clock_t start){sampling += (double)(clock()-start)/CLOCKS_PER_SEC;}
    void r_score(clock_t start){scoring += (double)(clock()-start)/CLOCKS_PER_SEC;}
    void r_filter(clock_t start){filtering += (double)(clock()-start)/CLOCKS_PER_SEC;}
public:
    double initialization;
    double sampling;
    double scoring;
    double filtering;
};

Timer::Timer(){
    initialization = 0.0;
    sampling = 0.0;
    scoring = 0.0;
    filtering = 0.0;
}
#endif /*__UTILMEX_H__*/
