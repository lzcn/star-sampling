#ifndef __UTILMEX_H__
#define __UTILMEX_H__

#include "mex.h"

void progressbar(double progress){
    int barWidth = 40;
    int pos = barWidth * progress;
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
#endif /*__UTILMEX_H__*/
