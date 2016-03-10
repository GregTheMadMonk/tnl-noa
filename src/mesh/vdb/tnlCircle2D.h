#ifndef _TNLCIRCLE2D_H_INCLUDED_
#define _TNLCIRCLE2D_H_INCLUDED_

class tnlCircle2D
{
public:
    tnlCircle2D( unsigned a,
                 unsigned b,
                 unsigned r );

    bool isIntercept( float x1,
                      float x2,
                      float y1,
                      float y2 );

    bool isInInterval( float x1,
                       float x2,
                       float x );

    ~tnlCircle2D();

private:
    // x and y define center of the circle
    // r defines its radius
    unsigned a;
    unsigned b;
    unsigned r;
};

#include "tnlCircle2D_impl.h"
#endif // _TNLCIRCLE2D_H_INCLUDED_
