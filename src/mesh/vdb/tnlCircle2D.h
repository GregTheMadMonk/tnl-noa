#ifndef _TNLCIRCLE2D_H_INCLUDED_
#define _TNLCIRCLE2D_H_INCLUDED_

template< typename Real >
class tnlCircle2D
{
public:
    tnlCircle2D( unsigned a,
                 unsigned b,
                 unsigned r );

    bool isIntercept( Real x1,
                      Real x2,
                      Real y1,
                      Real y2,
                      bool verbose = false );

    bool isInInterval( Real x1,
                       Real x2,
                       Real x );

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
