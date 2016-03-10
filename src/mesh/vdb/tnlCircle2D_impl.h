#ifndef _TNLCIRCLE2D_IMPL_H_INCLUDED_
#define _TNLCIRCLE2D_IMPL_H_INCLUDED_

#include <cmath>
#include "tnlCircle2D.h"

tnlCircle2D::tnlCircle2D( unsigned a,
                          unsigned b,
                          unsigned r )
{
    this->a = a;
    this->b = b;
    this->r = r;
}

bool tnlCircle2D::isIntercept( float x1,
                               float x2,
                               float y1,
                               float y2 )
{
    float R = this->r * this->r;
    float A = this->a * this->a;
    float B = this->b * this->b;

    float aux = x1 - this->a;
    if( R - aux * aux >= 0 &&
        this->isInInterval( y1, y2, sqrt( R - aux * aux ) + this->b ) )
        return true;
    
    aux = x2 - this->a;
    if( R - aux * aux >= 0 &&
        this->isInInterval( y1, y2, sqrt( R - aux * aux ) + this->b ) )
        return true;

    aux = y1 - this->b;
    if( R - aux * aux >= 0 &&
        this->isInInterval( x1, x2, sqrt( R - aux * aux ) + this->a ) )
        return true;

    aux = y2 - this->b;
    if( R - aux * aux >= 0 &&
        this->isInInterval( x1, x2, sqrt( R - aux * aux ) + this->a ) )
        return true;

    return false;
}

bool tnlCircle2D::isInInterval( float x1,
                                float x2,
                                float x )
{
    return ( ( x1 <= x ) and ( x <= x2 ) );
}

#endif // _TNLCIRCLE2D_IMPL_H_INCLUDED_
