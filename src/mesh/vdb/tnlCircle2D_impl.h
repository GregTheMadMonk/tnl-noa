#ifndef _TNLCIRCLE2D_IMPL_H_INCLUDED_
#define _TNLCIRCLE2D_IMPL_H_INCLUDED_

#include <cmath>
#include "tnlCircle2D.h"

tnlCircle2D::tnlCircle2D( unsigned x,
                          unsigned y,
                          unsigned r )
{
    this->x = x;
    this->y = y;
    this->r = r;
}

tnlCircle2D::isIntercept( unsigned x1,
                          unsigned x2,
                          unsigned y1,
                          unsigned y2 )
{
    unsigned R = this->r * this->r;
    unsigned X = this->x * this->x;
    unsigned Y = this->y * this->y;

    unsigned aux = x1 - this->x;
    if( R - aux * aux > 0 &&
        this->isInInterval( y1, y2, sqrt( R - aux * aux ) + this->y ) )
        return true;
    
    aux = x2 - this->x;
    if( R - aux * aux > 0 &&
        this->isInInterval( y1, y2, sqrt( R - aux * aux ) + this->y ) )
        return true;

    aux = y1 - this->y;
    if( R - aux * aux > 0 &&
        this->isInInterval( x1, x2, sqrt( R - aux * aux ) + this->x ) )
        return true;

    aux = y2 - this->y;
    if( R - aux * aux > 0 &&
        this->isInInterval( x1, x2, sqrt( R - aux * aux ) + this->x ) )
        return true;

    return false;
}

tnlCircle2D::isInInterval( unsigned x1,
                           unsigned x2,
                           unsigned x )
{
    return ( ( x1 < x ) and ( x < x2 ) );
}

#endif // _TNLCIRCLE2D_IMPL_H_INCLUDED_
