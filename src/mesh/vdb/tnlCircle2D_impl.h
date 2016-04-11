#ifndef _TNLCIRCLE2D_IMPL_H_INCLUDED_
#define _TNLCIRCLE2D_IMPL_H_INCLUDED_

#include <iostream>
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
                               float y2,
                               bool verbose )
{
    if( this->isInInterval( x1, x2, this->a - this->r ) &&
        this->isInInterval( x1, x2, this->a + this->r ) &&
        this->isInInterval( y1, y2, this->b - this->r ) &&
        this->isInInterval( y1, y2, this->b + this->r ) )
    {
        if( verbose )
            std::cout << "Circle is inside area." << std::endl;
        return true;
    }
    else if( verbose )
        std::cout << "Circle is not inside area." << std::endl;

    float R = this->r * this->r;

    float aux = x1 - this->a;
    if( R - aux * aux >= 0 &&
        ( this->isInInterval( y1, y2, sqrt( R - aux * aux ) + this->b ) ||
        this->isInInterval( y1, y2, -sqrt( R - aux * aux ) + this->b ) ) )
    {
        if( verbose )
            std::cout << "Circle intercepts left boundry of area." << std::endl;
        return true;
    }
    
    aux = x2 - this->a;
    if( R - aux * aux >= 0 &&
        ( this->isInInterval( y1, y2, sqrt( R - aux * aux ) + this->b ) ||
        this->isInInterval( y1, y2, -sqrt( R - aux * aux ) + this->b ) ) )
    {
        if( verbose )
            std::cout << "Circle intercepts right boundry of area." << std::endl;
        return true;
    }

    aux = y1 - this->b;
    if( R - aux * aux >= 0 &&
        ( this->isInInterval( x1, x2, sqrt( R - aux * aux ) + this->a ) ||
        this->isInInterval( x1, x2, -sqrt( R - aux * aux ) + this->a ) ) )
    {
        if( verbose )
            std::cout << "Circle intercepts bottom boundry of area." << std::endl;
        return true;
    }

    aux = y2 - this->b;
    if( R - aux * aux >= 0 &&
        ( this->isInInterval( x1, x2, sqrt( R - aux * aux ) + this->a ) ||
        this->isInInterval( x1, x2, sqrt( R - aux * aux ) + this->a ) ) )
    {
        if( verbose )
            std::cout << "Circle intercepts top boundry of area." << std::endl;
        return true;
    }

    if( verbose )
        std::cout << "Circle does not intercept area." << std::endl;

    return false;
}

bool tnlCircle2D::isInInterval( float x1,
                                float x2,
                                float x )
{
    return ( ( x1 <= x ) and ( x <= x2 ) );
}

#endif // _TNLCIRCLE2D_IMPL_H_INCLUDED_
