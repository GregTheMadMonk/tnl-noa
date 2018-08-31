#ifndef _TNLAREA2D_IMPL_H_INCLUDED_
#define _TNLAREA2D_IMPL_H_INCLUDED_

#include "tnlArea2D.h"

template< typename Real >
tnlArea2D< Real >::tnlArea2D( Real startX,
                              Real endX,
                              Real startY,
                              Real endY )
{
    this->startX = startX;
    this->endX = endX;
    this->startY = startY;
    this->endY = endY;
}

template< typename Real >
Real tnlArea2D< Real >::getStartX()
{
    return this->startX;
}

template< typename Real >
Real tnlArea2D< Real >::getEndX()
{
    return this->endX;
}

template< typename Real >
Real tnlArea2D< Real >::getLengthX()
{
    return this->endX - this->startX;
}

template< typename Real >
Real tnlArea2D< Real >::getStartY()
{
    return this->startY;
}

template< typename Real >
Real tnlArea2D< Real >::getEndY()
{
    return this->endY;
}

template< typename Real >
Real tnlArea2D< Real >::getLengthY()
{
    return this->endY - this->startY;
}

#endif // _TNLAREA2D_IMPL_H_INCLUDED_
