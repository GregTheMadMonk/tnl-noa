#ifndef _TNLAREA2D_IMPL_H_INCLUDED_
#define _TNLAREA2D_IMPL_H_INCLUDED_

#include "tnlArea2D.h"

tnlArea2D::tnlArea2D( unsigned startX,
                      unsigned endX,
                      unsigned startY,
                      unsigned endY )
{
    this->startX = startX;
    this->endX = endX;
    this->startY = startY;
    this->endY = endY;
}

unsigned tnlArea2D::getStartX()
{
    return this->startX;
}

unsigned tnlArea2D::getEndX()
{
    return this->endX;
}

unsigned tnlArea2D::getStartY()
{
    return this->startY;
}

unsigned tnlArea2D::getEndY()
{
    return this->endY;
}

#endif // _TNLAREA2D_IMPL_H_INCLUDED_
