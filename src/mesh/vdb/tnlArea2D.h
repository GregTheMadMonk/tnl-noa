#ifndef _TNLAREA2D_H_INCLUDED_
#define _TNLAREA2D_H_INCLUDED_

class tnlArea2D
{
public:
    tnlArea2D( unsigned startX,
               unsigned endX,
               unsigned startY,
               unsigned endY );

    unsigned getStartX();

    unsigned getEndX();

    unsigned getStartY();

    unsigned getEndY();

    ~tnlArea2D(){};

private:
    unsigned startX;
    unsigned endX;
    unsigned startY;
    unsigned endY;
};

#include "tnlArea2D_impl.h"
#endif // _TNLAREA2D_H_INCLUDED_
