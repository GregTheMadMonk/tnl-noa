#ifndef _TNLAREA2D_H_INCLUDED_
#define _TNLAREA2D_H_INCLUDED_

template< typename Real >
class tnlArea2D
{
public:
    tnlArea2D( Real startX,
               Real endX,
               Real startY,
               Real endY );

    Real getStartX();

    Real getEndX();

    Real getLengthX();

    Real getStartY();

    Real getEndY();

    Real getLengthY();

    ~tnlArea2D(){};

private:
    Real startX;
    Real endX;
    Real startY;
    Real endY;
};

#include "tnlArea2D_impl.h"
#endif // _TNLAREA2D_H_INCLUDED_
