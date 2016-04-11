#ifndef _TNLITERATOR2D_H_INCLUDED_
#define _TNLITERATOR2D_H_INCLUDED_

#include "tnlBitmaskArray.h"
#include "tnlCircle2D.h"

template< unsigned size,
          int LogX,
          int LogY = LogX >
class tnlIterator2D
{
public:
    tnlIterator2D( unsigned cellsX,
                   unsigned cellsY,
                   float stepX,
                   float stepY,
                   float startX,
                   float startY );

    void computeBitmaskArray( tnlBitmaskArray< size >* bitmaskArray,
                              tnlCircle2D* circle );

    ~tnlIterator2D(){};

private:
    unsigned cellsX;
    unsigned cellsY;
    float stepX;
    float stepY;
    float startX;
    float startY;
};

#include "tnlIterator2D_impl.h"
#endif // _TNLITERATOR2D_H_INCLUDED_
