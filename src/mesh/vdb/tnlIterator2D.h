#ifndef _TNLITERATOR2D_H_INCLUDED_
#define _TNLITERATOR2D_H_INCLUDED_

#include <tnlBitmaskArray.h>

class tnlIterator2D
{
public:
    tnlIterator2D( unsigned logX,
                   unsigned logY,
                   unsigned parentX = 0,
                   unsigned parentY = 0,
                   unsigned level = 0 );

    void computeBitmaskArray( tnlBitmaskArray bitmaskArray,
                              unsigned size );

    ~tnlIterator2D(){};

private:
    unsigned level;
    unsigned parentX;
    unsigned parentY;
    unsigned logX;
    unsigned logY;
};

#include <tnlIterator2D_impl.h>
#endif // _TNLITERATOR2D_H_INCLUDED_
