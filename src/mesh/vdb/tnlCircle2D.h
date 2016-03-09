#ifndef _TNLCIRCLE2D_H_INCLUDED_
#define _TNLCIRCLE2D_H_INCLUDED_

class tnlCircle2D
{
public:
    tnlCircle2D( unsigned x,
                 unsigned y,
                 unsigned r );

    bool isIntercept( unsigned x1,
                      unsigned x2,
                      unsigned y1,
                      unsigned y2 );

    bool isInInterval( unsigned x1,
                       unsigned x2,
                       unsigned x );

    ~tnlCircle2D()

private:
    unsigned x;
    unsigned y;
    unsigned r;
};

#endif // _TNLCIRCLE2D_H_INCLUDED_
