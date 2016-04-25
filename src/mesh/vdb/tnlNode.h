#ifndef _TNLNODE_H_INCLUDED_
#define _TNLNODE_H_INCLUDED_

#include "tnlBitmaskArray.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include <fstream>


template< int LogX,
          int LogY = LogX >
class tnlNode
{
public:
    tnlNode( tnlArea2D* area,
             tnlCircle2D* circle,
             int X,
             int Y,
             int level );

    void setNode( int splitX,
                  int splitY,
                  tnlBitmaskArray< LogX * LogY >* bitmaskArray );

    virtual void setNode( int splitX = 0,
                          int splitY = 0,
                          int depth = 0 ){};

    virtual void print( int splitX,
                        int splitY,
                        int depth,
                        fstream& file ){};

    virtual void write( fstream& f,
                        int level ){};

    int getLevel();

    ~tnlNode();

protected:
    tnlArea2D* area;
    tnlCircle2D* circle;
    int X, Y, level;
};

#include "tnlNode_impl.h"
#endif // _TNLNODE_H_INCLUDED_
