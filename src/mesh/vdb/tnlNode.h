#ifndef _TNLNODE_H_INCLUDED_
#define _TNLNODE_H_INCLUDED_

#include "tnlBitmaskArray.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"


template< int LogX,
          int LogY = LogX >
class tnlNode
{
public:
    tnlNode(){};

    virtual void setNode( int splitX,
                          int splitY,
                          int depth ){};

    int getLevel();

    ~tnlNode(){};

protected:
    tnlArea2D* area;
    tnlCircle2D* circle;
    int X, Y;
    int level;
    tnlBitmaskArray< LogX * LogY >* bitmaskArray;

private:
    int size;
};

#include "tnlNode_impl.h"
#endif // _TNLNODE_H_INCLUDED_
