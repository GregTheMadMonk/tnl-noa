#ifndef _TNLROOTNODE_H_INCLUDED_
#define _TNLROOTNODE_H_INCLUDED_

#include "tnlBitmaskArray.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"

template< unsigned size >
class tnlRootNode
{
public:
    tnlRootNode( tnlArea2D* area,
                 tnlCircle2D* circle,
                 unsigned nodesX,
                 unsigned nodesY );

    void setNode();

    void printStates();

    ~tnlRootNode();

private:
    tnlArea2D* area;
    unsigned nodesX;
    unsigned nodesY;
    tnlCircle2D* circle;
    tnlBitmaskArray< size >* bitmaskArray;
};

#include "tnlRootNode_impl.h"
#endif // _TNLROOTNODE_H_INCLUDED_
