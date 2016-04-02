#ifndef _TNLROOTNODE_H_INCLUDED_
#define _TNLROOTNODE_H_INCLUDED_

#include "tnlBitmaskArray.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlNode.h"

template< unsigned size,
          int LogX,
          int LogY = LogX >
class tnlRootNode
{
public:
    tnlRootNode( tnlArea2D* area,
                 tnlCircle2D* circle,
                 unsigned nodesX,
                 unsigned nodesY,
                 unsigned childSplitX,
                 unsigned childSplitY,
                 unsigned treeDepth );

    void setNode();

    void createTree();

    void printStates();

    ~tnlRootNode();

private:
    tnlArea2D* area;
    unsigned nodesX;
    unsigned nodesY;
    tnlCircle2D* circle;
    tnlBitmaskArray< size >* bitmaskArray;
    tnlNode< LogX, LogY >* children[ LogX * LogY ];
    unsigned level;
    unsigned treeDepth;
};

#include "tnlRootNode_impl.h"
#endif // _TNLROOTNODE_H_INCLUDED_
