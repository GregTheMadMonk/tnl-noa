#ifndef _TNLROOTNODE_H_INCLUDED_
#define _TNLROOTNODE_H_INCLUDED_

#include <fstream>
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
                 unsigned depth );

    void setNode();

    void createTree();

    void printStates( fstream& file );

    ~tnlRootNode();

    friend class tnlNode< LogX, LogY >;

private:
    tnlArea2D* area;
    unsigned nodesX;
    unsigned nodesY;
    tnlCircle2D* circle;
    tnlBitmaskArray< size >* bitmaskArray;
    tnlNode< LogX, LogY >* children[ LogX * LogY ];
    unsigned level;
    unsigned depth;
};

#include "tnlRootNode_impl.h"
#endif // _TNLROOTNODE_H_INCLUDED_
