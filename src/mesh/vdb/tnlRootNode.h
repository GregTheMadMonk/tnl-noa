#ifndef _TNLROOTNODE_H_INCLUDED_
#define _TNLROOTNODE_H_INCLUDED_

#include "tnlNode.h"

template< unsigned size,
          int LogX,
          int LogY = LogX >
class tnlRootNode : public tnlNode< LogX, LogY >
{
public:
    tnlRootNode( tnlArea2D* area,
                 tnlCircle2D* circle,
                 unsigned nodesX,
                 unsigned nodesY,
                 unsigned depth );

    void setNode();

    void createTree();

    void write();

    ~tnlRootNode();

private:
    unsigned nodesX;
    unsigned nodesY;
    tnlBitmaskArray< size >* bitmaskArray;
    tnlNode< LogX, LogY >* children[ size ];
    unsigned depth;
};

#include "tnlRootNode_impl.h"
#endif // _TNLROOTNODE_H_INCLUDED_
