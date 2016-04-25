#ifndef _TNLROOTNODE_H_INCLUDED_
#define _TNLROOTNODE_H_INCLUDED_

#include "tnlNode.h"

template< typename Real,
          typename Index,
          unsigned size,
          Index LogX,
          Index LogY = LogX >
class tnlRootNode : public tnlNode< Real, Index, LogX, LogY >
{
public:
    tnlRootNode( tnlArea2D< Real >* area,
                 tnlCircle2D< Real >* circle,
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
    tnlNode< Real, Index, LogX, LogY >* children[ size ];
    unsigned depth;
};

#include "tnlRootNode_impl.h"
#endif // _TNLROOTNODE_H_INCLUDED_
