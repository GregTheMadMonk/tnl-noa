#ifndef _TNLLEAFNODE_H_INCLUDED_
#define _TNLLEAFNODE_H_INCLUDED_

#include <fstream>
#include "tnlNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY = LogX >
class tnlLeafNode : public tnlNode< Real, Index, LogX, LogY >
{
public:
    tnlLeafNode( tnlArea2D< Real >* area,
                 tnlCircle2D< Real >* circle,
                 Index X,
                 Index Y,
                 Index level );

    void setNode( Index splitX,
                  Index splitY,
                  Index depth );

    void write( fstream& file,
                Index level );

    ~tnlLeafNode();

private:
    tnlBitmaskArray< LogX * LogY >* bitmaskArray;
};

#include "tnlLeafNode_impl.h"
#endif // _TNLLEAFNODE_H_INCLUDED_
