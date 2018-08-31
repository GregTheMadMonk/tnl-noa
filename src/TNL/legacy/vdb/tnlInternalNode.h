#ifndef _TNLINTERNALNODE_H_INCLUDED_
#define _TNLINTERNALNODE_H_INCLUDED_

#include "tnlNode.h"

template< typename Real,
          typename Index,
          Index LogX,
          Index LogY = LogX >
class tnlInternalNode : public tnlNode< Real, Index, LogX, LogY >
{
public:
    tnlInternalNode( tnlArea2D< Real >* area,
                     tnlCircle2D< Real >* circle,
                     Index X,
                     Index Y,
                     Index level );

    void setNode( Index splitX,
                  Index splitY,
                  Index depth );

    void setChildren( Index splitX,
                      Index splitY,
                      Index depth );

    void write( fstream& f,
                Index level );

    ~tnlInternalNode();

private:
    tnlBitmaskArray< LogX * LogY >* bitmaskArray;
    tnlNode< Real, Index, LogX, LogY >* children[ LogX * LogY ];
};


#include "tnlInternalNode_impl.h"
#endif // _TNLINTERNALNODE_H_INCLUDED_
