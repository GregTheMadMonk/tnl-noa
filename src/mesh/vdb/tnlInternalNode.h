#ifndef _TNLINTERNALNODE_H_INCLUDED_
#define _TNLINTERNALNODE_H_INCLUDED_

#include "tnlNode.h"
#include "tnlArea2D"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"
#include "tnlBitmaskArray.h"

template< int LogX,
          int LogY = LogX >
class tnlInternalNode : public tnlNode
{
public:
    tnlInternalNode( tnlArea2D* area,
                     tnlCircle2D* circle,
                     tnlBitmask* coordinates,
                     int level );

    void setNode( int splitX,
                  int splitY,
                  int depth );

    void setChildren( int splitX,
                      int splitY,
                      int depth );

    ~tnlInternalNode();

private:
    tnlNode< LogX, LogY >* children[ LogX * LogY ];
};


#include "tnlInternalNode_impl.h"
#endif // _TNLINTERNALNODE_H_INCLUDED_
