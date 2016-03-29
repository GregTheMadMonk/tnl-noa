#ifndef _TNLINTERNALNODE_H_INCLUDED_
#define _TNLINTERNALNODE_H_INCLUDED_

#include "tnlNode.h"
#include "tnlBitmask.h"

template< int LogX,
          int LogY = LogX >
class tnlInternalNode : public tnlNode
{
public:
    tnlInternalNode();

    ~tnlInternalNode();

private:
    
};


#include "tnlInternalNode_impl.h"
#endif // _TNLINTERNALNODE_H_INCLUDED_
