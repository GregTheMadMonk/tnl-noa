#ifndef _TNLLEAFNODE_H_INCLUDED_
#define _TNLLEAFNODE_H_INCLUDED_

#include "tnlNode.h"

//template< int LogX, int LogY >
//class tnlNode;

template< int LogX,
          int LogY = LogX >
class tnlLeafNode : public tnlNode< LogX, LogY >
{
public:
    tnlLeafNode( tnlArea2D* area,
                 tnlCircle2D* circle,
                 tnlBitmask* coordinates,
                 int level );

    void setNode( int splitX,
                  int splitY,
                  int depth );

    ~tnlLeafNode();

private:
};

#endif // _TNLLEAFNODE_H_INCLUDED_
