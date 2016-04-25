#ifndef _TNLLEAFNODE_H_INCLUDED_
#define _TNLLEAFNODE_H_INCLUDED_

#include <fstream>
#include "tnlNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include "tnlBitmask.h"

template< int LogX,
          int LogY = LogX >
class tnlLeafNode : public tnlNode< LogX, LogY >
{
public:
    tnlLeafNode( tnlArea2D* area,
                 tnlCircle2D* circle,
                 int X,
                 int Y,
                 int level );

    void setNode( int splitX,
                  int splitY,
                  int depth );

    void write( fstream& file,
                int level );

    ~tnlLeafNode();

private:
    tnlBitmaskArray< LogX * LogY >* bitmaskArray;
};

#include "tnlLeafNode_impl.h"
#endif // _TNLLEAFNODE_H_INCLUDED_
