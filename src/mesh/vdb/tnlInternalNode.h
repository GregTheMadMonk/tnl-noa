#ifndef _TNLINTERNALNODE_H_INCLUDED_
#define _TNLINTERNALNODE_H_INCLUDED_

#include "tnlNode.h"

template< int LogX,
          int LogY = LogX >
class tnlInternalNode : public tnlNode< LogX, LogY >
{
public:
    tnlInternalNode( tnlArea2D* area,
                     tnlCircle2D* circle,
                     int X,
                     int Y,
                     int level );

    void setNode( int splitX,
                  int splitY,
                  int depth );

    void setChildren( int splitX,
                      int splitY,
                      int depth );

    void write( fstream& f,
                int level );

    ~tnlInternalNode();

private:
    tnlBitmaskArray< LogX * LogY >* bitmaskArray;
    tnlNode< LogX, LogY >* children[ LogX * LogY ];
};


#include "tnlInternalNode_impl.h"
#endif // _TNLINTERNALNODE_H_INCLUDED_
