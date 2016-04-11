#ifndef _TNLINTERNALNODE_H_INCLUDED_
#define _TNLINTERNALNODE_H_INCLUDED_

#include <fstream>
#include "tnlNode.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"

//template< int LogX, int LogY >
//class tnlNode;

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

    void print( int splitX,
                int splitY,
                int depth,
                fstream& file );

    ~tnlInternalNode();

private:
    tnlNode< LogX, LogY >* children[ LogX * LogY ];
};


#include "tnlInternalNode_impl.h"
#endif // _TNLINTERNALNODE_H_INCLUDED_
