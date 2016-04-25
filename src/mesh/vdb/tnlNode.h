#ifndef _TNLNODE_H_INCLUDED_
#define _TNLNODE_H_INCLUDED_

#include "tnlBitmaskArray.h"
#include "tnlArea2D.h"
#include "tnlCircle2D.h"
#include <fstream>


template< typename Real,
          typename Index,
          Index LogX,
          Index LogY = LogX >
class tnlNode
{
public:
    tnlNode( tnlArea2D< Real >* area,
             tnlCircle2D< Real >* circle,
             Index X,
             Index Y,
             Index level );

    void setNode( Index splitX,
                  Index splitY,
                  tnlBitmaskArray< LogX * LogY >* bitmaskArray );

    virtual void setNode( Index splitX = 0,
                          Index splitY = 0,
                          Index depth = 0 ){};

    virtual void write( fstream& f,
                        Index level ){};

    Index getLevel();

    ~tnlNode();

protected:
    tnlArea2D< Real >* area;
    tnlCircle2D< Real >* circle;
    Index X, Y, level;
};

#include "tnlNode_impl.h"
#endif // _TNLNODE_H_INCLUDED_
