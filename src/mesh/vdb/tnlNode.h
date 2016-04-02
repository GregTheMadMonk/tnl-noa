#ifndef _TNLNODE_H_INCLUDED_
#define _TNLNODE_H_INCLUDED_

#include "tnlBitmaskArray.h"
#include "tnlArea.h"
#include "tnlCircle2D.h"


template< int LogX,
          int LogY = logX >
class tnlNode
{
public:
    tnlNode(){};

    virtual void setChildren();

    ~tnlNode(){};

private:
    int size;
    tnlNode< LogX, LogY >* childNodes[ LogX * LogY ];
    tnlArea2D* area;
    tnlCircle2D* circle;
    tnlBitmask< LogX * LogY >* bitmaskArray;
};

#include "tnlNode_impl.h"
#endif // _TNLNODE_H_INCLUDED_
