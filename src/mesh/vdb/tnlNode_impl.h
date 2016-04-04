#ifndef _TNLNODE_IMPL_H_INCLUDED_
#define _TNLNODE_IMPL_H_INCLUDED_

#include "tnlNode.h"

template< int LogX,
          int LogY >
int tnlNode< LogX, LogY >::getLevel()
{
    return this->level;
}

#endif // _TNLNODE_IMPL_H_INCLUDED_
