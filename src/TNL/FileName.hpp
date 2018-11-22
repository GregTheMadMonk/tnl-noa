/***************************************************************************
                          FileName.hpp  -  description
                             -------------------
    begin                : Sep 28, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/FileName.h>

namespace TNL {

template< typename Coordinates >
void
FileName::
setDistributedSystemNodeId( const Coordinates& nodeId )
{
   this->distributedSystemNodeId = "-";
   this->distributedSystemNodeId += convertToString( nodeId[ 0 ] );
   for( int i = 1; i < nodeId.getSize(); i++ )
   {
      this->distributedSystemNodeId += "-";
      this->distributedSystemNodeId += convertToString( nodeId[ i ] );
   }
}

} // namespace TNL
