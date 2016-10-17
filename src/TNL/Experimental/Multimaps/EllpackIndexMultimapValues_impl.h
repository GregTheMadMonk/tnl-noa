/***************************************************************************
                          EllpackIndexMultimapValues_impl.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "EllpackIndexMultimapValues.h"

#include <TNL/Assert.h>

namespace TNL {
 
template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues()
{
}

template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
EllpackIndexMultimapValues( IndexType* networkPorts,
                            const IndexType& input,
                            const LocalIndexType& portsMaxCount )
{
   this->ports = &networkPorts[ input * portsMaxCount ];
   this->portsMaxCount = portsMaxCount;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
LocalIndex
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getPortsCount() const
{
   return this->portsMaxCount;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
setOutput( const LocalIndexType& portIndex,
           const IndexType& output )
{
   Assert( portIndex < this->portsMaxCount,
              std::cerr << " portIndex = " << portIndex
                        << " portsMaxCount = " << this->portsMaxCount
                        << std::endl );
   this->ports[ portIndex ] = output;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
Index
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
getOutput( const LocalIndexType& portIndex ) const
{
   Assert( portIndex < this->portsMaxCount,
              std::cerr << " portIndex = " << portIndex
                        << " portsMaxCount = " << this->portsMaxCount
                        << std::endl );
   return this->ports[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
Index&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator[]( const LocalIndexType& portIndex )
{
   Assert( portIndex < this->portsMaxCount,
              std::cerr << " portIndex = " << portIndex
                        << " portsMaxCount = " << this->portsMaxCount
                        << std::endl );
   return this->ports[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
const Index&
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
operator[]( const LocalIndexType& portIndex ) const
{
   Assert( portIndex < this->portsMaxCount,
              std::cerr << " portIndex = " << portIndex
                        << " portsMaxCount = " << this->portsMaxCount
                        << std::endl );
   return this->ports[ portIndex ];
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimapValues< Index, Device, LocalIndex >::
print( std::ostream& str ) const
{
   if( this->getPortsCount() == 0 )
   {
      str << "[]";
      return;
   }
   str << "[ " << this->getOutput( 0 );
   for( Index i = 1; i < this->getPortsCount(); i++ )
      str << ", " << this->getOutput( i );
   str << " ]";
}

template< typename Index,
          typename Device,
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimapValues< Index, Device, LocalIndex >& ports )
{
   ports.print( str );
   return str;
}

} // namespace TNL

