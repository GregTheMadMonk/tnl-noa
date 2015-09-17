/***************************************************************************
                          tnlEllpackNetworkPorts_impl.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLELLPACKNETWORKPORTS_IMPL_H
#define TNLELLPACKNETWORKPORTS_IMPL_H

#include "tnlEllpackNetworkPorts.h"


template< typename Index,
          typename Device >
tnlEllpackNetworkPorts< Index, Device >::
tnlEllpackNetworkPorts()
{
}

template< typename Index,
          typename Device >
tnlEllpackNetworkPorts< Index, Device >::
tnlEllpackNetworkPorts( IndexType* networkPorts, 
                        const IndexType input,
                        const IndexType portsMaxCount )
{
   this->ports = &networkPorts[ input * portsMaxCount ];
   this->portsMaxCount = portsMaxCount;
}

template< typename Index,
          typename Device >
Index
tnlEllpackNetworkPorts< Index, Device >::
getPortsCount() const
{
   return this->portsMaxCount;
}

template< typename Index,
          typename Device >
void 
tnlEllpackNetworkPorts< Index, Device >::
setOutput( const IndexType portIndex,
           const IndexType output )
{
   this->ports[ portIndex ] = output;
}

template< typename Index,
          typename Device >
Index
tnlEllpackNetworkPorts< Index, Device >::
getOutput( const IndexType portIndex ) const
{
   return this->ports[ portIndex ];
}

template< typename Index,
          typename Device >
Index&
tnlEllpackNetworkPorts< Index, Device >::
operator[]( const IndexType portIndex )
{
   return this->ports[ portIndex ];
}

template< typename Index,
          typename Device >
const Index&
tnlEllpackNetworkPorts< Index, Device >::
operator[]( const IndexType portIndex ) const
{
   return this->ports[ portIndex ];
}

template< typename Index,
          typename Device >
void
tnlEllpackNetworkPorts< Index, Device >::
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
          typename Device >
std::ostream& operator << ( std::ostream& str, const tnlEllpackNetworkPorts< Index, Device>& ports )
{
   ports.print( str );
   return str;
}

#endif	/* TNLELLPACKGRAPHLINKSACCESSOR_IMPL_H */

