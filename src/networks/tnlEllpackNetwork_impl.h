/***************************************************************************
                          tnlEllpackNetwork_impl.h  -  description
                             -------------------
    begin                : Sep 9, 2015
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

#ifndef TNLELLPACKNETWORK_IMPL_H
#define	TNLELLPACKNETWORK_IMPL_H

#include <networks/tnlEllpackNetwork.h>
#include <networks/tnlEllpackNetworkPorts.h>


template< typename Index,
          typename Device >
tnlEllpackNetwork< Index, Device >::
tnlEllpackNetwork()
:  inputs( 0 ), outputs( 0 ), portsMaxCount( 0 )
{
}

template< typename Index,
          typename Device >
tnlString tnlEllpackNetwork< Index, Device > :: getType()
{
   return tnlString( "tnlEllpackNetwork< ") +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( ::getType< Index >() ) +                    
          tnlString( " >" );
}

template< typename Index,
          typename Device >
tnlString tnlEllpackNetwork< Index, Device >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Index,
          typename Device >
void 
tnlEllpackNetwork< Index, Device >::
setDimensions( const IndexType inputs,
               const IndexType outputs )
{
   this->inputs = inputs;
   this->outputs = outputs;
}

template< typename Index,
          typename Device >
const Index
tnlEllpackNetwork< Index, Device >::
getInputsCount() const
{
   return this->inputs;
}

template< typename Index,
          typename Device >
const Index
tnlEllpackNetwork< Index, Device >::
getOutputsCount() const
{
   return this->outputs;
}

template< typename Index,
          typename Device >
void
tnlEllpackNetwork< Index, Device >::
allocatePorts( const PortsAllocationVectorType& portsCount )
{
   tnlAssert( portsCount.getSize() == this->inputs,
              cerr << "portsCount.getSize() =  " << portsCount.getSize()
                   << "this->inputs = " << this->inputs );
   this->portsMaxCount = portsCount.max();
   
   tnlAssert( this->portsMaxCount >= 0 && this->portsMaxCount <= this->outputs, 
              cerr << "this->portsMaxCount = " << this->portsMaxCount
                   << " this->outputs = " << this->outputs );
   this->ports.setSize( this->inputs * this->portsMaxCount );
}

template< typename Index,
          typename Device >
typename tnlEllpackNetwork< Index, Device >::PortsType 
tnlEllpackNetwork< Index, Device >::
getPorts( const IndexType& inputIndex )
{
   return PortsType( this->ports.getData(), inputIndex, this->portsMaxCount );
}

template< typename Index,
          typename Device >
typename tnlEllpackNetwork< Index, Device >::ConstPortsType
tnlEllpackNetwork< Index, Device >::
getPorts( const IndexType& inputIndex ) const
{
   return ConstPortsType( this->ports.getData(), inputIndex, this->portsMaxCount );
}

#endif	/* TNLELLPACKGRAPH_IMPL_H */

