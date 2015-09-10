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


template< typename Device,
          typename Index >
tnlEllpackNetwork< Device, Index >::
tnlEllpackNetwork()
:  inputs( 0 ), outputs( 0 ), portsMaxCount( 0 )
{
}

template< typename Device,
          typename Index >
void 
tnlEllpackNetwork< Device, Index >::
setDimensions( const IndexType inputs,
               const IndexType outputs )
{
   this->inputs = inputs;
   this->outputs = outputs;
}

template< typename Device,
          typename Index >
const Index
tnlEllpackNetwork< Device, Index >::
getInputsCount() const
{
   return this->inputs;
}

template< typename Device,
          typename Index >
const Index
tnlEllpackNetwork< Device, Index >::
getOutputsCount() const
{
   return this->outputs;
}

template< typename Device,
          typename Index >
void
tnlEllpackNetwork< Device, Index >::
allocatePorts( const PortsAllocationVectorType& portsCount )
{
   tnlAssert( portsCount.getSize() == this->inputs,
              cerr << "portsCount.getSize() =  " << portsCount.getSize()
                   << "this->inputs = " << this->inputs );
   this->portsMaxCount = portsCount.max();
   
   tnlAssert( this->portsMaxCount >= 0 && this->portsMaxCount <= this->outputs, 
              cerr << "this->portsMaxCount = " << this->portsMaxCount
                   << " this->outputs = " << this->outputs );
}

template< typename Device,
          typename Index >
typename tnlEllpackNetwork< Device, Index >::LinksAccessorType 
tnlEllpackNetwork< Device, Index >::
getPorts( const IndexType& inputIndex )
{
   return PortsType( this->links.getData(), inputIndex, this->portsMaxCount );
}

template< typename Device,
          typename Index >
typename tnlEllpackNetwork< Device, Index >::ConstLinksAccessorType
tnlEllpackNetwork< Device, Index >::
getPorts( const IndexType& inputIndex ) const
{
   return ConstPortsType( this->links.getData(), inputIndex, this->portsMaxCount );
}




#endif	/* TNLELLPACKGRAPH_IMPL_H */

