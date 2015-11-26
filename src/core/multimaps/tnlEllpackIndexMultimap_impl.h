/***************************************************************************
                          tnlEllpackIndexMultimap_impl.h  -  description
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

#ifndef TNLELLPACKINDEXMULTIMAP_IMPL_H
#define	TNLELLPACKINDEXMULTIMAP_IMPL_H

#include <core/multimaps/tnlEllpackIndexMultimap.h>
#include <core/multimaps/tnlEllpackIndexMultimapValues.h>


template< typename Index,
          typename Device >
tnlEllpackIndexMultimap< Index, Device >::
tnlEllpackIndexMultimap()
:  keysRange( 0 ), valuesRange( 0 ), valuesMaxCount( 0 )
{
}

template< typename Index,
          typename Device >
tnlString tnlEllpackIndexMultimap< Index, Device > :: getType()
{
   return tnlString( "tnlEllpackIndexMultimap< ") +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( ::getType< Index >() ) +                    
          tnlString( " >" );
}

template< typename Index,
          typename Device >
tnlString tnlEllpackIndexMultimap< Index, Device >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Index,
          typename Device >
void 
tnlEllpackIndexMultimap< Index, Device >::
setRanges( const IndexType inputs,
               const IndexType outputs )
{
   this->keysRange = inputs;
   this->valuesRange = outputs;
}

template< typename Index,
          typename Device >
const Index
tnlEllpackIndexMultimap< Index, Device >::
getKeysRange() const
{
   return this->keysRange;
}

template< typename Index,
          typename Device >
const Index
tnlEllpackIndexMultimap< Index, Device >::
getValuesRange() const
{
   return this->valuesRange;
}

template< typename Index,
          typename Device >
void
tnlEllpackIndexMultimap< Index, Device >::
allocate( const ValuesAllocationVectorType& portsCount )
{
   tnlAssert( portsCount.getSize() == this->keysRange,
              cerr << "portsCount.getSize() =  " << portsCount.getSize()
                   << "this->inputs = " << this->keysRange );
   this->valuesMaxCount = portsCount.max();
   
   tnlAssert( this->valuesMaxCount >= 0 && this->valuesMaxCount <= this->valuesRange, 
              cerr << "this->portsMaxCount = " << this->valuesMaxCount
                   << " this->outputs = " << this->valuesRange );
   this->values.setSize( this->keysRange * this->valuesMaxCount );
}

template< typename Index,
          typename Device >
typename tnlEllpackIndexMultimap< Index, Device >::ValuesAccessorType 
tnlEllpackIndexMultimap< Index, Device >::
getValues( const IndexType& inputIndex )
{
   return ValuesAccessorType( this->values.getData(), inputIndex, this->valuesMaxCount );
}

template< typename Index,
          typename Device >
typename tnlEllpackIndexMultimap< Index, Device >::ConstValuesAccessorType
tnlEllpackIndexMultimap< Index, Device >::
getValues( const IndexType& inputIndex ) const
{
   return ConstPortsType( this->values.getData(), inputIndex, this->valuesMaxCount );
}

#endif	/* TNLELLPACKGRAPH_IMPL_H */

