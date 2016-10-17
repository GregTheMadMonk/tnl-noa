/***************************************************************************
                          EllpackIndexMultimap_impl.h  -  description
                             -------------------
    begin                : Sep 9, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Experimental/Multimaps/EllpackIndexMultimap.h>

namespace TNL {

template< typename Index,
          typename Device,
          typename LocalIndex >
EllpackIndexMultimap< Index, Device, LocalIndex >::
EllpackIndexMultimap()
:  keysRange( 0 ), valuesRange( 0 ), valuesMaxCount( 0 )
{
}

template< typename Index,
          typename Device,
          typename LocalIndex >
String
EllpackIndexMultimap< Index, Device, LocalIndex >::
getType()
{
   return String( "EllpackIndexMultimap< ") +
          String( TNL::getType< Index >() ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( ", " ) +
          String( TNL::getType< LocalIndexType >() ) +
          String( " >" );
}

template< typename Index,
          typename Device,
          typename LocalIndex >
String
EllpackIndexMultimap< Index, Device, LocalIndex >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimap< Index, Device, LocalIndex >::
setRanges( const IndexType inputs,
           const LocalIndexType outputs )
{
   Assert( inputs >= 0, );
   Assert( outputs >= 0, );
   this->keysRange = inputs;
   this->valuesRange = outputs;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
const Index
EllpackIndexMultimap< Index, Device, LocalIndex >::
getKeysRange() const
{
   return this->keysRange;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
const LocalIndex
EllpackIndexMultimap< Index, Device, LocalIndex >::
getValuesRange() const
{
   return this->valuesRange;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
bool
EllpackIndexMultimap< Index, Device, LocalIndex >::
allocate( const ValuesAllocationVectorType& portsCount )
{
   TNL_ASSERT( portsCount.getSize() == this->keysRange,
              std::cerr << "portsCount.getSize() =  " << portsCount.getSize()
                        << "this->inputs = " << this->keysRange );
   this->valuesMaxCount = portsCount.max();
 
   TNL_ASSERT( this->valuesMaxCount >= 0 && this->valuesMaxCount <= this->valuesRange,
              std::cerr << "this->portsMaxCount = " << this->valuesMaxCount
                        << " this->outputs = " << this->valuesRange );
   return this->values.setSize( this->keysRange * this->valuesMaxCount );
}

template< typename Index,
          typename Device,
          typename LocalIndex >
typename EllpackIndexMultimap< Index, Device, LocalIndex >::ValuesAccessorType
EllpackIndexMultimap< Index, Device, LocalIndex >::
getValues( const IndexType& inputIndex )
{
   return ValuesAccessorType( this->values.getData(), inputIndex, this->valuesMaxCount );
}

template< typename Index,
          typename Device,
          typename LocalIndex >
typename EllpackIndexMultimap< Index, Device, LocalIndex >::ConstValuesAccessorType
EllpackIndexMultimap< Index, Device, LocalIndex >::
getValues( const IndexType& inputIndex ) const
{
   return ConstValuesAccessorType( this->values.getData(), inputIndex, this->valuesMaxCount );
}

} // namespace TNL

