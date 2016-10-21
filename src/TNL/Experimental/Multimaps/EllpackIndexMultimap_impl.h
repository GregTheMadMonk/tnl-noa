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
: keysRange( 0 ), valuesRange( 0 ), maxValuesCount( 0 )
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
allocate( const LocalIndexType& maxValuesCount )
{
   Assert( maxValuesCount >= 0 && maxValuesCount <= this->valuesRange,
              std::cerr << "maxValuesCount = " << maxValuesCount
                        << " this->valuesRange = " << this->valuesRange );
   this->maxValuesCount = maxValuesCount;
   if( ! this->values.setSize( this->keysRange * ( this->maxValuesCount + 1 ) ) )
      return false;
   // TODO: maybe the local sizes should be stored differently?
   for( IndexType i = 0; i < this->keysRange; i++ )
      this->getValues( i ).setSize( maxValuesCount );
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
bool
EllpackIndexMultimap< Index, Device, LocalIndex >::
allocate( const ValuesAllocationVectorType& valuesCounts )
{
   Assert( valuesCounts.getSize() == this->keysRange,
              std::cerr << "valuesCounts.getSize() =  " << valuesCounts.getSize()
                        << "this->keysRange = " << this->keysRange );
   this->maxValuesCount = valuesCounts.max();
 
   Assert( this->maxValuesCount >= 0 && this->maxValuesCount <= this->valuesRange,
              std::cerr << "this->maxValuesCount = " << this->maxValuesCount
                        << " this->valuesRange = " << this->valuesRange );
   if( ! this->values.setSize( this->keysRange * ( this->maxValuesCount + 1 ) ) )
      return false;
   // TODO: maybe the local sizes should be stored differently?
   for( IndexType i = 0; i < this->keysRange; i++ )
      this->getValues( i ).setSize( valuesCounts[ i ] );
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
typename EllpackIndexMultimap< Index, Device, LocalIndex >::ValuesAccessorType
EllpackIndexMultimap< Index, Device, LocalIndex >::
getValues( const IndexType& inputIndex )
{
   return ValuesAccessorType( this->values.getData(), inputIndex, this->maxValuesCount );
}

template< typename Index,
          typename Device,
          typename LocalIndex >
typename EllpackIndexMultimap< Index, Device, LocalIndex >::ConstValuesAccessorType
EllpackIndexMultimap< Index, Device, LocalIndex >::
getValues( const IndexType& inputIndex ) const
{
   return ConstValuesAccessorType( this->values.getData(), inputIndex, this->maxValuesCount );
}

} // namespace TNL

