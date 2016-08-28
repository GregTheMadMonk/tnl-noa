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
#include <TNL/Experimental/Multimaps/EllpackIndexMultimapValues.h>

namespace TNL {

template< typename Index,
          typename Device >
EllpackIndexMultimap< Index, Device >::
EllpackIndexMultimap()
:  keysRange( 0 ), valuesRange( 0 ), valuesMaxCount( 0 )
{
}

template< typename Index,
          typename Device >
String EllpackIndexMultimap< Index, Device > :: getType()
{
   return String( "EllpackIndexMultimap< ") +
          Device :: getDeviceType() +
          String( ", " ) +
          String( TNL::getType< Index >() ) +
          String( " >" );
}

template< typename Index,
          typename Device >
String EllpackIndexMultimap< Index, Device >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Index,
          typename Device >
void
EllpackIndexMultimap< Index, Device >::
setRanges( const IndexType inputs,
               const IndexType outputs )
{
   this->keysRange = inputs;
   this->valuesRange = outputs;
}

template< typename Index,
          typename Device >
const Index
EllpackIndexMultimap< Index, Device >::
getKeysRange() const
{
   return this->keysRange;
}

template< typename Index,
          typename Device >
const Index
EllpackIndexMultimap< Index, Device >::
getValuesRange() const
{
   return this->valuesRange;
}

template< typename Index,
          typename Device >
void
EllpackIndexMultimap< Index, Device >::
allocate( const ValuesAllocationVectorType& portsCount )
{
   Assert( portsCount.getSize() == this->keysRange,
              std::cerr << "portsCount.getSize() =  " << portsCount.getSize()
                   << "this->inputs = " << this->keysRange );
   this->valuesMaxCount = portsCount.max();
 
   Assert( this->valuesMaxCount >= 0 && this->valuesMaxCount <= this->valuesRange,
              std::cerr << "this->portsMaxCount = " << this->valuesMaxCount
                   << " this->outputs = " << this->valuesRange );
   this->values.setSize( this->keysRange * this->valuesMaxCount );
}

template< typename Index,
          typename Device >
typename EllpackIndexMultimap< Index, Device >::ValuesAccessorType
EllpackIndexMultimap< Index, Device >::
getValues( const IndexType& inputIndex )
{
   return ValuesAccessorType( this->values.getData(), inputIndex, this->valuesMaxCount );
}

template< typename Index,
          typename Device >
typename EllpackIndexMultimap< Index, Device >::ConstValuesAccessorType
EllpackIndexMultimap< Index, Device >::
getValues( const IndexType& inputIndex ) const
{
   return ConstPortsType( this->values.getData(), inputIndex, this->valuesMaxCount );
}

} // namespace TNL

