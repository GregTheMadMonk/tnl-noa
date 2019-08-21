/***************************************************************************
                          StaticEllpackIndexMultimap_impl.h  -  description
                             -------------------
    begin                : Sep 9, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Containers/Multimaps/StaticEllpackIndexMultimap.h>

namespace TNL {
namespace Containers {
namespace Multimaps {

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
   template< typename Device_ >
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
StaticEllpackIndexMultimap( const StaticEllpackIndexMultimap< ValuesCount, Index, Device_, LocalIndex, SliceSize >& other )
{
   operator=( other );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
   template< typename Device_ >
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >&
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
operator=( const StaticEllpackIndexMultimap< ValuesCount, Index, Device_, LocalIndex, SliceSize >& other )
{
   values = other.values;
   keysRange = other.keysRange;
   return *this;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
setKeysRange( const IndexType& keysRange )
{
   TNL_ASSERT( keysRange >= 0, );
   this->keysRange = keysRange;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
const Index
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
getKeysRange() const
{
   return this->keysRange;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
allocate()
{
   const IndexType ldSize = getAllocationKeysRange( this->getKeysRange() );
   values.setSize( ldSize * ValuesCount );

   // extra cost at initialization, which allows to have much simpler operator==
   values.setValue( 0 );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
   template< typename Device_ >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
setLike( const StaticEllpackIndexMultimap< ValuesCount, Index, Device_, LocalIndex, SliceSize >& other )
{
   const IndexType ldSize = getAllocationKeysRange( other.getKeysRange() );
   values.setSize( ldSize * ValuesCount );

   // extra cost at initialization, which allows to have much simpler operator==
   values.setValue( 0 );

   keysRange = other.keysRange;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
typename StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::ValuesAccessorType
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
getValues( const IndexType& inputIndex )
{
   TNL_ASSERT( inputIndex < this->getKeysRange(),
               std::cerr << "inputIndex = " << inputIndex << std::endl
                         << "this->getKeysRange() = " << this->getKeysRange() << std::endl; );
   TNL_ASSERT( getAllocationKeysRange( this->getKeysRange() ) * ValuesCount == this->values.getSize(),
               std::cerr << "The map has not been reallocated after calling setKeysRange()." << std::endl
                         << "this->getKeysRange() = " << this->getKeysRange() << std::endl
                         << "this->values.getSize() = " << this->values.getSize() << std::endl; );
   const IndexType sliceIdx = inputIndex / SliceSize;
   const IndexType sliceOffset = sliceIdx * SliceSize * ValuesCount;
   const IndexType offset = sliceOffset + inputIndex - sliceIdx * SliceSize;
   return ValuesAccessorType( &this->values[ offset ] );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
typename StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::ConstValuesAccessorType
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
getValues( const IndexType& inputIndex ) const
{
   TNL_ASSERT( inputIndex < this->getKeysRange(),
               std::cerr << "inputIndex = " << inputIndex << std::endl
                         << "this->getKeysRange() = " << this->getKeysRange() << std::endl; );
   TNL_ASSERT( getAllocationKeysRange( this->getKeysRange() ) * ValuesCount == this->values.getSize(),
               std::cerr << "The map has not been reallocated after calling setKeysRange()." << std::endl
                         << "this->getKeysRange() = " << this->getKeysRange() << std::endl
                         << "this->values.getSize() = " << this->values.getSize() << std::endl; );
   const IndexType sliceIdx = inputIndex / SliceSize;
   const IndexType sliceOffset = sliceIdx * SliceSize * ValuesCount;
   const IndexType offset = sliceOffset + inputIndex - sliceIdx * SliceSize;
   return ConstValuesAccessorType( &this->values[ offset ] );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
bool
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
operator==( const StaticEllpackIndexMultimap& other ) const
{
   return ( this->keysRange == other.keysRange && this->values == other.values );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
save( File& file ) const
{
   Object::save( file );
   file.save( &this->keysRange );
   file << this->values;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
load( File& file )
{
   Object::load( file );
   file.load( &this->keysRange );
   file >> this->values;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
print( std::ostream& str ) const
{
   str << "[ ";
   if( this->getKeysRange() > 0 )
   {
      str << this->getValues( 0 );
      for( Index i = 1; i < this->getKeysRange(); i++ )
         str << ",\n  " << this->getValues( i );
   }
   str << " ]";
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
Index
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >::
getAllocationKeysRange( IndexType keysRange ) const
{
   return SliceSize * roundUpDivision( keysRange, SliceSize );
}


template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
std::ostream& operator << ( std::ostream& str, const StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >& multimap )
{
   multimap.print( str );
   return str;
}

} // namespace Multimaps
} // namespace Containers
} // namespace TNL
