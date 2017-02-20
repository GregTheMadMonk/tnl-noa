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
          typename LocalIndex,
          int SliceSize >
   template< typename Device_ >
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
EllpackIndexMultimap( const EllpackIndexMultimap< Index, Device_, LocalIndex, SliceSize >& other )
{
   operator=( other );
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
   template< typename Device_ >
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >&
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
operator=( const EllpackIndexMultimap< Index, Device_, LocalIndex, SliceSize >& other )
{
   values = other.values;
   valuesCounts = other.valuesCounts;
   keysRange = other.keysRange;
   maxValuesCount = other.maxValuesCount;
   return *this;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
String
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
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
          typename LocalIndex,
          int SliceSize >
String
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
void
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
setKeysRange( const IndexType& keysRange )
{
   TNL_ASSERT( keysRange >= 0, );
   this->keysRange = keysRange;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
const Index
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
getKeysRange() const
{
   return this->keysRange;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
bool
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
allocate( const LocalIndexType& maxValuesCount )
{
   TNL_ASSERT( maxValuesCount >= 0, );
   this->maxValuesCount = maxValuesCount;
   const IndexType ldSize = getAllocationKeysRange( this->getKeysRange() );
   if( ! this->values.setSize( ldSize * this->maxValuesCount ) )
      return false;
   if( ! this->valuesCounts.setSize( this->getKeysRange() ) )
      return false;
   this->valuesCounts.setValue( maxValuesCount );

   // extra cost at initialization, which allows to have much simpler operator==
   values.setValue( 0 );

   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
bool
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
allocate( const ValuesAllocationVectorType& valuesCounts )
{
   TNL_ASSERT( valuesCounts.getSize() == this->keysRange,
               std::cerr << "valuesCounts.getSize() =  " << valuesCounts.getSize()
                         << "this->keysRange = " << this->keysRange
                         << std::endl; );
   this->maxValuesCount = valuesCounts.max();
 
   TNL_ASSERT( this->maxValuesCount >= 0,
               std::cerr << "this->maxValuesCount = " << this->maxValuesCount << std::endl; );
   const IndexType ldSize = getAllocationKeysRange( this->getKeysRange() );
   if( ! this->values.setSize( ldSize * this->maxValuesCount ) )
      return false;
   if( ! this->valuesCounts.setSize( this->getKeysRange() ) )
      return false;
   this->valuesCounts = valuesCounts;

   // extra cost at initialization, which allows to have much simpler operator==
   values.setValue( 0 );

   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
   template< typename Device_, int SliceSize_ >
bool
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
setLike( const EllpackIndexMultimap< Index, Device_, LocalIndex, SliceSize_ >& other )
{
   const IndexType ldSize = getAllocationKeysRange( other.getKeysRange() );
   if( ! values.setSize( ldSize * other.maxValuesCount ) )
      return false;
   if( ! valuesCounts.setLike( other.valuesCounts ) )
      return false;
   keysRange = other.keysRange;
   maxValuesCount = other.maxValuesCount;

   // extra cost at initialization, which allows to have much simpler operator==
   values.setValue( 0 );

   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
typename EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::ValuesAccessorType
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
getValues( const IndexType& inputIndex )
{
   TNL_ASSERT( inputIndex < this->getKeysRange(),
               std::cerr << "inputIndex = " << inputIndex << std::endl
                         << "this->getKeysRange() = " << this->getKeysRange()
                         << std::endl; );
   TNL_ASSERT( getAllocationKeysRange( this->getKeysRange() ) * this->maxValuesCount == this->values.getSize() && this->getKeysRange() == this->valuesCounts.getSize(),
               std::cerr << "The map has not been reallocated after calling setKeysRange()." << std::endl
                         << "this->getKeysRange() = " << this->getKeysRange() << std::endl
                         << "this->maxValuesCount = " << this->maxValuesCount << std::endl
                         << "this->values.getSize() = " << this->values.getSize() << std::endl
                         << "this->valuesCounts.getSize() = " << this->valuesCounts.getSize() << std::endl; );
   const IndexType sliceIdx = inputIndex / SliceSize;
   const IndexType sliceOffset = sliceIdx * SliceSize * this->maxValuesCount;
   const IndexType offset = sliceOffset + inputIndex - sliceIdx * SliceSize;
   return ValuesAccessorType( &this->values[ offset ], &this->valuesCounts[ inputIndex ], this->maxValuesCount );
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
typename EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::ConstValuesAccessorType
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
getValues( const IndexType& inputIndex ) const
{
   TNL_ASSERT( inputIndex < this->getKeysRange(),
              std::cerr << "inputIndex = " << inputIndex << std::endl
                        << "this->getKeysRange() = " << this->getKeysRange()
                        << std::endl; );
   TNL_ASSERT( getAllocationKeysRange( this->getKeysRange() ) * this->maxValuesCount == this->values.getSize() && this->getKeysRange() == this->valuesCounts.getSize(),
              std::cerr << "The map has not been reallocated after calling setKeysRange()." << std::endl
                        << "this->getKeysRange() = " << this->getKeysRange() << std::endl
                        << "this->maxValuesCount = " << this->maxValuesCount << std::endl
                        << "this->values.getSize() = " << this->values.getSize() << std::endl
                        << "this->valuesCounts.getSize() = " << this->valuesCounts.getSize() << std::endl; );
   const IndexType sliceIdx = inputIndex / SliceSize;
   const IndexType sliceOffset = sliceIdx * SliceSize * this->maxValuesCount;
   const IndexType offset = sliceOffset + inputIndex - sliceIdx * SliceSize;
   return ConstValuesAccessorType( &this->values[ offset ], &this->valuesCounts[ inputIndex ], this->maxValuesCount );
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
bool
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
operator==( const EllpackIndexMultimap& other ) const
{
//   if( ! ( this->keysRange == other.keysRange &&
//           this->maxValuesCount == other.maxValuesCount &&
//           this->valuesCounts == other.valuesCounts ) )
//      return false;
//   // compare values for each key separately - the sizes may vary
//   for( IndexType i = 0; i < this->keysRange; i++ )
//      if( this->getValues( i ) != other.getValues( i ) )
//         return false;
//   return true;

   // we assume that invalid entries in the ellpack format are always 0
   return this->keysRange == other.keysRange &&
          this->maxValuesCount == other.maxValuesCount &&
          this->valuesCounts == other.valuesCounts &&
          this->values == other.values;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
bool
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
save( File& file ) const
{
   if( ! Object::save( file ) )
      return false;
   if( ! file.write( &this->keysRange ) )
      return false;
   if( ! file.write( &this->maxValuesCount ) )
      return false;
   if( ! this->values.save( file ) )
      return false;
   if( ! this->valuesCounts.save( file ) )
      return false;
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
bool
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
load( File& file )
{
   if( ! Object::load( file ) )
      return false;
   if( ! file.read( &this->keysRange ) )
      return false;
   if( ! file.read( &this->maxValuesCount ) )
      return false;
   if( ! this->values.load( file ) )
      return false;
   if( ! this->valuesCounts.load( file ) )
      return false;
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
void
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
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

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
__cuda_callable__
Index
EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >::
getAllocationKeysRange( IndexType keysRange ) const
{
   return SliceSize * roundUpDivision( keysRange, SliceSize );
}


template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >& multimap )
{
   multimap.print( str );
   return str;
}

} // namespace TNL

