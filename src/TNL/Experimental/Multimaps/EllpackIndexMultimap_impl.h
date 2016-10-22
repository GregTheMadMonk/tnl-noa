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
: keysRange( 0 ), maxValuesCount( 0 )
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
setKeysRange( const IndexType& keysRange )
{
   Assert( keysRange >= 0, );
   this->keysRange = keysRange;
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
bool
EllpackIndexMultimap< Index, Device, LocalIndex >::
allocate( const LocalIndexType& maxValuesCount )
{
   Assert( maxValuesCount >= 0, );
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
 
   Assert( this->maxValuesCount >= 0,
              std::cerr << "this->maxValuesCount = " << this->maxValuesCount << std::endl; );
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

template< typename Index,
          typename Device,
          typename LocalIndex >
bool
EllpackIndexMultimap< Index, Device, LocalIndex >::
operator==( const EllpackIndexMultimap< Index, Device, LocalIndex >& other ) const
{
   return ( this->keysRange == other.keysRange &&
            this->maxValuesCount == other.maxValuesCount &&
            this->values == other.values );
}

template< typename Index,
          typename Device,
          typename LocalIndex >
bool
EllpackIndexMultimap< Index, Device, LocalIndex >::
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
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
bool
EllpackIndexMultimap< Index, Device, LocalIndex >::
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
   return true;
}

template< typename Index,
          typename Device,
          typename LocalIndex >
void
EllpackIndexMultimap< Index, Device, LocalIndex >::
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
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimap< Index, Device, LocalIndex >& multimap )
{
   multimap.print( str );
   return str;
}

} // namespace TNL

