/***************************************************************************
                          StaticEllpackIndexMultimap_impl.h  -  description
                             -------------------
    begin                : Sep 9, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Experimental/Multimaps/StaticEllpackIndexMultimap.h>

namespace TNL {

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
StaticEllpackIndexMultimap()
: keysRange( 0 )
{
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
String
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
getType()
{
   return String( "StaticEllpackIndexMultimap< ") +
          String( TNL::getType< Index >() ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( ", " ) +
          String( TNL::getType< LocalIndexType >() ) +
          String( " >" );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
String
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
getTypeVirtual() const
{
   return this->getType();
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
setKeysRange( const IndexType& keysRange )
{
   Assert( keysRange >= 0, );
   this->keysRange = keysRange;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
const Index
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
getKeysRange() const
{
   return this->keysRange;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
bool
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
allocate()
{
   return this->values.setSize( this->keysRange * ValuesCount );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
typename StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::ValuesAccessorType
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
getValues( const IndexType& inputIndex )
{
   Assert( inputIndex < this->getKeysRange(), );
   return ValuesAccessorType( this->values.getData(), inputIndex );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
typename StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::ConstValuesAccessorType
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
getValues( const IndexType& inputIndex ) const
{
   Assert( inputIndex < this->getKeysRange(), );
   return ConstValuesAccessorType( this->values.getData(), inputIndex );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
bool
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
operator==( const StaticEllpackIndexMultimap& other ) const
{
   return ( this->keysRange == other.keysRange && this->values == other.values );
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
bool
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
save( File& file ) const
{
   if( ! Object::save( file ) )
      return false;
   if( ! file.write( &this->keysRange ) )
      return false;
   if( ! this->values.save( file ) )
      return false;
   return true;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
bool
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
load( File& file )
{
   if( ! Object::load( file ) )
      return false;
   if( ! file.read( &this->keysRange ) )
      return false;
   if( ! this->values.load( file ) )
      return false;
   return true;
}

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
void
StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >::
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
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >& multimap )
{
   multimap.print( str );
   return str;
}

} // namespace TNL

