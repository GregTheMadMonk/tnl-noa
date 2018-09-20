/***************************************************************************
                          IndexedMap_impl.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "IndexedMap.h"

namespace TNL {
namespace Containers {

template< typename Value,
          typename Index,
          typename Key >
void IndexedMap< Value, Index, Key >::reset()
{
   map.clear();
}

template< typename Value,
          typename Index,
          typename Key >
Index IndexedMap< Value, Index, Key >::getSize() const
{
   return map.size();
}

template< typename Value,
          typename Index,
          typename Key >
Index IndexedMap< Value, Index, Key >::insert( const Value &data )
{
   STDMapIteratorType iter = map.insert( STDMapValueType( Key( data ),
                                         DataWithIndex( data, getSize() ) ) ).first;
   return iter->second.index;
}

template< typename Value,
          typename Index,
          typename Key >
bool IndexedMap< Value, Index, Key >::find( const Value &data, Index& index ) const
{
   STDMapIteratorType iter = map.find( Key( data ) );
   if (iter == map.end())
      return false;
   index = iter->second.index;
   return true;
}

template< typename Value,
          typename Index,
          typename Key >
   template<typename ArrayType>
void IndexedMap< Value, Index, Key >::toArray( ArrayType& array ) const
{
   TNL_ASSERT( array.getSize() == getSize(),
               std::cerr << "array.getSize() = " << array.getSize()
                         << " getSize() = " << getSize() );

   for( STDMapIteratorType iter = map.begin();
        iter != map.end();
        ++iter)
      array[ iter->second.index ] = iter->second.data;
}

template< typename Value,
          typename Index,
          typename Key >
const Value& IndexedMap< Value, Index, Key >::getElement( KeyType key ) const
{
   return map[ key ];
}

template< typename Value,
          typename Index,
          typename Key >
Value& IndexedMap< Value, Index, Key >::getElement( KeyType key )
{
   return map[ key ];
}

template< typename Value,
          typename Index,
          typename Key >
void IndexedMap< Value, Index, Key >::print( std::ostream& str ) const
{
   STDMapIteratorType iter = map.begin();
   str << iter->second.data;
   iter++;
   while( iter != map.end() )
   {
      str << ", " << iter->second.data;
      iter++;
   }
}

template< typename Value,
          typename Index,
          typename Key >
std::ostream& operator<<( std::ostream& str, IndexedMap< Value, Index, Key >& set )
{
   set.print( str );
   return str;
}

} // namespace Containers
} // namespace TNL
