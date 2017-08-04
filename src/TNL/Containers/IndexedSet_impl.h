/***************************************************************************
                          IndexedSet_impl.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/IndexedSet.h>

namespace TNL {
namespace Containers {

template< class Key,
          class Index,
          class Compare,
          class Allocator >
void
IndexedSet< Key, Index, Compare, Allocator >::clear()
{
   map.clear();
}

template< class Key,
          class Index,
          class Compare,
          class Allocator >
typename IndexedSet< Key, Index, Compare, Allocator >::size_type
IndexedSet< Key, Index, Compare, Allocator >::size() const
{
   return map.size();
}

template< class Key,
          class Index,
          class Compare,
          class Allocator >
Index
IndexedSet< Key, Index, Compare, Allocator >::insert( const Key& key )
{
   auto iter = map.insert( value_type( key, size() ) ).first;
   return iter->second;
}

template< class Key,
          class Index,
          class Compare,
          class Allocator >
bool
IndexedSet< Key, Index, Compare, Allocator >::find( const Key& key, Index& index ) const
{
   auto iter = map.find( Key( key ) );
   if( iter == map.end() )
      return false;
   index = iter->second.index;
   return true;
}

template< class Key,
          class Index,
          class Compare,
          class Allocator >
typename IndexedSet< Key, Index, Compare, Allocator >::size_type
IndexedSet< Key, Index, Compare, Allocator >::count( const Key& key ) const
{
   return map.count( key );
}

template< class Key,
          class Index,
          class Compare,
          class Allocator >
typename IndexedSet< Key, Index, Compare, Allocator >::size_type
IndexedSet< Key, Index, Compare, Allocator >::erase( const Key& key )
{
   return map.erase( key );
}

template< class Key,
          class Index,
          class Compare,
          class Allocator >
void IndexedSet< Key, Index, Compare, Allocator >::print( std::ostream& str ) const
{
   auto iter = map.begin();
   str << iter->second.data;
   iter++;
   while( iter != map.end() )
   {
      str << ", " << iter->second.data;
      iter++;
   }
}

template< class Key,
          class Index,
          class Compare,
          class Allocator >
std::ostream& operator<<( std::ostream& str, IndexedSet< Key, Index, Compare, Allocator >& set )
{
   set.print( str );
   return str;
}

} // namespace Containers
} // namespace TNL
