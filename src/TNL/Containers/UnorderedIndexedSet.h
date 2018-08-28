/***************************************************************************
                          UnorderedIndexedSet.h  -  description
                             -------------------
    begin                : Dec 12, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <unordered_map>
#include <ostream>

namespace TNL {
namespace Containers {

template< class Key,
          class Index,
          class Hash = std::hash< Key >,
          class KeyEqual = std::equal_to< Key >,
          class Allocator = std::allocator< std::pair<const Key, Index> > >
class UnorderedIndexedSet
{
protected:
   using map_type = std::unordered_map< Key, Index, Hash, KeyEqual, Allocator >;
   map_type map;

public:
   using key_type = Key;
   using index_type = Index;
   using value_type = typename map_type::value_type;
   using size_type = typename map_type::size_type;
   using hasher = Hash;
   using key_equal = KeyEqual;

   void clear();

   size_type size() const;

   Index insert( const Key& key );

   std::pair< Index, bool > try_insert( const Key& key );

   bool find( const Key& key, Index& index ) const;

   size_type count( const Key& key ) const;

   size_type erase( const Key& key );

   void print( std::ostream& str ) const;
};

template< typename Element,
          typename Index >
std::ostream& operator <<( std::ostream& str, UnorderedIndexedSet< Element, Index >& set );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/UnorderedIndexedSet_impl.h>
