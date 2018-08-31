/***************************************************************************
                          IndexedSet.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <map>
#include <ostream>

namespace TNL {
namespace Containers {

template< class Key,
          class Index,
          class Compare = std::less< Key >,
          class Allocator = std::allocator< std::pair< const Key, Index > > >
class IndexedSet
{
protected:
   using map_type = std::map< Key, Index, Compare, Allocator >;
   map_type map;

public:
   using key_type = Key;
   using index_type = Index;
   using value_type = typename map_type::value_type;
   using size_type = typename map_type::size_type;

   void clear();

   size_type size() const;

   Index insert( const Key& key );

   bool find( const Key& key, Index& index ) const;

   size_type count( const Key& key ) const;

   size_type erase( const Key& key );

   void print( std::ostream& str ) const;
};

template< typename Element,
          typename Index >
std::ostream& operator <<( std::ostream& str, IndexedSet< Element, Index >& set );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/IndexedSet_impl.h>
