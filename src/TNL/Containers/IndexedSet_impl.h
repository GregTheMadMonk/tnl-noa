/***************************************************************************
                          IndexedSet_impl.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {

template< typename Element,
          typename Index,
          typename Key >
void IndexedSet< Element, Index, Key >::reset()
{
   map.clear();
}

template< typename Element,
          typename Index,
          typename Key >
Index IndexedSet< Element, Index, Key >::getSize() const
{
   return map.size();
}

template< typename Element,
          typename Index,
          typename Key >
Index IndexedSet< Element, Index, Key >::insert( const Element &data )
{
   STDMapIteratorType iter = map.insert( STDMapValueType( Key( data ),
                                         DataWithIndex( data, getSize() ) ) ).first;
   return iter->second.index;
}

template< typename Element,
          typename Index,
          typename Key >
bool IndexedSet< Element, Index, Key >::find( const Element &data, Index& index ) const
{
   STDMapIteratorType iter = map.find( Key( data ) );
   if (iter == map.end())
      return false;
   index = iter->second.index;
   return true;
}

template< typename Element,
          typename Index,
          typename Key >
   template<typename ArrayType>
void IndexedSet< Element, Index, Key >::toArray( ArrayType& array ) const
{
   Assert( array.getSize() == getSize(),
              std::cerr << "array.getSize() = " << array.getSize()
                   << " getSize() = " << getSize() );

   for( STDMapIteratorType iter = map.begin();
        iter != map.end();
        ++iter)
      array[ iter->second.index ] = iter->second.data;
}

template< typename Element,
          typename Index,
          typename Key >
const Element& IndexedSet< Element, Index, Key >::getElement( KeyType key ) const
{
   return map[ key ];
}

template< typename Element,
          typename Index,
          typename Key >
Element& IndexedSet< Element, Index, Key >::getElement( KeyType key )
{
   return map[ key ];
}

template< typename Element,
          typename Index,
          typename Key >
void IndexedSet< Element, Index, Key >::print( std::ostream& str ) const
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

template< typename Element,
          typename Index,
          typename Key >
std::ostream& operator<<( std::ostream& str, IndexedSet< Element, Index, Key >& set )
{
   set.print( str );
   return str;
}

} // namespace Containers
} // namespace TNL