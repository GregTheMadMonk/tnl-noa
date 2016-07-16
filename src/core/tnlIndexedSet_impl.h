/***************************************************************************
                          tnlIndexedSet_impl.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLINDEXEDSET_IMPL_H_
#define TNLINDEXEDSET_IMPL_H_

template< typename Element,
          typename Index,
          typename Key >
void tnlIndexedSet< Element, Index, Key >::reset()
{
   map.clear();
}

template< typename Element,
          typename Index,
          typename Key >
Index tnlIndexedSet< Element, Index, Key >::getSize() const
{
   return map.size();
}

template< typename Element,
          typename Index,
          typename Key >
Index tnlIndexedSet< Element, Index, Key >::insert( const Element &data )
{
   STDMapIteratorType iter = map.insert( STDMapValueType( Key( data ),
                                         DataWithIndex( data, getSize() ) ) ).first;
   return iter->second.index;
}

template< typename Element,
          typename Index,
          typename Key >
bool tnlIndexedSet< Element, Index, Key >::find( const Element &data, Index& index ) const
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
void tnlIndexedSet< Element, Index, Key >::toArray( ArrayType& array ) const
{
   tnlAssert( array.getSize() == getSize(),
              cerr << "array.getSize() = " << array.getSize()
                   << " getSize() = " << getSize() );

   for( STDMapIteratorType iter = map.begin();
        iter != map.end();
        ++iter)
      array[ iter->second.index ] = iter->second.data;
}

template< typename Element,
          typename Index,
          typename Key >
const Element& tnlIndexedSet< Element, Index, Key >::getElement( KeyType key ) const
{
   return map[ key ];
}

template< typename Element,
          typename Index,
          typename Key >
Element& tnlIndexedSet< Element, Index, Key >::getElement( KeyType key )
{
   return map[ key ];
}

template< typename Element,
          typename Index,
          typename Key >
void tnlIndexedSet< Element, Index, Key >::print( ostream& str ) const
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
ostream& operator<<( ostream& str, tnlIndexedSet< Element, Index, Key >& set )
{
   set.print( str );
   return str;
}

#endif /* TNLINDEXEDSET_IMPL_H_ */
