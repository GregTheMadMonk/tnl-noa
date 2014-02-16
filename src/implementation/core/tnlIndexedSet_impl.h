/***************************************************************************
                          tnlIndexedSet_impl.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLINDEXEDSET_IMPL_H_
#define TNLINDEXEDSET_IMPL_H_

template< typename Element,
          typename Index,
          typename Key >
void tnlIndexedSet< Element, Index, Key >::void reset()
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
void tnlIndexedSet< Element, Index, Key >::toArray( ArrayType &array ) const
{
   tnlAssert( array.getSize() == getSize(),
              cerr << "array.getSize() = " << array.getSize()
                   << " getSize() = " << getSize() );

   for( STDMapIteratorType iter = map.begin();
        iter != map.end();
        ++iter)
      array[ iter->second.index ] = iter->second.data;
}



#endif /* TNLINDEXEDSET_IMPL_H_ */
