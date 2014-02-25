/***************************************************************************
                          tnlIndexedSet.h  -  description
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

#ifndef TNLINDEXEDSET_H_
#define TNLINDEXEDSET_H_

#include <map>
#include <stdexcept>


template< typename Element,
          typename Index,
          typename Key >
class tnlIndexedSet
{
   public:

   typedef Element   ElementType;
   typedef Index     IndexType;
   typedef Key       KeyType;

   void reset();

   IndexType getSize() const;

   IndexType insert( const ElementType &data );

   bool find( const ElementType &data, IndexType& index ) const;

   template< typename ArrayType >
   void toArray( ArrayType& array ) const;

   protected:

   struct DataWithIndex
   {
      explicit DataWithIndex( const Element data) : data( data ) {}

      DataWithIndex( const Element data,
                     const Index index) : data(data), index(index) {}

      Element data;
      Index index;
   };

   typedef std::map<Key, DataWithIndex>        STDMapType;
   typedef typename STDMapType::value_type     STDMapValueType;
   typedef typename STDMapType::const_iterator STDMapIteratorType;

   STDMapType map;

};

#include <implementation/core/tnlIndexedSet_impl.h>

#endif /* TNLINDEXEDSET_H_ */
