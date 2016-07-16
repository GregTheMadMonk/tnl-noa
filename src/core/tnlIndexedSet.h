/***************************************************************************
                          tnlIndexedSet.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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

   const Element& getElement( KeyType key ) const;

   Element& getElement( KeyType key );
 
   void print( ostream& str ) const;

   protected:

   struct DataWithIndex
   {
      // This constructor is here only because of bug in g++, we might fix it later.
      // http://stackoverflow.com/questions/22357887/comparing-two-mapiterators-why-does-it-need-the-copy-constructor-of-stdpair
      DataWithIndex(){};
 
      DataWithIndex( const DataWithIndex& d ) : data( d.data ), index( d.index) {}
 
      explicit DataWithIndex( const Element data) : data( data ) {}

      DataWithIndex( const Element data,
                     const Index index) : data(data), index(index) {}

      Element data;
      Index index;
   };

   typedef std::map< Key, DataWithIndex >      STDMapType;
   typedef typename STDMapType::value_type     STDMapValueType;
   typedef typename STDMapType::const_iterator STDMapIteratorType;

   STDMapType map;

};

template< typename Element,
          typename Index,
          typename Key >
ostream& operator <<( ostream& str, tnlIndexedSet< Element, Index, Key >& set );

#include <core/tnlIndexedSet_impl.h>

#endif /* TNLINDEXEDSET_H_ */
