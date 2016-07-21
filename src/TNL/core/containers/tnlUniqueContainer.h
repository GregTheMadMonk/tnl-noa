/***************************************************************************
                          tnlUniqueContainer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <map>
#include <TNL/Object.h>

namespace TNL {

/****
 * Unique container (map) - stores each object at most once.
 * It uses operator < to compare objects by their keys.
 */
template< typename Element, typename Key >
class tnlUniqueContainer : public Object
{
   typedef std::map< Key, Element >              MapType;
   typedef typename MapType::value_type          MapValueType;
   typedef typename MapType::const_iterator_type MapConstIteratorType;

   public:

   typedef Element ElementType;
   typedef Key KeyType;

   class ConstIterator
   {
      friend class UniqueContainer;

      public:

      const ElementType& operator*() const;

      const ElementType* operator->() const;

      ConstIterator& operator++();

      ConstIterator operator++( int );

      bool operator==( const ConstIterator& it ) const;

      bool operator!=( const ConstIterator& it ) const;

      protected:

      MapConstIteratorType constIterator;
   };

   ConstIterator insertElement( const ElementType& e );

   ConstIterator findElement( const ElementType& e ) const;

   size_t getSize() const;

   void reset();

   bool save( File& file ) const;

   bool load( File& file );

   protected:

   MapType map;
};

} // namespace TNL
