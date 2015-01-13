/***************************************************************************
                          tnlUniqueContainer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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

#ifndef TNLUNIQUECONTAINER_H_
#define TNLUNIQUECONTAINER_H_

#include <map>
#include <core/tnlObject.h>

/****
 * Unique container (map) - stores each object at most once.
 * It uses operator < to compare objects by their keys.
 */
template< typename Element, typename Key >
class tnlUniqueContainer : public tnlObject
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

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   MapType map;
};

#endif /* TNLUNIQUECONTAINER_H_ */
