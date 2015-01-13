/***************************************************************************
                          tnlStaticStaticContainer.h  -  description
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

#ifndef TNLSTATICCONTAINER_H_
#define TNLSTATICCONTAINER_H_

#include <core/tnlObject.h>
#include <core/arrays/tnlStaticArray.h>

template< int Size, typename Element >
class tnlStaticContainer : public tnlObject
{
   public:

   typedef Element ElementType;
   enum { size = Size };

   tnlStaticContainer();

   static tnlString getType();

   int getSize() const;

   void reset();

   ElementType& operator[]( const int id );

   const ElementType& operator[]( const int id ) const;

   ElementType getElement( const int id ) const;

   void setElement( const int id,
                    const ElementType& data );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   tnlStaticArray< Size, Element > data;
};

#include <core/containers/tnlStaticContainer_impl.h>


#endif /* TNLSTATICCONTAINER_H_ */
