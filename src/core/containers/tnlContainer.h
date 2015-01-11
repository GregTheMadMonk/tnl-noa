/***************************************************************************
                          tnlContainer.h  -  description
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

#ifndef TNLCONTAINER_H_
#define TNLCONTAINER_H_

#include<core/tnlObject.h>
#include<core/arrays/tnlArray.h>

template< typename Element, typename Device = tnlHost, typename Index = int >
class tnlContainer : public tnlObject
{
   public:

   typedef Element ElementType;
   typedef Index IndexType;

   tnlContainer();

   tnlContainer( const IndexType size );

   static tnlString getType();

   bool setSize( const IndexType size );

   IndexType getSize() const;

   void reset();

   ElementType& operator[]( const IndexType id );

   const ElementType& operator[]( const IndexType id ) const;

   ElementType getElement( const IndexType id ) const;

   void setElement( const IndexType id,
                    const ElementType& data );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   protected:

   tnlArray< Element, Device, Index > data;
};

#include <core/containers/tnlContainer_impl.h>

#endif /* TNLCONTAINER_H_ */
