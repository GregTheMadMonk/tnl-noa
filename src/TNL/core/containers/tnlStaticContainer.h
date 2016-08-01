/***************************************************************************
                          tnlStaticStaticContainer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Arrays/StaticArray.h>

namespace TNL {

template< int Size, typename Element >
class tnlStaticContainer : public Object
{
   public:

   typedef Element ElementType;
   enum { size = Size };

   tnlStaticContainer();

   static String getType();

   int getSize() const;

   void reset();

   ElementType& operator[]( const int id );

   const ElementType& operator[]( const int id ) const;

   ElementType getElement( const int id ) const;

   void setElement( const int id,
                    const ElementType& data );

   bool save( File& file ) const;

   bool load( File& file );

   protected:

   Arrays::StaticArray< Size, Element > data;
};

} // namespace TNL

#include <TNL/core/containers/tnlStaticContainer_impl.h>

