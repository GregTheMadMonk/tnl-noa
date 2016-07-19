/***************************************************************************
                          tnlStaticStaticContainer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlObject.h>
#include <TNL/core/arrays/tnlStaticArray.h>

namespace TNL {

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

} // namespace TNL

#include <TNL/core/containers/tnlStaticContainer_impl.h>

