/***************************************************************************
                          tnlContainer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Containers/Array.h>

namespace TNL {

template< typename Element, typename Device = Devices::Host, typename Index = int >
class tnlContainer : public Object
{
   public:

   typedef Element ElementType;
   typedef Index IndexType;

   tnlContainer();

   tnlContainer( const IndexType size );

   static String getType();

   bool setSize( const IndexType size );

   IndexType getSize() const;

   void reset();

   ElementType& operator[]( const IndexType id );

   const ElementType& operator[]( const IndexType id ) const;

   ElementType getElement( const IndexType id ) const;

   void setElement( const IndexType id,
                    const ElementType& data );

   bool save( File& file ) const;

   bool load( File& file );

   protected:

   Containers::Array< Element, Device, Index > data;
};

} // namespace TNL

#include <TNL/core/containers/tnlContainer_impl.h>
