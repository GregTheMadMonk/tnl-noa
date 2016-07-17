/***************************************************************************
 tnlStack.h  -  description
 -------------------
 begin                : Nov 21, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
 email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/tnlList.h>

namespace TNL {

/*
 *
 */
template< class T > class tnlStack : protected tnlList< T >
{
   public:

   //! Push data
   bool Push( const T& data )
   {
      return tnlList< T > :: Append( data );
   };

   //! Pop data
   bool Pop( T& data )
   {
      if( tnlList< T > :: isEmpty() )
         return false;
      data = tnlList< T > :: operator[] ( tnlList< T > :: getSize() - 1 );
      tnlList< T > :: Erase( tnlList< T > :: getSize() - 1 );
   };

   //! Get stack size
   int GetSize()
   {
      return tnlList< T > :: getSize();
   }

   //! Check whether stack is empty
   bool isEmpty()
   {
      return tnlList< T > :: isEmpty();
   }
};

} // namespace TNL
