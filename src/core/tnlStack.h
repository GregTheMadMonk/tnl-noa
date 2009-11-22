/***************************************************************************
 tnlStack.h  -  description
 -------------------
 begin                : Nov 21, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLSTACK_H_
#define TNLSTACK_H_

#include <core/tnlList.h>

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
      if( tnlList< T > :: IsEmpty() )
         return false;
      data = tnlList< T > :: operator[] ( tnlList< T > :: Size() - 1 );
      tnlList< T > :: Erase( tnlList< T > :: Size() - 1 );
   };

   //! Get stack size
   int GetSize()
   {
      return tnlList< T > :: Size();
   }

   //! Check whether stack is empty
   bool IsEmpty()
   {
      return tnlList< T > :: IsEmpty();
   }
};

#endif /* TNLSTACK_H_ */
