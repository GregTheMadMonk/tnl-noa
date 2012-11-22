/***************************************************************************
                          tnlHost.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLHOST_H_
#define TNLHOST_H_

#include <core/tnlDevice.h>
#include <core/tnlString.h>

class tnlHost
{
   public:

   static tnlString getDeviceType()
   {
      return tnlString( "tnlHost" );
   }

   static tnlDeviceEnum getDevice()
   {
      return tnlHostDevice;
   };

   template< typename Element, typename Index >
   static void allocateMemory( Element*& data, const Index size )
   {
      allocateMemoryHost( data, size );
   }

   template< typename Element >
   static void freeMemory( Element* data )
   {
      freeMemoryHost( data );
   }

   template< typename Element >
   static void setMemoryElement( Element* data,
                                 const Element& value )
   {
      *data = value;
   }

   template< typename Element >
   static Element getMemoryElement( Element* data )
   {
      return *data;
   }

   template< typename Element, typename Index >
   static Element& getArrayElementReference( Element* data, const Index i )
   {
      return data[ i ];
   }

   template< typename Element, typename Index >
   static const Element& getArrayElementReference(const Element* data, const Index i )
   {
      return data[ i ];
   }

   template< typename Element, typename Index, typename Device >
   static bool memcpy( Element* destination,
                       const Element* source,
                       const Index size )
   {
      switch( Device :: getDevice() )
      {
         case tnlHostDevice:
            return copyMemoryHostToHost( destination, source, size );
            break;
         case tnlCudaDevice:
            return copyMemoryCudaToHost( destination, source, size );
            break;
      }
      return true;
   }

   template< typename Element, typename Index, typename Device >
   static bool memcmp( const Element* data1,
                         const Element* data2,
                         const Index size )
   {
      switch( Device :: getDevice() )
      {
         case tnlHostDevice:
            return compareMemoryHost( data1, data2, size );
            break;
         case tnlCudaDevice:
            return compareMemoryHostCuda( data1, data2, size );
            break;
      }
   }

   template< typename Element, typename Index, typename Device >
   static bool memset( Element* destination,
                       const Element& value,
                       const Index size )
   {
      return setMemoryHost( destination, value, size );
   }

};


#endif /* TNLHOST_H_ */
