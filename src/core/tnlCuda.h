/***************************************************************************
                          tnlCuda.h  -  description
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

#ifndef TNLCUDA_H_
#define TNLCUDA_H_

#include <core/tnlDevice.h>
#include <core/tnlString.h>
#include <core/tnlAssert.h>
#include <implementation/core/memory-operations.h>

class tnlCuda
{
   public:

   static tnlString getDeviceType()
   {
      return tnlString( "tnlCuda" );
   }

   static tnlDeviceEnum getDevice()
   {
      return tnlCudaDevice;
   };

   template< typename Element, typename Index >
   static void allocateMemory( Element*& data, const Index size )
   {
      allocateMemoryCuda( data, size );
   }

   template< typename Element >
   static void freeMemory( Element* data )
   {
      freeMemoryCuda( data );
   }


   template< typename Element >
   static void setMemoryElement( Element* data,
                                 const Element& value )
   {
      setMemoryCuda( data, value, 1 );
   }

   template< typename Element >
   static Element getMemoryElement( const Element* data )
   {
      Element result;
      copyMemoryCudaToHost( &result, data, 1 );
      return result;
   }

   template< typename Element, typename Index >
   static Element& getArrayElementReference( Element* data, const Index i )
   {
      tnlAssert( false, );
      abort();
   }

   template< typename Element, typename Index >
   static const Element& getArrayElementReference(const Element* data, const Index i )
   {
      tnlAssert( false, );
      abort();
   }

   template< typename DestinationElement,
             typename SourceElement,
             typename Index,
             typename Device >
   static bool memcpy( DestinationElement* destination,
                       const SourceElement* source,
                       const Index size )
   {
      switch( Device :: getDevice() )
      {
         case tnlHostDevice:
            return copyMemoryHostToCuda( destination, source, size );
         case tnlCudaDevice:
            return copyMemoryCudaToCuda( destination, source, size );
      }
      return true;
   }


   template< typename Element, typename Index, typename Device >
   static bool memcpy( Element* destination,
                       const Element* source,
                       const Index size )
   {
      return tnlCuda :: memcpy< Element, Element, Index, Device >
                              ( destination,
                                source,
                                size );
   }

   template< typename Element, typename Index, typename Device >
   static bool memcmp( const Element* data1,
                       const Element* data2,
                       const Index size )
   {
      switch( Device :: getDevice() )
      {
         case tnlHostDevice:
            return compareMemoryHostCuda( data2, data1, size );
         case tnlCudaDevice:
            return compareMemoryCuda( data1, data2, size );
      }
   }

   template< typename Element, typename Index >
   static bool memset( Element* destination,
                       const Element& value,
                       const Index size )
   {
      return setMemoryCuda( destination, value, size );
   }
};


#endif /* TNLCUDA_H_ */
