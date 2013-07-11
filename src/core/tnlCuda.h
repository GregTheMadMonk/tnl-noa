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

   static tnlString getDeviceType();

   static tnlDeviceEnum getDevice();

   template< typename Element, typename Index >
   static void allocateMemory( Element*& data, const Index size );

   template< typename Element >
   static void freeMemory( Element* data );


   template< typename Element >
   static void setMemoryElement( Element* data,
                                 const Element& value );

   template< typename Element >
   static Element getMemoryElement( const Element* data );

   template< typename Element, typename Index >
   static Element& getArrayElementReference( Element* data, const Index i );

   template< typename Element, typename Index >
   static const Element& getArrayElementReference(const Element* data, const Index i );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index,
             typename Device >
   static bool memcpy( DestinationElement* destination,
                       const SourceElement* source,
                       const Index size );

   template< typename Element, typename Index, typename Device >
   static bool memcpy( Element* destination,
                       const Element* source,
                       const Index size );

   template< typename Element, typename Index, typename Device >
   static bool memcmp( const Element* data1,
                       const Element* data2,
                       const Index size );

   template< typename Element, typename Index >
   static bool memset( Element* destination,
                       const Element& value,
                       const Index size );

   static int getMaxGridSize();

   static void setMaxGridSize( int newMaxGridSize );

   static int getMaxBlockSize();

   static void setMaxBlockSize( int newMaxBlockSize );

   protected:

   static int maxGridSize, maxBlockSize;
};

#include <implementation/core/tnlCuda_impl.h>

#endif /* TNLCUDA_H_ */
