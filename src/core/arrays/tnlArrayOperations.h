/***************************************************************************
                          tnlArrayOperationsTest.cpp  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLARRAYOPERATIONS_H_
#define TNLARRAYOPERATIONS_H_

#include <core/tnlHost.h>
#include <core/tnlCuda.h>

template< typename Device >
class tnlArrayOperations{};

template<>
class tnlArrayOperations< tnlHost >
{
   public:

   template< typename Element, typename Index >
   static bool allocateMemory( Element*& data,
                               const Index size );

   template< typename Element >
   bool freeMemory( Element* data );

   template< typename Element, typename Index >
   bool setMemory( Element* data,
                       const Element& value,
                       const Index size );

   template< typename DestinationElement, typename SourceElement, typename Index >
   bool copyMemoryToHost( DestinationElement* destination,
                          const SourceElement* source,
                          const Index size );

   template< typename Element, typename Index >
   bool copyMemoryToHost( Element* destination,
                              const Element* source,
                              const Index size );

   template< typename DestinationElement, typename SourceElement, typename Index >
   bool copyMemoryToCuda( DestinationElement* destination,
                          const SourceElement* source,
                          const Index size );

   template< typename Element, typename Index >
   bool copyMemoryToCuda( Element* destination,
                              const Element* source,
                              const Index size );

   template< typename Element,
             typename Index >
   bool compareMemoryOnHost( const Element* data1,
                           const Element* data2,
                           const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   bool compareMemoryOnHost( const Element1* data1,
                           const Element2* data2,
                           const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   bool compareMemoryOnCuda( const Element1* hostData,
                               const Element2* deviceData,
                               const Index size );
   template< typename Element,
             typename Index >
   bool compareMemoryOnCuda( const Element* deviceData1,
                             const Element* deviceData2,
                             const Index size );
};

template<>
class tnlArrayOperations< tnlCuda >
{
   public:

   template< typename Element, typename Index >
   static bool allocateMemory( Element*& data,
                               const Index size );

   template< typename Element >
   bool freeMemory( Element* data );

   template< typename Element, typename Index >
   bool setMemory( Element* data,
                       const Element& value,
                       const Index size );

   template< typename DestinationElement, typename SourceElement, typename Index >
   bool copyMemoryToHost( DestinationElement* destination,
                          const SourceElement* source,
                          const Index size );

   template< typename Element, typename Index >
   bool copyMemoryToHost( Element* destination,
                              const Element* source,
                              const Index size );

   template< typename DestinationElement, typename SourceElement, typename Index >
   bool copyMemoryToCuda( DestinationElement* destination,
                          const SourceElement* source,
                          const Index size );

   template< typename Element, typename Index >
   bool copyMemoryToCuda( Element* destination,
                              const Element* source,
                              const Index size );

   template< typename Element,
             typename Index >
   bool compareMemoryOnHost( const Element* data1,
                           const Element* data2,
                           const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   bool compareMemoryOnHost( const Element1* data1,
                           const Element2* data2,
                           const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   bool compareMemoryOnCuda( const Element1* hostData,
                               const Element2* deviceData,
                               const Index size );
   template< typename Element,
             typename Index >
   bool compareMemoryOnCuda( const Element* deviceData1,
                             const Element* deviceData2,
                             const Index size );
};


#endif /* TNLARRAYOPERATIONS_H_ */
