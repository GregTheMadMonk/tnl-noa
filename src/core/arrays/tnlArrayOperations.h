/***************************************************************************
                          tnlArrayOperations.h  -  description
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
#include <core/tnlMIC.h>

template< typename DestinationDevice,
          typename SourceDevice = DestinationDevice >
class tnlArrayOperations{};

template<>
class tnlArrayOperations< tnlMIC >
{
    public:
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool copyMemory( DestinationElement* destination,
                           const SourceElement* source,
                           const Index size )
   {
       cout << "Never Will be implemented tnlArrayOperations-copyMemory on MIC" <<endl;
   };  
    
};

template<>
class tnlArrayOperations< tnlHost >
{
   public:

   template< typename Element, typename Index >
   static bool allocateMemory( Element*& data,
                               const Index size );

   template< typename Element >
   static bool freeMemory( Element* data );

   template< typename Element >
   static void setMemoryElement( Element* data,
                                 const Element& value );

   template< typename Element >
   static Element getMemoryElement( Element* data );

   template< typename Element, typename Index >
   static Element& getArrayElementReference( Element* data, const Index i );

   template< typename Element, typename Index >
   static const Element& getArrayElementReference( const Element* data, const Index i );


   template< typename Element, typename Index >
   static bool setMemory( Element* data,
                          const Element& value,
                          const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool copyMemory( DestinationElement* destination,
                           const SourceElement* source,
                           const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   static bool compareMemory( const Element1* destination,
                              const Element2* source,
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
   static bool freeMemory( Element* data );

   template< typename Element >
   static void setMemoryElement( Element* data,
                                 const Element& value );

   template< typename Element >
   static Element getMemoryElement( const Element* data );

   template< typename Element, typename Index >
   static Element& getArrayElementReference( Element* data, const Index i );

   template< typename Element, typename Index >
   static const Element& getArrayElementReference( const Element* data, const Index i );

   template< typename Element, typename Index >
   static bool setMemory( Element* data,
                          const Element& value,
                          const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool copyMemory( DestinationElement* destination,
                           const SourceElement* source,
                           const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   static bool compareMemory( const Element1* destination,
                              const Element2* source,
                              const Index size );
};

template<>
class tnlArrayOperations< tnlCuda, tnlHost >
{
   public:

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool copyMemory( DestinationElement* destination,
                           const SourceElement* source,
                           const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool compareMemory( const DestinationElement* destination,
                              const SourceElement* source,
                              const Index size );
};

template<>
class tnlArrayOperations< tnlHost, tnlCuda >
{
   public:

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool copyMemory( DestinationElement* destination,
                           const SourceElement* source,
                           const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   static bool compareMemory( const Element1* destination,
                              const Element2* source,
                              const Index size );
};

#include <core/arrays/tnlArrayOperationsHost_impl.h>
#include <core/arrays/tnlArrayOperationsCuda_impl.h>

#endif /* TNLARRAYOPERATIONS_H_ */
