/***************************************************************************
                          ArrayOperations.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/MIC.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename DestinationDevice,
          typename SourceDevice = DestinationDevice >
struct ArrayOperations;

template<>
struct ArrayOperations< Devices::Host >
{
   template< typename Element >
   static void setElement( Element* data,
                           const Element& value );

   template< typename Element >
   static Element getElement( const Element* data );

   template< typename Element, typename Index >
   static void set( Element* data,
                    const Element& value,
                    const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size );

   template< typename DestinationElement,
             typename Index,
             typename SourceIterator >
   static void copyFromIterator( DestinationElement* destination,
                                 Index destinationSize,
                                 SourceIterator first,
                                 SourceIterator last );

   template< typename Element1,
             typename Element2,
             typename Index >
   static bool compare( const Element1* destination,
                        const Element2* source,
                        const Index size );

   template< typename Element,
             typename Index >
   static bool containsValue( const Element* data,
                              const Index size,
                              const Element& value );

   template< typename Element,
             typename Index >
   static bool containsOnlyValue( const Element* data,
                                  const Index size,
                                  const Element& value );
};

template<>
struct ArrayOperations< Devices::Cuda >
{
   template< typename Element >
   static void setElement( Element* data,
                           const Element& value );

   template< typename Element >
   static Element getElement( const Element* data );

   template< typename Element, typename Index >
   static void set( Element* data,
                    const Element& value,
                    const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size );

   template< typename DestinationElement,
             typename Index,
             typename SourceIterator >
   static void copyFromIterator( DestinationElement* destination,
                                 Index destinationSize,
                                 SourceIterator first,
                                 SourceIterator last );

   template< typename Element1,
             typename Element2,
             typename Index >
   static bool compare( const Element1* destination,
                        const Element2* source,
                        const Index size );

   template< typename Element,
             typename Index >
   static bool containsValue( const Element* data,
                              const Index size,
                              const Element& value );

   template< typename Element,
             typename Index >
   static bool containsOnlyValue( const Element* data,
                                  const Index size,
                                  const Element& value );
};

template<>
struct ArrayOperations< Devices::Cuda, Devices::Host >
{
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool compare( const DestinationElement* destination,
                        const SourceElement* source,
                        const Index size );
};

template<>
struct ArrayOperations< Devices::Host, Devices::Cuda >
{
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size );

   template< typename Element1,
             typename Element2,
             typename Index >
   static bool compare( const Element1* destination,
                        const Element2* source,
                        const Index size );
};


template<>
struct ArrayOperations< Devices::MIC >
{
   template< typename Element >
   static void setElement( Element* data,
                           const Element& value );

   template< typename Element >
   static Element getElement( const Element* data );

   template< typename Element, typename Index >
   static void set( Element* data,
                    const Element& value,
                    const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size );

   template< typename DestinationElement,
             typename Index,
             typename SourceIterator >
   static void copyFromIterator( DestinationElement* destination,
                                 Index destinationSize,
                                 SourceIterator first,
                                 SourceIterator last );

   template< typename Element1,
             typename Element2,
             typename Index >
   static bool compare( const Element1* destination,
                        const Element2* source,
                        const Index size );

   template< typename Element,
             typename Index >
   static bool containsValue( const Element* data,
                              const Index size,
                              const Element& value );

   template< typename Element,
             typename Index >
   static bool containsOnlyValue( const Element* data,
                                  const Index size,
                                  const Element& value );
};

template<>
struct ArrayOperations< Devices::MIC, Devices::Host >
{
   public:

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copy( DestinationElement* destination,
                        const SourceElement* source,
                        const Index size );

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static bool compare( const DestinationElement* destination,
                           const SourceElement* source,
                           const Index size );
};

template<>
struct ArrayOperations< Devices::Host, Devices::MIC >
{
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool compare( const DestinationElement* destination,
                        const SourceElement* source,
                        const Index size );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/ArrayOperationsHost.hpp>
#include <TNL/Containers/Algorithms/ArrayOperationsCuda.hpp>
#include <TNL/Containers/Algorithms/ArrayOperationsMIC.hpp>
