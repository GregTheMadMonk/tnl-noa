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
class ArrayOperations{};

template<>
class ArrayOperations< Devices::Host >
{
   public:

      template< typename Element, typename Index >
      static void allocateMemory( Element*& data,
                                  const Index size );

      template< typename Element >
      static void freeMemory( Element* data );

      template< typename Element >
      static void setMemoryElement( Element* data,
                                    const Element& value );

      template< typename Element >
      static Element getMemoryElement( const Element* data );

      template< typename Element, typename Index >
      static void setMemory( Element* data,
                             const Element& value,
                             const Index size );

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copyMemory( DestinationElement* destination,
                              const SourceElement* source,
                              const Index size );

      template< typename DestinationElement,
                typename SourceElement >
      static void copySTLList( DestinationElement* destination,
                               const std::list< SourceElement >& source );

      template< typename Element1,
                typename Element2,
                typename Index >
      static bool compareMemory( const Element1* destination,
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
class ArrayOperations< Devices::Cuda >
{
   public:

      template< typename Element, typename Index >
      static void allocateMemory( Element*& data,
                                  const Index size );

      template< typename Element >
      static void freeMemory( Element* data );

      template< typename Element >
      static void setMemoryElement( Element* data,
                                    const Element& value );

      template< typename Element >
      static Element getMemoryElement( const Element* data );

      template< typename Element, typename Index >
      static void setMemory( Element* data,
                             const Element& value,
                             const Index size );

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copyMemory( DestinationElement* destination,
                              const SourceElement* source,
                              const Index size );

      template< typename Element1,
                typename Element2,
                typename Index >
      static bool compareMemory( const Element1* destination,
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
class ArrayOperations< Devices::Cuda, Devices::Host >
{
   public:

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copyMemory( DestinationElement* destination,
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
class ArrayOperations< Devices::Host, Devices::Cuda >
{
   public:

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copyMemory( DestinationElement* destination,
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
class ArrayOperations< Devices::MIC >
{
   public:

      template< typename Element, typename Index >
      static void allocateMemory( Element*& data,
                                  const Index size );

      template< typename Element >
      static void freeMemory( Element* data );

      template< typename Element >
      static void setMemoryElement( Element* data,
                                    const Element& value );

      template< typename Element >
      static Element getMemoryElement( const Element* data );

      template< typename Element, typename Index >
      static void setMemory( Element* data,
                             const Element& value,
                             const Index size );

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copyMemory( DestinationElement* destination,
                              const SourceElement* source,
                              const Index size );

      template< typename Element1,
                typename Element2,
                typename Index >
      static bool compareMemory( const Element1* destination,
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
class ArrayOperations< Devices::MIC, Devices::Host >
{
   public:

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copyMemory( DestinationElement* destination,
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
class ArrayOperations< Devices::Host, Devices::MIC >
{
   public:

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static void copyMemory( DestinationElement* destination,
                              const SourceElement* source,
                              const Index size );

      template< typename DestinationElement,
                typename SourceElement,
                typename Index >
      static bool compareMemory( const DestinationElement* destination,
                                 const SourceElement* source,
                                 const Index size );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/ArrayOperationsHost.hpp>
#include <TNL/Containers/Algorithms/ArrayOperationsCuda.hpp>
#include <TNL/Containers/Algorithms/ArrayOperationsMIC.hpp>
