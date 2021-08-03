/***************************************************************************
                          MemoryOperations.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
namespace Algorithms {

template< typename DestinationDevice >
struct MemoryOperations;

template<>
struct MemoryOperations< Devices::Sequential >
{
   template< typename Element, typename Index >
   __cuda_callable__
   static void construct( Element* data,
                          const Index size );

   // note that args are passed by reference to the constructor, not via
   // std::forward since move-semantics does not apply for the construction of
   // multiple elements
   template< typename Element, typename Index, typename... Args >
   __cuda_callable__
   static void construct( Element* data,
                          const Index size,
                          const Args&... args );

   template< typename Element, typename Index >
   __cuda_callable__
   static void destruct( Element* data,
                         const Index size );

   template< typename Element >
   __cuda_callable__
   static void setElement( Element* data,
                           const Element& value );

   template< typename Element >
   __cuda_callable__
   static Element getElement( const Element* data );

   template< typename Element, typename Index >
   __cuda_callable__
   static void set( Element* data,
                    const Element& value,
                    const Index size );

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   __cuda_callable__
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
   __cuda_callable__
   static bool compare( const Element1* destination,
                        const Element2* source,
                        const Index size );
};

template<>
struct MemoryOperations< Devices::Host >
{
   template< typename Element, typename Index >
   static void construct( Element* data,
                          const Index size );

   // note that args are passed by reference to the constructor, not via
   // std::forward since move-semantics does not apply for the construction of
   // multiple elements
   template< typename Element, typename Index, typename... Args >
   static void construct( Element* data,
                          const Index size,
                          const Args&... args );

   template< typename Element, typename Index >
   static void destruct( Element* data,
                         const Index size );

   // this is __cuda_callable__ only to silence nvcc warnings
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Element >
   __cuda_callable__
   static void setElement( Element* data,
                           const Element& value );

   // this is __cuda_callable__ only to silence nvcc warnings
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Element >
   __cuda_callable__
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
};

template<>
struct MemoryOperations< Devices::Cuda >
{
   template< typename Element, typename Index >
   static void construct( Element* data,
                          const Index size );

   // note that args are passed by value to the constructor, not via
   // std::forward or even by reference, since move-semantics does not apply for
   // the construction of multiple elements and pass-by-reference cannot be used
   // with CUDA kernels
   template< typename Element, typename Index, typename... Args >
   static void construct( Element* data,
                          const Index size,
                          const Args&... args );

   template< typename Element, typename Index >
   static void destruct( Element* data,
                         const Index size );

   template< typename Element >
   __cuda_callable__
   static void setElement( Element* data,
                           const Element& value );

   template< typename Element >
   __cuda_callable__
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
};

} // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/MemoryOperationsSequential.hpp>
#include <TNL/Algorithms/MemoryOperationsHost.hpp>
#include <TNL/Algorithms/MemoryOperationsCuda.hpp>
