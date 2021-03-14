/***************************************************************************
                          MemoryOperationsCuda.hpp  -  description
                             -------------------
    begin                : Jul 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <memory>  // std::unique_ptr
#include <stdexcept>

#include <TNL/Algorithms/MemoryOperations.h>
#include <TNL/Algorithms/MultiDeviceMemoryOperations.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/Reduction.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

namespace TNL {
namespace Algorithms {

template< typename Element >
__cuda_callable__ void
MemoryOperations< Devices::Cuda >::
setElement( Element* data,
            const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
#ifdef __CUDA_ARCH__
   *data = value;
#else
#ifdef HAVE_CUDA
   cudaMemcpy( ( void* ) data, ( void* ) &value, sizeof( Element ), cudaMemcpyHostToDevice );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
   // TODO: For some reason the following does not work after adding
   // #ifdef __CUDA_ARCH__ to Array::setElement and ArrayView::setElement.
   // Probably it might be a problem with lambda function 'kernel' which
   // nvcc probably does not handle properly.
   //MemoryOperations< Devices::Cuda >::set( data, value, 1 );
#endif
}

template< typename Element >
__cuda_callable__ Element
MemoryOperations< Devices::Cuda >::
getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
#ifdef __CUDA_ARCH__
   return *data;
#else
   // TODO: For some reason the following does not work after adding
   // #ifdef __CUDA_ARCH__ to Array::getElement and ArrayView::getElement 
   // Probably it might be a problem with lambda function 'kernel' which
   // nvcc probably does not handle properly.
   //MultiDeviceMemoryOperations< void, Devices::Cuda >::template copy< Element, Element, int >( &result, data, 1 );
   #ifdef HAVE_CUDA
      Element result;
      cudaMemcpy( ( void* ) &result, ( void* ) data, sizeof( Element ), cudaMemcpyDeviceToHost );
      TNL_CHECK_CUDA_DEVICE;
      return result;
   #else
      throw Exceptions::CudaSupportMissing();
   #endif
#endif
}

template< typename Element, typename Index >
void
MemoryOperations< Devices::Cuda >::
set( Element* data,
     const Element& value,
     const Index size )
{
   if( size == 0 ) return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   auto kernel = [data, value] __cuda_callable__ ( Index i )
   {
      data[ i ] = value;
   };
   ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
void
MemoryOperations< Devices::Cuda >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   if( size == 0 ) return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );

   // our ParallelFor kernel is faster than cudaMemcpy
   auto kernel = [destination, source] __cuda_callable__ ( Index i )
   {
      destination[ i ] = source[ i ];
   };
   ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
}

template< typename DestinationElement,
          typename Index,
          typename SourceIterator >
void
MemoryOperations< Devices::Cuda >::
copyFromIterator( DestinationElement* destination,
                  Index destinationSize,
                  SourceIterator first,
                  SourceIterator last )
{
   using BaseType = typename std::remove_cv< DestinationElement >::type;
   const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof(BaseType), destinationSize );
   std::unique_ptr< BaseType[] > buffer{ new BaseType[ buffer_size ] };
   Index copiedElements = 0;
   while( copiedElements < destinationSize && first != last ) {
      Index i = 0;
      while( i < buffer_size && first != last )
         buffer[ i++ ] = *first++;
      MultiDeviceMemoryOperations< Devices::Cuda, void >::copy( &destination[ copiedElements ], buffer.get(), i );
      copiedElements += i;
   }
   if( first != last )
      throw std::length_error( "Source iterator is larger than the destination array." );
}

template< typename Element1,
          typename Element2,
          typename Index >
bool
MemoryOperations< Devices::Cuda >::
compare( const Element1* destination,
         const Element2* source,
         const Index size )
{
   if( size == 0 ) return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return destination[ i ] == source[ i ]; };
   return Reduction< Devices::Cuda >::reduce( ( Index ) 0, size, fetch, std::logical_and<>{}, true );
}

template< typename Element,
          typename Index >
bool
MemoryOperations< Devices::Cuda >::
containsValue( const Element* data,
               const Index size,
               const Element& value )
{
   if( size == 0 ) return false;
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "" );

   auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return data[ i ] == value; };
   return Reduction< Devices::Cuda >::reduce( ( Index ) 0, size, fetch, std::logical_or<>{}, false );
}

template< typename Element,
          typename Index >
bool
MemoryOperations< Devices::Cuda >::
containsOnlyValue( const Element* data,
                   const Index size,
                   const Element& value )
{
   if( size == 0 ) return false;
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "" );

   auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return data[ i ] == value; };
   return Reduction< Devices::Cuda >::reduce( ( Index ) 0, size, fetch, std::logical_and<>{}, true );
}

} // namespace Algorithms
} // namespace TNL
