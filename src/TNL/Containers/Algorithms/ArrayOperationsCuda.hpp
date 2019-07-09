/***************************************************************************
                          ArrayOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Jul 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>

#include <TNL/Math.h>
#include <TNL/ParallelFor.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Containers/Algorithms/ReductionOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Element >
void
ArrayOperations< Devices::Cuda >::
setElement( Element* data,
            const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   ArrayOperations< Devices::Cuda >::set( data, value, 1 );
}

template< typename Element >
Element
ArrayOperations< Devices::Cuda >::
getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
   Element result;
   ArrayOperations< Devices::Host, Devices::Cuda >::copy< Element, Element, int >( &result, data, 1 );
   return result;
}

template< typename Element, typename Index >
void
ArrayOperations< Devices::Cuda >::
set( Element* data,
     const Element& value,
     const Index size )
{
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
ArrayOperations< Devices::Cuda >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
#ifdef HAVE_CUDA
      cudaMemcpy( destination,
                  source,
                  size * sizeof( DestinationElement ),
                  cudaMemcpyDeviceToDevice );
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
   else
   {
      auto kernel = [destination, source] __cuda_callable__ ( Index i )
      {
         destination[ i ] = source[ i ];
      };
      ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
   }
}

template< typename DestinationElement,
          typename Index,
          typename SourceIterator >
void
ArrayOperations< Devices::Cuda >::
copyFromIterator( DestinationElement* destination,
                  Index destinationSize,
                  SourceIterator first,
                  SourceIterator last )
{
   using BaseType = typename std::remove_cv< DestinationElement >::type;
   std::unique_ptr< BaseType[] > buffer{ new BaseType[ Devices::Cuda::getGPUTransferBufferSize() ] };
   Index copiedElements = 0;
   while( copiedElements < destinationSize && first != last ) {
      Index i = 0;
      while( i < Devices::Cuda::getGPUTransferBufferSize() && first != last )
         buffer[ i++ ] = *first++;
      ArrayOperations< Devices::Cuda, Devices::Host >::copy( &destination[ copiedElements ], buffer.get(), i );
      copiedElements += i;
   }
   if( first != last )
      throw std::length_error( "Source iterator is larger than the destination array." );
}

template< typename Element1,
          typename Element2,
          typename Index >
bool
ArrayOperations< Devices::Cuda >::
compare( const Element1* destination,
         const Element2* source,
         const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return  ( destination[ i ] == source[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Reduction< Devices::Cuda >::reduce( size, reduction, volatileReduction, fetch, true );
}

template< typename Element,
          typename Index >
bool
ArrayOperations< Devices::Cuda >::
containsValue( const Element* data,
               const Index size,
               const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "" );

   if( size == 0 ) return false;
   auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return  ( data[ i ] == value ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a |= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a |= b; };
   return Reduction< Devices::Cuda >::reduce( size, reduction, volatileReduction, fetch, false );
}

template< typename Element,
          typename Index >
bool
ArrayOperations< Devices::Cuda >::
containsOnlyValue( const Element* data,
                   const Index size,
                   const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "" );
   if( size == 0 ) return false;

   if( size == 0 ) return false;
   auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return  ( data[ i ] == value ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Reduction< Devices::Cuda >::reduce( size, reduction, volatileReduction, fetch, true );
}


/****
 * Operations CUDA -> Host
 */
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
void
ArrayOperations< Devices::Host, Devices::Cuda >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
#ifdef HAVE_CUDA
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      if( cudaMemcpy( destination,
                      source,
                      size * sizeof( DestinationElement ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
   }
   else
   {
      using BaseType = typename std::remove_cv< SourceElement >::type;
      std::unique_ptr< BaseType[] > buffer{ new BaseType[ Devices::Cuda::getGPUTransferBufferSize() ] };
      Index i( 0 );
      while( i < size )
      {
         if( cudaMemcpy( (void*) buffer.get(),
                         (void*) &source[ i ],
                         TNL::min( size - i, Devices::Cuda::getGPUTransferBufferSize() ) * sizeof( SourceElement ),
                         cudaMemcpyDeviceToHost ) != cudaSuccess )
            std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
         TNL_CHECK_CUDA_DEVICE;
         Index j( 0 );
         while( j < Devices::Cuda::getGPUTransferBufferSize() && i + j < size )
         {
            destination[ i + j ] = buffer[ j ];
            j++;
         }
         i += j;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}


template< typename Element1,
          typename Element2,
          typename Index >
bool
ArrayOperations< Devices::Host, Devices::Cuda >::
compare( const Element1* destination,
         const Element2* source,
         const Index size )
{
   /***
    * Here, destination is on host and source is on CUDA device.
    */
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
#ifdef HAVE_CUDA
   std::unique_ptr< Element2[] > host_buffer{ new Element2[ Devices::Cuda::getGPUTransferBufferSize() ] };
   Index compared( 0 );
   while( compared < size )
   {
      Index transfer = min( size - compared, Devices::Cuda::getGPUTransferBufferSize() );
      if( cudaMemcpy( (void*) host_buffer.get(),
                      (void*) &source[ compared ],
                      transfer * sizeof( Element2 ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
      if( ! ArrayOperations< Devices::Host >::compare( &destination[ compared ], host_buffer.get(), transfer ) )
         return false;
      compared += transfer;
   }
   return true;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

/****
 * Operations Host -> CUDA
 */
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
void
ArrayOperations< Devices::Cuda, Devices::Host >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
#ifdef HAVE_CUDA
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      if( cudaMemcpy( destination,
                      source,
                      size * sizeof( DestinationElement ),
                      cudaMemcpyHostToDevice ) != cudaSuccess )
         std::cerr << "Transfer of data from host to CUDA device failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
   }
   else
   {
      std::unique_ptr< DestinationElement[] > buffer{ new DestinationElement[ Devices::Cuda::getGPUTransferBufferSize() ] };
      Index i( 0 );
      while( i < size )
      {
         Index j( 0 );
         while( j < Devices::Cuda::getGPUTransferBufferSize() && i + j < size )
         {
            buffer[ j ] = source[ i + j ];
            j++;
         }
         if( cudaMemcpy( (void*) &destination[ i ],
                         (void*) buffer.get(),
                         j * sizeof( DestinationElement ),
                         cudaMemcpyHostToDevice ) != cudaSuccess )
            std::cerr << "Transfer of data from host to CUDA device failed." << std::endl;
         TNL_CHECK_CUDA_DEVICE;
         i += j;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename Element1,
          typename Element2,
          typename Index >
bool
ArrayOperations< Devices::Cuda, Devices::Host >::
compare( const Element1* hostData,
         const Element2* deviceData,
         const Index size )
{
   TNL_ASSERT_TRUE( hostData, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( deviceData, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
   return ArrayOperations< Devices::Host, Devices::Cuda >::compare( deviceData, hostData, size );
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
