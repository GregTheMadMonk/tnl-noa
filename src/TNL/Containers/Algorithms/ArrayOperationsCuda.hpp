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

#include <TNL/Math.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Containers/Algorithms/ReductionOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Element, typename Index >
void
ArrayOperations< Devices::Cuda >::
allocateMemory( Element*& data,
                const Index size )
{
#ifdef HAVE_CUDA
   TNL_CHECK_CUDA_DEVICE;
   if( cudaMalloc( ( void** ) &data,
                   ( size_t ) size * sizeof( Element ) ) != cudaSuccess )
   {
      data = 0;
      throw Exceptions::CudaBadAlloc();
   }
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename Element >
void
ArrayOperations< Devices::Cuda >::
freeMemory( Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to free a nullptr." );
#ifdef HAVE_CUDA
   TNL_CHECK_CUDA_DEVICE;
   cudaFree( data );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename Element >
void
ArrayOperations< Devices::Cuda >::
setMemoryElement( Element* data,
                  const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   ArrayOperations< Devices::Cuda >::setMemory( data, value, 1 );
}

template< typename Element >
Element
ArrayOperations< Devices::Cuda >::
getMemoryElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
   Element result;
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< Element, Element, int >( &result, data, 1 );
   return result;
}


#ifdef HAVE_CUDA
template< typename Element, typename Index >
__global__ void
setArrayValueCudaKernel( Element* data,
                         const Index size,
                         const Element value )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      data[ elementIdx ] = value;
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Element, typename Index >
void
ArrayOperations< Devices::Cuda >::
setMemory( Element* data,
           const Element& value,
           const Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
#ifdef HAVE_CUDA
   dim3 blockSize( 0 ), gridSize( 0 );
   blockSize. x = 256;
   Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
   gridSize. x = min( blocksNumber, Devices::Cuda::getMaxGridSize() );
   setArrayValueCudaKernel<<< gridSize, blockSize >>>( data, size, value );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

#ifdef HAVE_CUDA
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
__global__ void
copyMemoryCudaToCudaKernel( DestinationElement* destination,
                            const SourceElement* source,
                            const Index size )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      destination[ elementIdx ] = source[ elementIdx ];
      elementIdx += maxGridSize;
   }
}
#endif

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
void
ArrayOperations< Devices::Cuda >::
copyMemory( DestinationElement* destination,
            const SourceElement* source,
            const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
#ifdef HAVE_CUDA
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      cudaMemcpy( destination,
                  source,
                  size * sizeof( DestinationElement ),
                  cudaMemcpyDeviceToDevice );
      TNL_CHECK_CUDA_DEVICE;
   }
   else
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      gridSize. x = min( blocksNumber, Devices::Cuda::getMaxGridSize() );
      copyMemoryCudaToCudaKernel<<< gridSize, blockSize >>>( destination, source, size );
      TNL_CHECK_CUDA_DEVICE;
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename DestinationElement,
          typename SourceElement >
void
ArrayOperations< Devices::Cuda >::
copySTLList( DestinationElement* destination,
             const std::list< SourceElement >& source )
{
   const auto size = source.size();
   const std::size_t copy_buffer_size = std::min( Devices::Cuda::TransferBufferSize / (std::size_t) sizeof( DestinationElement ), ( std::size_t ) size );
   using BaseType = typename std::remove_cv< DestinationElement >::type;
   std::unique_ptr< BaseType[] > copy_buffer{ new BaseType[ copy_buffer_size ] };
   size_t copiedElements = 0;
   auto it = source.begin();
   while( copiedElements < size )
   {
      const auto copySize = std::min( size - copiedElements, copy_buffer_size );
      for( size_t i = 0; i < copySize; i++ )
         copy_buffer[ copiedElements ++ ] = static_cast< DestinationElement >( * it ++ );
      ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( destination, copy_buffer, copySize );
   }
}

template< typename Element1,
          typename Element2,
          typename Index >
bool
ArrayOperations< Devices::Cuda >::
compareMemory( const Element1* destination,
               const Element2* source,
               const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   Algorithms::ParallelReductionEqualities< Element1, Element2 > reductionEqualities;
   return Reduction< Devices::Cuda >::reduce( reductionEqualities, size, destination, source );
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
   TNL_ASSERT_GE( size, 0, "" );
   if( size == 0 ) return false;
   Algorithms::ParallelReductionContainsValue< Element > reductionContainsValue;
   reductionContainsValue.setValue( value );
   return Reduction< Devices::Cuda >::reduce( reductionContainsValue, size, data, nullptr );
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
   Algorithms::ParallelReductionContainsOnlyValue< Element > reductionContainsOnlyValue;
   reductionContainsOnlyValue.setValue( value );
   return Reduction< Devices::Cuda >::reduce( reductionContainsOnlyValue, size, data, nullptr );
}


/****
 * Operations CUDA -> Host
 */
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
void
ArrayOperations< Devices::Host, Devices::Cuda >::
copyMemory( DestinationElement* destination,
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
      std::unique_ptr< SourceElement[] > buffer{ new SourceElement[ Devices::Cuda::getGPUTransferBufferSize() ] };
      Index i( 0 );
      while( i < size )
      {
         if( cudaMemcpy( (void*) buffer.get(),
                         (void*) &source[ i ],
                         min( size - i, Devices::Cuda::getGPUTransferBufferSize() ) * sizeof( SourceElement ),
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
compareMemory( const Element1* destination,
               const Element2* source,
               const Index size )
{
   /***
    * Here, destination is on host and source is on CUDA device.
    */
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "Array size must be non-negative." );
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
      if( ! ArrayOperations< Devices::Host >::compareMemory( &destination[ compared ], host_buffer.get(), transfer ) )
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
copyMemory( DestinationElement* destination,
            const SourceElement* source,
            const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
   TNL_ASSERT_GE( size, 0, "Array size must be non-negative." );
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
compareMemory( const Element1* hostData,
               const Element2* deviceData,
               const Index size )
{
   TNL_ASSERT_TRUE( hostData, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( deviceData, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "Array size must be non-negative." );
   return ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory( deviceData, hostData, size );
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
