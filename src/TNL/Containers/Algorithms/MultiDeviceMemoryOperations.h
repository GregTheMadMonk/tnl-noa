/***************************************************************************
                          MultiDeviceMemoryOperations.h  -  description
                             -------------------
    begin                : Aug 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Algorithms/MemoryOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename DestinationDevice,
          typename SourceDevice = DestinationDevice >
struct MultiDeviceMemoryOperations
{
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size )
   {
      // use DestinationDevice, unless it is void
      using Device = std::conditional_t< std::is_void< DestinationDevice >::value, SourceDevice, DestinationDevice >;
      MemoryOperations< Device >::copy( destination, source, size );
   }

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool compare( const DestinationElement* destination,
                        const SourceElement* source,
                        const Index size )
   {
      // use DestinationDevice, unless it is void
      using Device = std::conditional_t< std::is_void< DestinationDevice >::value, SourceDevice, DestinationDevice >;
      return MemoryOperations< Device >::compare( destination, source, size );
   }
};


template< typename DeviceType >
struct MultiDeviceMemoryOperations< Devices::Cuda, DeviceType >
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

template< typename DeviceType >
struct MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >
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


// CUDA <-> CUDA to disambiguate from partial specializations below
template<>
struct MultiDeviceMemoryOperations< Devices::Cuda, Devices::Cuda >
{
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static void copy( DestinationElement* destination,
                     const SourceElement* source,
                     const Index size )
   {
      MemoryOperations< Devices::Cuda >::copy( destination, source, size );
   }

   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
   static bool compare( const DestinationElement* destination,
                        const SourceElement* source,
                        const Index size )
   {
      return MemoryOperations< Devices::Cuda >::compare( destination, source, size );
   }
};


/****
 * Operations CUDA -> Host
 */
template< typename DeviceType >
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
void
MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   if( size == 0 ) return;
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
      std::unique_ptr< BaseType[] > buffer{ new BaseType[ Cuda::getTransferBufferSize() ] };
      Index i( 0 );
      while( i < size )
      {
         if( cudaMemcpy( (void*) buffer.get(),
                         (void*) &source[ i ],
                         TNL::min( size - i, Cuda::getTransferBufferSize() ) * sizeof( SourceElement ),
                         cudaMemcpyDeviceToHost ) != cudaSuccess )
            std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
         TNL_CHECK_CUDA_DEVICE;
         Index j( 0 );
         while( j < Cuda::getTransferBufferSize() && i + j < size )
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


template< typename DeviceType >
   template< typename Element1,
             typename Element2,
             typename Index >
bool
MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >::
compare( const Element1* destination,
         const Element2* source,
         const Index size )
{
   if( size == 0 ) return true;
   /***
    * Here, destination is on host and source is on CUDA device.
    */
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
#ifdef HAVE_CUDA
   std::unique_ptr< Element2[] > host_buffer{ new Element2[ Cuda::getTransferBufferSize() ] };
   Index compared( 0 );
   while( compared < size )
   {
      Index transfer = min( size - compared, Cuda::getTransferBufferSize() );
      if( cudaMemcpy( (void*) host_buffer.get(),
                      (void*) &source[ compared ],
                      transfer * sizeof( Element2 ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
      if( ! MemoryOperations< Devices::Host >::compare( &destination[ compared ], host_buffer.get(), transfer ) )
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
template< typename DeviceType >
   template< typename DestinationElement,
             typename SourceElement,
             typename Index >
void
MultiDeviceMemoryOperations< Devices::Cuda, DeviceType >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   if( size == 0 ) return;
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
      std::unique_ptr< DestinationElement[] > buffer{ new DestinationElement[ Cuda::getTransferBufferSize() ] };
      Index i( 0 );
      while( i < size )
      {
         Index j( 0 );
         while( j < Cuda::getTransferBufferSize() && i + j < size )
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

template< typename DeviceType >
   template< typename Element1,
             typename Element2,
             typename Index >
bool
MultiDeviceMemoryOperations< Devices::Cuda, DeviceType >::
compare( const Element1* hostData,
         const Element2* deviceData,
         const Index size )
{
   if( size == 0 ) return true;
   TNL_ASSERT_TRUE( hostData, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( deviceData, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
   return MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >::compare( deviceData, hostData, size );
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
