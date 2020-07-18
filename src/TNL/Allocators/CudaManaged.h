/***************************************************************************
                          CudaManaged.h  -  description
                             -------------------
    begin                : Apr 8, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Cuda/CheckDevice.h>

namespace TNL {
namespace Allocators {

/**
 * \brief Allocator for the CUDA Unified Memory system.
 *
 * The memory allocated by this allocator will be automatically managed by the
 * CUDA Unified Memory system. The allocation is done using the
 * `cudaMallocManaged` function and the deallocation is done using the
 * `cudaFree` function.
 */
template< class T >
struct CudaManaged
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   CudaManaged() = default;
   CudaManaged( const CudaManaged& ) = default;
   CudaManaged( CudaManaged&& ) = default;

   CudaManaged& operator=( const CudaManaged& ) = default;
   CudaManaged& operator=( CudaManaged&& ) = default;

   template< class U >
   CudaManaged( const CudaManaged< U >& )
   {}

   template< class U >
   CudaManaged( CudaManaged< U >&& )
   {}

   template< class U >
   CudaManaged& operator=( const CudaManaged< U >& )
   {
      return *this;
   }

   template< class U >
   CudaManaged& operator=( CudaManaged< U >&& )
   {
      return *this;
   }

   value_type* allocate( size_type n )
   {
#ifdef HAVE_CUDA
      TNL_CHECK_CUDA_DEVICE;
      value_type* result = nullptr;
      if( cudaMallocManaged( &result, n * sizeof(value_type) ) != cudaSuccess )
         throw Exceptions::CudaBadAlloc();
      TNL_CHECK_CUDA_DEVICE;
      return result;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }

   void deallocate(value_type* ptr, size_type)
   {
#ifdef HAVE_CUDA
      TNL_CHECK_CUDA_DEVICE;
      cudaFree( (void*) ptr );
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

template<class T1, class T2>
bool operator==(const CudaManaged<T1>&, const CudaManaged<T2>&)
{
   return true;
}

template<class T1, class T2>
bool operator!=(const CudaManaged<T1>& lhs, const CudaManaged<T2>& rhs)
{
   return !(lhs == rhs);
}

} // namespace Allocators
} // namespace TNL
