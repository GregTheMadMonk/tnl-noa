/***************************************************************************
                          Cuda.h  -  description
                             -------------------
    begin                : Apr 8, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Allocators {

/**
 * \brief Allocator for the CUDA device memory space.
 *
 * The allocation is done using the `cudaMalloc` function and the deallocation
 * is done using the `cudaFree` function.
 */
template< class T >
struct Cuda
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   Cuda() = default;
   Cuda( const Cuda& ) = default;
   Cuda( Cuda&& ) = default;

   Cuda& operator=( const Cuda& ) = default;
   Cuda& operator=( Cuda&& ) = default;

   template< class U >
   Cuda( const Cuda< U >& )
   {}

   template< class U >
   Cuda( Cuda< U >&& )
   {}

   template< class U >
   Cuda& operator=( const Cuda< U >& )
   {
      return *this;
   }

   template< class U >
   Cuda& operator=( Cuda< U >&& )
   {
      return *this;
   }

   value_type* allocate( size_type n )
   {
#ifdef HAVE_CUDA
      TNL_CHECK_CUDA_DEVICE;
      value_type* result = nullptr;
      if( cudaMalloc( (void**) &result, n * sizeof(value_type) ) != cudaSuccess )
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
      cudaFree( ptr );
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

template<class T1, class T2>
bool operator==(const Cuda<T1>&, const Cuda<T2>&)
{
   return true;
}

template<class T1, class T2>
bool operator!=(const Cuda<T1>& lhs, const Cuda<T2>& rhs)
{
   return !(lhs == rhs);
}

} // namespace Allocators
} // namespace TNL