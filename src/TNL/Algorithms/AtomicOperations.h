/***************************************************************************
                          AtomicOperations.h  -  description
                             -------------------
    begin                : Feb 26, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <cuda.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Algorithms {

template< typename Device >
struct AtomicOperations{};

template<>
struct AtomicOperations< Devices::Host >
{
   template< typename Value >
   static void add( Value& v, const Value& a )
   {
#pragma omp atomic update
      v += a;
   }
};

template<>
struct AtomicOperations< Devices::Cuda >
{
   template< typename Value >
   __cuda_callable__
   static void add( Value& v, const Value& a )
   {
#ifdef HAVE_CUDA
      atomicAdd( &v, a );
#endif // HAVE_CUDA
   }

#ifdef HAVE_CUDA
   __device__
   static void add( double& v, const double& a )
   {
#if __CUDA_ARCH__ < 600
      unsigned long long int* v_as_ull = ( unsigned long long int* ) &v;
      unsigned long long int old = *v_as_ull, assumed;

      do
      {
         assumed = old;
         old = atomicCAS( v_as_ull,
                          assumed,
                          __double_as_longlong( a + __longlong_as_double( assumed ) ) ) ;

      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
      }
      while( assumed != old );
#else // __CUDA_ARCH__ < 600
      atomicAdd( &v, a );
#endif //__CUDA_ARCH__ < 600

#else // HAVE_CUDA
   static void add( double& v, const double& a ){}
#endif // HAVE_CUDA
   }

   __cuda_callable__
   static void add( long int& v, const long int& a )
   {
#ifdef HAVE_CUDA
      TNL_ASSERT_TRUE( false, "Atomic add for long int is not supported on CUDA." );
#endif // HAVE_CUDA
   }
   
   __cuda_callable__
   static void add( short int& v, const short int& a )
   {
#ifdef HAVE_CUDA
      TNL_ASSERT_TRUE( false, "Atomic add for short int is not supported on CUDA." );
#endif // HAVE_CUDA
   }
};
} //namespace Algorithms
} //namespace TNL
