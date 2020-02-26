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
#if __CUDA_ARCH__ < 600
      if( std::is_same< Value, double >::value )
      {
         unsigned long long int* v_as_ull = ( unsigned long long int* ) &v;
         unsigned long long int old = *v_as_ull, assumed;

         do
         {
            assumed = old;
            old = atomicCAS( v_as_ull,
                             assumed,
                             __double_as_longlong( s + __longlong_as_double( assumed ) ) ) ;

         // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
         }
         while( assumed != old );
         return;
      }
#endif
      atomicAdd( &v, a );
#endif
   }

};

} //namespace Algorithms
} //namespace TNL