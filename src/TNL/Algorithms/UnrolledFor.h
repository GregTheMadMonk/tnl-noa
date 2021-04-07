/***************************************************************************
                          UnrolledFor.h  -  description
                             -------------------
    begin                : Jul 16, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>

#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief UnrolledFor is a wrapper for common for-loop with explicit unrolling.
 *
 * UnrolledFor can be used only for for-loops bounds of which are known at the
 * compile time. UnrolledFor performs explicit loop unrolling for better performance.
 * This, however, does not make sense for loops with a large iterations
 * count. For a very large iterations count it could trigger the compiler's
 * limit on recursive template instantiation. Also note that the compiler
 * will (at least partially) unroll loops with static bounds anyway. For theses
 * reasons, the explicit loop unrolling can be controlled by the third template
 * parameter.
 *
 * \tparam Begin the loop will iterate over indexes [Begin,End)
 * \tparam End the loop will iterate over indexes [Begin,End)
 * \tparam unrolled controls the explicit loop unrolling. If it is true, the
 *   unrolling is performed.
 *
 * \par Example
 * \include Algorithms/UnrolledForExample.cpp
 * \par Output
 * \include UnrolledForExample.out
 */
template< int Begin, int End, bool unrolled = (End - Begin <= 8) >
struct UnrolledFor;

template< int Begin, int End >
struct UnrolledFor< Begin, End, true >
{
   static_assert( Begin < End, "Wrong index interval for UnrolledFor. Begin must be less than end." );

   /**
    * \brief Static method for the execution of the UnrolledFor.
    *
    * \param f is a (lambda) function to be performed in each iteration.
    * \param args are auxiliary data to be passed to the function f.
    */
   template< typename Function, typename... Args >
   __cuda_callable__
   static void exec( const Function& f, Args&&... args )
   {
      f( Begin, args... );
      UnrolledFor< Begin + 1, End >::exec( f, std::forward< Args >( args )... );
   }
};

template< int End >
struct UnrolledFor< End, End, true >
{
   template< typename Function, typename... Args >
   __cuda_callable__
   static void exec( const Function& f, Args&&... args ) {}
};

template< int Begin, int End >
struct UnrolledFor< Begin, End, false >
{
   static_assert( Begin <= End, "Wrong index interval for UnrolledFor. Begin must be less than or equal to end." );

   template< typename Function, typename... Args >
   __cuda_callable__
   static void exec( const Function& f, Args&&... args )
   {
      for( int i = Begin; i < End; i++ )
         f( i, std::forward< Args >( args )... );
   }
};

} // namespace Algorithms
} // namespace TNL
