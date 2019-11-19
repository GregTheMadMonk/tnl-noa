/***************************************************************************
                          TemplateStaticFor.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>
#include <type_traits>

#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief TemplateStaticFor serves for coding for-loops in template parameters.
 *
 * The result of calling this loop with a templated class \p LoopBody is as follows:
 *
 * LoopBody< begin >::exec( ... );
 *
 * LoodBody< begin + 1 >::exec( ... );
 *
 * ...
 *
 * LoopBody< end - 1 >::exec( ... );
 *
 * \tparam IndexType is type of the loop indexes
 * \tparam begin the loop iterates over index interval [begin,end).
 * \tparam end the loop iterates over index interval [begin,end).
 * \tparam LoopBody is a templated class having one template parameter of IndexType.
 */
template< typename IndexType,
          IndexType begin,
          IndexType end,
          template< IndexType > class LoopBody >
struct TemplateStaticFor;

namespace detail {

template< typename IndexType,
          typename Begin,
          typename N,
          template< IndexType > class LoopBody >
struct TemplateStaticForExecutor
{
   template< typename... Args >
   __cuda_callable__
   static void exec( Args&&... args )
   {
      using Decrement = std::integral_constant< IndexType, N::value - 1 >;
      TemplateStaticForExecutor< IndexType, Begin, Decrement, LoopBody >::exec( std::forward< Args >( args )... );
      LoopBody< Begin::value + N::value - 1 >::exec( std::forward< Args >( args )... );
   }

   template< typename... Args >
   static void execHost( Args&&... args )
   {
      using Decrement = std::integral_constant< IndexType, N::value - 1 >;
      TemplateStaticForExecutor< IndexType, Begin, Decrement, LoopBody >::execHost( std::forward< Args >( args )... );
      LoopBody< Begin::value + N::value - 1 >::exec( std::forward< Args >( args )... );
   }
};

template< typename IndexType,
          typename Begin,
          template< IndexType > class LoopBody >
struct TemplateStaticForExecutor< IndexType,
                                  Begin,
                                  std::integral_constant< IndexType, 0 >,
                                  LoopBody >
{
   template< typename... Args >
   __cuda_callable__
   static void exec( Args&&... args )
   {}

   template< typename... Args >
   static void execHost( Args&&... args )
   {}
};

} // namespace detail

template< typename IndexType,
          IndexType begin,
          IndexType end,
          template< IndexType > class LoopBody >
struct TemplateStaticFor
{
   template< typename... Args >
   __cuda_callable__
   static void exec( Args&&... args )
   {
      detail::TemplateStaticForExecutor< IndexType,
                                 std::integral_constant< IndexType, begin >,
                                 std::integral_constant< IndexType, end - begin >,
                                 LoopBody >::exec( std::forward< Args >( args )... );
   }

   // nvcc would complain if we wonted to call a host-only function from the __cuda_callable__ exec above
   template< typename... Args >
   static void execHost( Args&&... args )
   {
      detail::TemplateStaticForExecutor< IndexType,
                                 std::integral_constant< IndexType, begin >,
                                 std::integral_constant< IndexType, end - begin >,
                                 LoopBody >::execHost( std::forward< Args >( args )... );
   }
};

} // namespace Algorithms
} // namespace TNL
