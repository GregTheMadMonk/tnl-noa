/***************************************************************************
                          StaticFor.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>

#include <TNL/Devices/CudaCallable.h>

namespace TNL {

template< typename IndexType, IndexType val >
struct StaticForIndexTag
{
   static constexpr IndexType value = val;
   using Decrement = StaticForIndexTag<IndexType, val - 1>;
};


template< typename IndexType,
          typename Begin,
          typename N,
          template< IndexType > class LoopBody >
struct StaticForExecutor
{
   template< typename... Args >
   __cuda_callable__
   static void exec( Args&&... args )
   {
      StaticForExecutor< IndexType, Begin, typename N::Decrement, LoopBody >::exec( std::forward< Args >( args )... );
      LoopBody< Begin::value + N::value - 1 >::exec( std::forward< Args >( args )... );
   }

   template< typename... Args >
   static void execHost( Args&&... args )
   {
      StaticForExecutor< IndexType, Begin, typename N::Decrement, LoopBody >::execHost( std::forward< Args >( args )... );
      LoopBody< Begin::value + N::value - 1 >::exec( std::forward< Args >( args )... );
   }
};

template< typename IndexType,
          typename Begin,
          template< IndexType > class LoopBody >
struct StaticForExecutor< IndexType,
                          Begin,
                          StaticForIndexTag< IndexType, 0 >,
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

template< typename IndexType,
          IndexType begin,
          IndexType end,
          template< IndexType > class LoopBody >
struct StaticFor
{
   template< typename... Args >
   __cuda_callable__
   static void exec( Args&&... args )
   {
      StaticForExecutor< IndexType,
                         StaticForIndexTag< IndexType, begin >,
                         StaticForIndexTag< IndexType, end - begin >,
                         LoopBody >::exec( std::forward< Args >( args )... );
   }

   // nvcc would complain if we wonted to call a host-only function from the __cuda_callable__ exec above
   template< typename... Args >
   static void execHost( Args&&... args )
   {
      StaticForExecutor< IndexType,
                         StaticForIndexTag< IndexType, begin >,
                         StaticForIndexTag< IndexType, end - begin >,
                         LoopBody >::execHost( std::forward< Args >( args )... );
   }
};

} // namespace TNL
