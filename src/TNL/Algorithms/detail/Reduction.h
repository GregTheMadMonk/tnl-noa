/***************************************************************************
                          Reduction.h  -  description
                             -------------------
    begin                : Jul 5, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::pair, std::forward

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
   namespace Algorithms {
      namespace detail {

template< typename Device >
struct Reduction;

template<>
struct Reduction< Devices::Sequential >
{
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static constexpr Result
   reduce( const Index begin,
           const Index end,
           Fetch&& fetch,
           Reduce&& reduce,
           const Result& identity );

   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static constexpr std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       const Result& identity );
};

template<>
struct Reduction< Devices::Host >
{
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static Result
   reduce( const Index begin,
           const Index end,
           Fetch&& fetch,
           Reduce&& reduce,
           const Result& identity );

   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       const Result& identity );
};

template<>
struct Reduction< Devices::Cuda >
{
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static Result
   reduce( const Index begin,
           const Index end,
           Fetch&& fetch,
           Reduce&& reduce,
           const Result& identity );

   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       const Result& identity );
};

      } // namespace detail
   } // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/detail/Reduction.hpp>
