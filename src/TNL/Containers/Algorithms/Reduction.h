/***************************************************************************
                          Reduction.h  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::pair

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Device >
struct Reduction;

template<>
struct Reduction< Devices::Host >
{
   template< typename Index,
             typename Result,
             typename ReductionOperation,
             typename VolatileReductionOperation,
             typename DataFetcher >
   static Result
   reduce( const Index size,
           ReductionOperation& reduction,
           VolatileReductionOperation& volatileReduction,
           DataFetcher& dataFetcher,
           const Result& zero );

   template< typename Index,
             typename Result,
             typename ReductionOperation,
             typename VolatileReductionOperation,
             typename DataFetcher >
   static std::pair< Index, Result >
   reduceWithArgument( const Index size,
                       ReductionOperation& reduction,
                       VolatileReductionOperation& volatileReduction,
                       DataFetcher& dataFetcher,
                       const Result& zero );
};

template<>
struct Reduction< Devices::Cuda >
{
   template< typename Index,
             typename Result,
             typename ReductionOperation,
             typename VolatileReductionOperation,
             typename DataFetcher >
   static Result
   reduce( const Index size,
           ReductionOperation& reduction,
           VolatileReductionOperation& volatileReduction,
           DataFetcher& dataFetcher,
           const Result& zero );

   template< typename Index,
             typename Result,
             typename ReductionOperation,
             typename VolatileReductionOperation,
             typename DataFetcher >
   static std::pair< Index, Result >
   reduceWithArgument( const Index size,
                       ReductionOperation& reduction,
                       VolatileReductionOperation& volatileReduction,
                       DataFetcher& dataFetcher,
                       const Result& zero );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/Reduction.hpp>
