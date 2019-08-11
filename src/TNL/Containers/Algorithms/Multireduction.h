/***************************************************************************
                          Multireduction.h  -  description
                             -------------------
    begin                : May 13, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Device >
struct Multireduction;

template<>
struct Multireduction< Devices::Host >
{
   /**
    * Parameters:
    *    zero: starting value for reduction
    *    dataFetcher: callable object such that `dataFetcher( i, j )` yields
    *                 the i-th value to be reduced from the j-th dataset
    *                 (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *    volatileReduction: callable object representing the reduction operation
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    result: output array of size = n
    */
   template< typename Result,
             typename DataFetcher,
             typename Reduction,
             typename VolatileReduction,
             typename Index >
   static void
   reduce( const Result zero,
           DataFetcher dataFetcher,
           const Reduction reduction,
           const VolatileReduction volatileReduction,
           const Index size,
           const int n,
           Result* result );
};

template<>
struct Multireduction< Devices::Cuda >
{
   /**
    * Parameters:
    *    zero: starting value for reduction
    *    dataFetcher: callable object such that `dataFetcher( i, j )` yields
    *                 the i-th value to be reduced from the j-th dataset
    *                 (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *    volatileReduction: callable object representing the reduction operation
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    hostResult: output array of size = n
    */
   template< typename Result,
             typename DataFetcher,
             typename Reduction,
             typename VolatileReduction,
             typename Index >
   static void
   reduce( const Result zero,
           DataFetcher dataFetcher,
           const Reduction reduction,
           const VolatileReduction volatileReduction,
           const Index size,
           const int n,
           Result* hostResult );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include "Multireduction.hpp"
