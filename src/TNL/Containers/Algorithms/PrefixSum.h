/***************************************************************************
                          PrefixSum.h  -  description
                             -------------------
    begin                : May 9, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/MIC.h>
#include <TNL/Containers/Algorithms/PrefixSumType.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Device,
           PrefixSumType Type = PrefixSumType::Inclusive >
class PrefixSum {};

template< typename Device,
           PrefixSumType Type = PrefixSumType::Inclusive >
class SegmentedPrefixSum {};


template< PrefixSumType Type >
class PrefixSum< Devices::Host, Type >
{
   public:
      template< typename Vector,
                typename PrefixSumOperation,
                typename VolatilePrefixSumOperation >
      static void
      perform( Vector& v,
               const typename Vector::IndexType begin,
               const typename Vector::IndexType end,
               PrefixSumOperation& reduction,
               VolatilePrefixSumOperation& volatilePrefixSum,
               const typename Vector::RealType& zero );
};

template< PrefixSumType Type >
class PrefixSum< Devices::Cuda, Type >
{
   public:
      template< typename Vector,
                typename PrefixSumOperation,
                typename VolatilePrefixSumOperation >
      static void
      perform( Vector& v,
               const typename Vector::IndexType begin,
               const typename Vector::IndexType end,
               PrefixSumOperation& reduction,
               VolatilePrefixSumOperation& volatilePrefixSum,
               const typename Vector::RealType& zero );
};

template< PrefixSumType Type >
class SegmentedPrefixSum< Devices::Host, Type >
{
   public:
      template< typename Vector,
                typename PrefixSumOperation,
                typename VolatilePrefixSumOperation,
                typename Flags >
      static void
      perform( Vector& v,
               Flags& flags,
               const typename Vector::IndexType begin,
               const typename Vector::IndexType end,
               PrefixSumOperation& reduction,
               VolatilePrefixSumOperation& volatilePrefixSum,
               const typename Vector::RealType& zero );
};

template< PrefixSumType Type >
class SegmentedPrefixSum< Devices::Cuda, Type >
{
   public:
      template< typename Vector,
                typename PrefixSumOperation,
                typename VolatilePrefixSumOperation,
                typename Flags >
      static void
      perform( Vector& v,
               Flags& flags,
               const typename Vector::IndexType begin,
               const typename Vector::IndexType end,
               PrefixSumOperation& reduction,
               VolatilePrefixSumOperation& volatilePrefixSum,
               const typename Vector::RealType& zero );
};



} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/PrefixSum.hpp>
