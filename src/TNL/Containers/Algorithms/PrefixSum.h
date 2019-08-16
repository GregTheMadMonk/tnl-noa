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

namespace TNL {
namespace Containers {
namespace Algorithms {

enum class PrefixSumType {
   Exclusive,
   Inclusive
};

template< typename Device,
           PrefixSumType Type = PrefixSumType::Inclusive >
struct PrefixSum;

template< typename Device,
           PrefixSumType Type = PrefixSumType::Inclusive >
struct SegmentedPrefixSum;


template< PrefixSumType Type >
struct PrefixSum< Devices::Host, Type >
{
   template< typename Vector,
             typename Reduction >
   static void
   perform( Vector& v,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::RealType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::RealType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::RealType shift );
};

template< PrefixSumType Type >
struct PrefixSum< Devices::Cuda, Type >
{
   template< typename Vector,
             typename Reduction >
   static void
   perform( Vector& v,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::RealType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::RealType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::RealType shift );
};

template< PrefixSumType Type >
struct SegmentedPrefixSum< Devices::Host, Type >
{
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::RealType zero );
};

template< PrefixSumType Type >
struct SegmentedPrefixSum< Devices::Cuda, Type >
{
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::RealType zero );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/PrefixSum.hpp>
