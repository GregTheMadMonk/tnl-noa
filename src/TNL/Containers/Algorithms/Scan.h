/***************************************************************************
                          Scan.h  -  description
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

enum class ScanType {
   Exclusive,
   Inclusive
};

template< typename Device,
           ScanType Type = ScanType::Inclusive >
struct Scan;

template< typename Device,
           ScanType Type = ScanType::Inclusive >
struct SegmentedScan;


template< ScanType Type >
struct Scan< Devices::Host, Type >
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

template< ScanType Type >
struct Scan< Devices::Cuda, Type >
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

template< ScanType Type >
struct SegmentedScan< Devices::Host, Type >
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

template< ScanType Type >
struct SegmentedScan< Devices::Cuda, Type >
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

#include <TNL/Containers/Algorithms/Scan.hpp>
