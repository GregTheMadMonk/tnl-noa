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

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Algorithms/detail/ScanType.h>

namespace TNL {
namespace Algorithms {
namespace detail {

template< typename Device, ScanType Type >
struct Scan;

template< ScanType Type >
struct Scan< Devices::Sequential, Type >
{
   template< typename Vector,
             typename Reduction >
   static void
   perform( Vector& v,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::ValueType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::ValueType zero );
};

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
            const typename Vector::ValueType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::ValueType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::ValueType zero );
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
            const typename Vector::ValueType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::ValueType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::ValueType zero );
};

} // namespace detail
} // namespace Algorithms
} // namespace TNL

#include "Scan.hpp"
