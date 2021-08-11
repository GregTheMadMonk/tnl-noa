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

template< typename Device, ScanType Type, ScanPhaseType PhaseType = ScanPhaseType::WriteInSecondPhase >
struct Scan;

template< ScanType Type, ScanPhaseType PhaseType >
struct Scan< Devices::Sequential, Type, PhaseType >
{
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   // returns the last value of inclusive scan (reduction of the whole input)
   static typename OutputArray::ValueType
   perform( const InputArray& input,
            OutputArray& output,
            typename InputArray::IndexType begin,
            typename InputArray::IndexType end,
            typename OutputArray::IndexType outputBegin,
            Reduction&& reduction,
            typename OutputArray::ValueType identity );

   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static auto
   performFirstPhase( const InputArray& input,
                      OutputArray& output,
                      typename InputArray::IndexType begin,
                      typename InputArray::IndexType end,
                      typename OutputArray::IndexType outputBegin,
                      Reduction&& reduction,
                      typename OutputArray::ValueType identity );

   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( const InputArray& input,
                       OutputArray& output,
                       const BlockShifts& blockShifts,
                       typename InputArray::IndexType begin,
                       typename InputArray::IndexType end,
                       typename OutputArray::IndexType outputBegin,
                       Reduction&& reduction,
                       typename OutputArray::ValueType identity,
                       typename OutputArray::ValueType shift );
};

template< ScanType Type, ScanPhaseType PhaseType >
struct Scan< Devices::Host, Type, PhaseType >
{
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static void
   perform( const InputArray& input,
            OutputArray& output,
            typename InputArray::IndexType begin,
            typename InputArray::IndexType end,
            typename OutputArray::IndexType outputBegin,
            Reduction&& reduction,
            typename OutputArray::ValueType identity );

   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static auto
   performFirstPhase( const InputArray& input,
                      OutputArray& output,
                      typename InputArray::IndexType begin,
                      typename InputArray::IndexType end,
                      typename OutputArray::IndexType outputBegin,
                      Reduction&& reduction,
                      typename OutputArray::ValueType identity );

   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( const InputArray& input,
                       OutputArray& output,
                       const BlockShifts& blockShifts,
                       typename InputArray::IndexType begin,
                       typename InputArray::IndexType end,
                       typename OutputArray::IndexType outputBegin,
                       Reduction&& reduction,
                       typename OutputArray::ValueType identity,
                       typename OutputArray::ValueType shift );
};

template< ScanType Type, ScanPhaseType PhaseType >
struct Scan< Devices::Cuda, Type, PhaseType >
{
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static void
   perform( const InputArray& input,
            OutputArray& output,
            typename InputArray::IndexType begin,
            typename InputArray::IndexType end,
            typename OutputArray::IndexType outputBegin,
            Reduction&& reduction,
            typename OutputArray::ValueType identity );

   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static auto
   performFirstPhase( const InputArray& input,
                      OutputArray& output,
                      typename InputArray::IndexType begin,
                      typename InputArray::IndexType end,
                      typename OutputArray::IndexType outputBegin,
                      Reduction&& reduction,
                      typename OutputArray::ValueType identity );

   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( const InputArray& input,
                       OutputArray& output,
                       const BlockShifts& blockShifts,
                       typename InputArray::IndexType begin,
                       typename InputArray::IndexType end,
                       typename OutputArray::IndexType outputBegin,
                       Reduction&& reduction,
                       typename OutputArray::ValueType identity,
                       typename OutputArray::ValueType shift );
};

} // namespace detail
} // namespace Algorithms
} // namespace TNL

#include "Scan.hpp"
