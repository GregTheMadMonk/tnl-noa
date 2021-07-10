/***************************************************************************
                          SegmentedScan.hpp  -  description
                             -------------------
    begin                : Mar 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "SegmentedScan.h"

#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Algorithms {

template< ScanType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedScan< Devices::Sequential, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::ValueType zero )
{
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   if( Type == ScanType::Inclusive )
   {
      for( IndexType i = begin + 1; i < end; i++ )
         if( ! flags[ i ] )
            v[ i ] = reduction( v[ i ], v[ i - 1 ] );
   }
   else // Exclusive scan
   {
      ValueType aux( v[ begin ] );
      v[ begin ] = zero;
      for( IndexType i = begin + 1; i < end; i++ )
      {
         ValueType x = v[ i ];
         if( flags[ i ] )
            aux = zero;
         v[ i ] = aux;
         aux = reduction( aux, x );
      }
   }
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedScan< Devices::Host, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::ValueType zero )
{
#ifdef HAVE_OPENMP
   // TODO: parallelize with OpenMP
   SegmentedScan< Devices::Sequential, Type >::perform( v, flags, begin, end, reduction, zero );
#else
   SegmentedScan< Devices::Sequential, Type >::perform( v, flags, begin, end, reduction, zero );
#endif
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedScan< Devices::Cuda, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::ValueType zero )
{
#ifdef HAVE_CUDA
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   throw Exceptions::NotImplementedError( "Segmented scan (prefix sum) is not implemented for CUDA." );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Algorithms
} // namespace TNL
