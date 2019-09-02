/***************************************************************************
                          Scan.h  -  description
                             -------------------
    begin                : Aug 16, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Containers/Algorithms/Scan.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< ScanType Type >
struct DistributedScan
{
   template< typename DistributedVector,
             typename Reduction >
   static void
   perform( DistributedVector& v,
            typename DistributedVector::IndexType begin,
            typename DistributedVector::IndexType end,
            const Reduction& reduction,
            const typename DistributedVector::RealType zero )
   {
      using RealType = typename DistributedVector::RealType;
      using DeviceType = typename DistributedVector::DeviceType;
      using CommunicatorType = typename DistributedVector::CommunicatorType;

      const auto group = v.getCommunicationGroup();
      if( group != CommunicatorType::NullGroup ) {
         // adjust begin and end for the local range
         const auto localRange = v.getLocalRange();
         begin = min( max( begin, localRange.getBegin() ), localRange.getEnd() ) - localRange.getBegin();
         end = max( min( end, localRange.getEnd() ), localRange.getBegin() ) - localRange.getBegin();

         // perform first phase on the local data
         auto localView = v.getLocalView();
         const auto blockShifts = Scan< DeviceType, Type >::performFirstPhase( localView, begin, end, reduction, zero );
         const RealType localSum = blockShifts.getElement( blockShifts.getSize() - 1 );

         // exchange local sums between ranks
         const int nproc = CommunicatorType::GetSize( group );
         RealType dataForScatter[ nproc ];
         for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = localSum;
         Vector< RealType, Devices::Host > rankSums( nproc );
         // NOTE: exchanging general data types does not work with MPI
         CommunicatorType::Alltoall( dataForScatter, 1, rankSums.getData(), 1, group );

         // compute prefix-sum of the per-rank sums
         Scan< Devices::Host, ScanType::Exclusive >::perform( rankSums, 0, nproc, reduction, zero );

         // perform second phase: shift by the per-block and per-rank offsets
         const int rank = CommunicatorType::GetRank( group );
         Scan< DeviceType, Type >::performSecondPhase( localView, blockShifts, begin, end, reduction, rankSums[ rank ] );
      }
   }
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
