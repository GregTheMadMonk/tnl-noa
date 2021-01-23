/***************************************************************************
                          DistributedScan.h  -  description
                             -------------------
    begin                : Aug 16, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Algorithms/Scan.h>
#include <TNL/Containers/Vector.h>
#include <TNL/MPI/Wrappers.h>

namespace TNL {
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

      const auto group = v.getCommunicationGroup();
      if( group != MPI::NullGroup() ) {
         // adjust begin and end for the local range
         const auto localRange = v.getLocalRange();
         begin = min( max( begin, localRange.getBegin() ), localRange.getEnd() ) - localRange.getBegin();
         end = max( min( end, localRange.getEnd() ), localRange.getBegin() ) - localRange.getBegin();

         // perform first phase on the local data
         auto localView = v.getLocalView();
         const auto blockShifts = Scan< DeviceType, Type >::performFirstPhase( localView, begin, end, reduction, zero );
         const RealType localSum = blockShifts.getElement( blockShifts.getSize() - 1 );

         // exchange local sums between ranks
         const int nproc = MPI::GetSize( group );
         RealType dataForScatter[ nproc ];
         for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = localSum;
         Containers::Vector< RealType, Devices::Host > rankSums( nproc );
         // NOTE: exchanging general data types does not work with MPI
         MPI::Alltoall( dataForScatter, 1, rankSums.getData(), 1, group );

         // compute the scan of the per-rank sums
         Scan< Devices::Host, ScanType::Exclusive >::perform( rankSums, 0, nproc, reduction, zero );

         // perform second phase: shift by the per-block and per-rank offsets
         const int rank = MPI::GetRank( group );
         Scan< DeviceType, Type >::performSecondPhase( localView, blockShifts, begin, end, reduction, rankSums[ rank ] );
      }
   }
};

} // namespace Algorithms
} // namespace TNL
