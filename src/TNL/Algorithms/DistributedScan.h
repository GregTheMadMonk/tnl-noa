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
            const typename DistributedVector::ValueType zero )
   {
      using ValueType = typename DistributedVector::ValueType;
      using DeviceType = typename DistributedVector::DeviceType;

      const auto group = v.getCommunicationGroup();
      if( group != MPI::NullGroup() ) {
         // adjust begin and end for the local range
         const auto localRange = v.getLocalRange();
         begin = min( max( begin, localRange.getBegin() ), localRange.getEnd() ) - localRange.getBegin();
         end = max( min( end, localRange.getEnd() ), localRange.getBegin() ) - localRange.getBegin();

         // perform first phase on the local data
         auto localView = v.getLocalView();
         const auto block_results = Scan< DeviceType, Type >::performFirstPhase( localView, begin, end, reduction, zero );
         const ValueType local_result = block_results.getElement( block_results.getSize() - 1 );

         // exchange local results between ranks
         const int nproc = MPI::GetSize( group );
         ValueType dataForScatter[ nproc ];
         for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = local_result;
         Containers::Vector< ValueType, Devices::Host > rank_results( nproc );
         // NOTE: exchanging general data types does not work with MPI
         MPI::Alltoall( dataForScatter, 1, rank_results.getData(), 1, group );

         // compute the scan of the per-rank results
         Scan< Devices::Host, ScanType::Exclusive >::perform( rank_results, 0, nproc, reduction, zero );

         // perform the second phase, using the per-block and per-rank results
         const int rank = MPI::GetRank( group );
         Scan< DeviceType, Type >::performSecondPhase( localView, block_results, begin, end, reduction, rank_results[ rank ] );
      }
   }
};

} // namespace Algorithms
} // namespace TNL
