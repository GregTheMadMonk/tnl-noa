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

#include "Scan.h"

#include <TNL/Containers/Array.h>
#include <TNL/MPI/Wrappers.h>

namespace TNL {
namespace Algorithms {
namespace detail {

template< ScanType Type >
struct DistributedScan
{
   template< typename InputDistributedArray,
             typename OutputDistributedArray,
             typename Reduction >
   static void
   perform( const InputDistributedArray& input,
            OutputDistributedArray& output,
            typename InputDistributedArray::IndexType begin,
            typename InputDistributedArray::IndexType end,
            Reduction&& reduction,
            typename OutputDistributedArray::ValueType zero )
   {
      using ValueType = typename OutputDistributedArray::ValueType;
      using DeviceType = typename OutputDistributedArray::DeviceType;

      const auto group = input.getCommunicationGroup();
      if( group != MPI::NullGroup() ) {
         // adjust begin and end for the local range
         const auto localRange = input.getLocalRange();
         begin = min( max( begin, localRange.getBegin() ), localRange.getEnd() ) - localRange.getBegin();
         end = max( min( end, localRange.getEnd() ), localRange.getBegin() ) - localRange.getBegin();

         // perform first phase on the local data
         const auto inputLocalView = input.getConstLocalView();
         auto outputLocalView = output.getLocalView();
         const auto block_results = Scan< DeviceType, Type >::performFirstPhase( inputLocalView, outputLocalView, begin, end, begin, reduction, zero );
         const ValueType local_result = block_results.getElement( block_results.getSize() - 1 );

         // exchange local results between ranks
         const int nproc = MPI::GetSize( group );
         ValueType dataForScatter[ nproc ];
         for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = local_result;
         Containers::Array< ValueType, Devices::Host > rank_results( nproc );
         // NOTE: exchanging general data types does not work with MPI
         MPI::Alltoall( dataForScatter, 1, rank_results.getData(), 1, group );

         // compute the scan of the per-rank results
         Scan< Devices::Host, ScanType::Exclusive >::perform( rank_results, rank_results, 0, nproc, 0, reduction, zero );

         // perform the second phase, using the per-block and per-rank results
         const int rank = MPI::GetRank( group );
         Scan< DeviceType, Type >::performSecondPhase( inputLocalView, outputLocalView, block_results, begin, end, begin, reduction, rank_results[ rank ] );
      }
   }
};

} // namespace detail
} // namespace Algorithms
} // namespace TNL
