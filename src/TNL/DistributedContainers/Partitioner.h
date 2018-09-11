/***************************************************************************
                          DistributedArray.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "IndexMap.h"

#include <TNL/Math.h>

namespace TNL {
namespace DistributedContainers {

template< typename IndexMap, typename Communicator >
class Partitioner
{};

template< typename Index, typename Communicator >
class Partitioner< Subrange< Index >, Communicator >
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
public:
   using IndexMap = Subrange< Index >;

   static IndexMap splitRange( Index globalSize, CommunicationGroup group )
   {
      if( group != Communicator::NullGroup ) {
         const int rank = Communicator::GetRank( group );
         const int partitions = Communicator::GetSize( group );
         const Index begin = min( globalSize, rank * globalSize / partitions );
         const Index end = min( globalSize, (rank + 1) * globalSize / partitions );
         return IndexMap( begin, end, globalSize );
      }
      else
         return IndexMap( 0, 0, globalSize );
   }

   // Gets the owner of given global index.
   __cuda_callable__
   static int getOwner( Index i, Index globalSize, int partitions )
   {
      return i * partitions / globalSize;
   }

   // Gets the offset of data for given rank.
   __cuda_callable__
   static Index getOffset( Index globalSize, int rank, int partitions )
   {
      return rank * globalSize / partitions;
   }

   // Gets the size of data assigned to given rank.
   __cuda_callable__
   static Index getSizeForRank( Index globalSize, int rank, int partitions )
   {
      const Index begin = min( globalSize, rank * globalSize / partitions );
      const Index end = min( globalSize, (rank + 1) * globalSize / partitions );
      return end - begin;
   }
};

} // namespace DistributedContainers
} // namespace TNL
