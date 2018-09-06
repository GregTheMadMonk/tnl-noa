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
};

} // namespace DistributedContainers
} // namespace TNL
