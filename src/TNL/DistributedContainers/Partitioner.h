/***************************************************************************
                          Partitioner.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "Subrange.h"

#include <TNL/Math.h>

namespace TNL {
namespace DistributedContainers {

template< typename Index, typename Communicator >
class Partitioner
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
public:
   using SubrangeType = Subrange< Index >;

   static SubrangeType splitRange( Index globalSize, CommunicationGroup group )
   {
      if( group != Communicator::NullGroup ) {
         const int rank = Communicator::GetRank( group );
         const int partitions = Communicator::GetSize( group );
         const Index begin = min( globalSize, rank * globalSize / partitions );
         const Index end = min( globalSize, (rank + 1) * globalSize / partitions );
         return SubrangeType( begin, end );
      }
      else
         return SubrangeType( 0, 0 );
   }

   // Gets the owner of given global index.
   __cuda_callable__
   static int getOwner( Index i, Index globalSize, int partitions )
   {
      int owner = i * partitions / globalSize;
      if( owner < partitions - 1 && i >= getOffset( globalSize, owner + 1, partitions ) )
         owner++;
      TNL_ASSERT_GE( i, getOffset( globalSize, owner, partitions ), "BUG in getOwner" );
      TNL_ASSERT_LT( i, getOffset( globalSize, owner + 1, partitions ), "BUG in getOwner" );
      return owner;
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

// TODO:
// - partitioner in deal.II stores also ghost indices:
//   https://www.dealii.org/8.4.0/doxygen/deal.II/classUtilities_1_1MPI_1_1Partitioner.html
// - ghost indices are stored in a general IndexMap class (based on collection of subranges):
//   https://www.dealii.org/8.4.0/doxygen/deal.II/classIndexSet.html

} // namespace DistributedContainers
} // namespace TNL
