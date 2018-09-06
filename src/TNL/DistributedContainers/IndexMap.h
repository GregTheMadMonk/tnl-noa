/***************************************************************************
                          IndexMap.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Assert.h>
#include <TNL/String.h>
#include <TNL/param-types.h>

namespace TNL {
namespace DistributedContainers {

// Specifies a subrange [begin, end) of a range [0, gloablSize).
template< typename Index >
class Subrange
{
public:
   using IndexType = Index;

   __cuda_callable__
   Subrange() = default;

   __cuda_callable__
   Subrange( Index begin, Index end, Index globalSize )
   {
      setSubrange( begin, end, globalSize );
   }

   // Sets the local subrange and global range size.
   __cuda_callable__
   void setSubrange( Index begin, Index end, Index globalSize )
   {
      TNL_ASSERT_LE( begin, end, "begin must be before end" );
      TNL_ASSERT_GE( begin, 0, "begin must be non-negative" );
      TNL_ASSERT_LE( end - begin, globalSize, "end of the subrange is outside of gloabl range" );
      offset = begin;
      localSize = end - begin;
      this->globalSize = globalSize;
   }

   __cuda_callable__
   void reset()
   {
      offset = 0;
      localSize = 0;
      globalSize = 0;
   }

   static String getType()
   {
      return "Subrange< " + TNL::getType< Index >() + " >";
   }

   // Checks if a global index is in the set of local indices.
   __cuda_callable__
   bool isLocal( Index i ) const
   {
      return offset <= i && i < offset + localSize;
   }

   // Gets the offset of the subrange.
   __cuda_callable__
   Index getOffset() const
   {
      return offset;
   }

   // Gets number of local indices.
   __cuda_callable__
   Index getLocalSize() const
   {
      return localSize;
   }

   // Gets number of global indices.
   __cuda_callable__
   Index getGlobalSize() const
   {
      return globalSize;
   }

   // Gets local index for given global index.
   __cuda_callable__
   Index getLocalIndex( Index i ) const
   {
      TNL_ASSERT_TRUE( isLocal( i ), "Given global index was not found in the local index set." );
      return i - offset;
   }

   // Gets global index for given local index.
   __cuda_callable__
   Index getGlobalIndex( Index i ) const
   {
      TNL_ASSERT_GE( i, 0, "Given local index was not found in the local index set." );
      TNL_ASSERT_LT( i, localSize, "Given local index was not found in the local index set." );
      return i + offset;
   }

   bool operator==( const Subrange& other ) const
   {
      return offset == other.offset &&
             localSize == other.localSize &&
             globalSize == other.globalSize;
   }

   bool operator!=( const Subrange& other ) const
   {
      return ! (*this == other);
   }

protected:
   Index offset = 0;
   Index localSize = 0;
   Index globalSize = 0;
};

// TODO: implement a general IndexMap class, e.g. based on collection of subranges as in deal.II:
// https://www.dealii.org/8.4.0/doxygen/deal.II/classIndexSet.html

} // namespace DistributedContainers
} // namespace TNL
