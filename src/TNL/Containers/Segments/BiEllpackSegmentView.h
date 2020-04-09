/***************************************************************************
                          BiEllpackSegmentView.h -  description
                             -------------------
    begin                : Apr 7, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Index,
          bool RowMajorOrder = false,
          int WarpSize = 32 >
class BiEllpackSegmentView
{
   public:
      
      static constexpr int getWarpSize() { return WarpSize; };

      static constexpr int getLogWarpSize() { return std::log2( WarpSize ); };

      static constexpr int getGroupsCount() { return getLogWarpSize() + 1; };

      using IndexType = Index;
      using GroupsWidthType = Containers::StaticVector< getGroupsCount(), IndexType >;


      /**
       * \brief Constructor.
       * 
       * \param offset is offset of the first group of the strip the segment belongs to.
       * \param size is the segment size
       * \param inStripIdx is index of the segment within its strip.
       * \param groupsWidth is a static vector containing widths of the strip groups
       */
      __cuda_callable__
      BiEllpackSegmentView( const IndexType offset,
                            const IndexType inStripIdx,
                            const GroupsWidthType& groupsWidth )
      : groupOffset( offset ), segmentSize( TNL::sum( groupsWidth ) ), inStripIdx( inStripIdx ), groupsWidth( groupsWidth ){};

      __cuda_callable__
      IndexType getSize() const
      {
         return this->segmentSize;
      };

      __cuda_callable__
      IndexType getGlobalIndex( IndexType localIdx ) const
      {
         IndexType i( 0 ), offset( groupOffset ), groupHeight( getWarpSize() );
         while( localIdx > groupsWidth[ i ] )
         {
            localIdx -= groupsWidth[ i ];
            offset += groupsWidth[ i++ ] * groupHeight;
            groupHeight /= 2;
         }
         TNL_ASSERT_LE( i, TNL::log2( getWarpSize() - inStripIdx + 1 ), "Local index exceeds segment bounds." );
         if( RowMajorOrder )
            return offset + inStripIdx * groupsWidth[ i ] + localIdx;
         else
            return offset + inStripIdx + localIdx * groupHeight;
      };

      protected:

         IndexType groupOffset, inStripIdx, segmentSize;

         GroupsWidthType groupsWidth;
};

      } //namespace Segments
   } //namespace Containers
} //namespace TNL
