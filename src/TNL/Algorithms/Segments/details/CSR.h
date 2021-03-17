/***************************************************************************
                          CSR.h -  description
                             -------------------
    begin                : Dec 12, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once


namespace TNL {
   namespace Algorithms {
      namespace Segments {
         namespace details {

template< typename Device,
          typename Index >
class CSR
{
   public:

      using DeviceType = Device;
      using IndexType = Index;

      template< typename SizesHolder, typename CSROffsets >
      static void setSegmentsSizes( const SizesHolder& sizes, CSROffsets& offsets )
      {
         offsets.setSize( sizes.getSize() + 1 );
         // GOTCHA: when sizes.getSize() == 0, getView returns a full view with size == 1
         if( sizes.getSize() > 0 ) {
            auto view = offsets.getView( 0, sizes.getSize() );
            view = sizes;
         }
         offsets.setElement( sizes.getSize(), 0 );
         offsets.template scan< Algorithms::ScanType::Exclusive >();
      }

      template< typename CSROffsets >
      __cuda_callable__
      static IndexType getSegmentsCount( const CSROffsets& offsets )
      {
         return offsets.getSize() - 1;
      }

      /***
       * \brief Returns size of the segment number \r segmentIdx
       */
      template< typename CSROffsets >
      __cuda_callable__
      static IndexType getSegmentSize( const CSROffsets& offsets, const IndexType segmentIdx )
      {
         if( ! std::is_same< DeviceType, Devices::Host >::value )
         {
#ifdef __CUDA_ARCH__
            return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
#else
            return offsets.getElement( segmentIdx + 1 ) - offsets.getElement( segmentIdx );
#endif
         }
         return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
      }

      /***
       * \brief Returns number of elements that needs to be allocated.
       */
      template< typename CSROffsets >
      __cuda_callable__
      static IndexType getStorageSize( const CSROffsets& offsets )
      {
         if( ! std::is_same< DeviceType, Devices::Host >::value )
         {
#ifdef __CUDA_ARCH__
            return offsets[ getSegmentsCount( offsets ) ];
#else
            return offsets.getElement( getSegmentsCount( offsets ) );
#endif
         }
         return offsets[ getSegmentsCount( offsets ) ];
      }

      __cuda_callable__
      IndexType getGlobalIndex( const Index segmentIdx, const Index localIdx ) const;

      __cuda_callable__
      void getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const;

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f' with arguments 'args'. The return type of 'f' is bool.
       * When its true, the for-loop continues. Once 'f' returns false, the for-loop
       * is terminated.
       */
      template< typename Function, typename... Args >
      void forElements( IndexType first, IndexType last, Function& f, Args... args ) const;

      template< typename Function, typename... Args >
      void forAllElements( Function& f, Args... args ) const;


      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;
};
         } // namespace details
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
