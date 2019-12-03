/***************************************************************************
                          CSR.h -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index >
class CSR
{
   public:

      using DeviceType = Device;
      using IndexType = Index;
      using OffsetsHolder = Containers::Vector< IndexType, DeviceType, IndexType >;

      CSR();

      CSR( const Vector< IndexType, DeviceType, IndexType >& sizes );

      CSR( const CSR& segments );

      CSR( const CSR&& segments );

      /**
       * \brief Set sizes of particular segmenets.
       */
      template< typename SizesHolder = OffsetsHolder >
      void setSizes( const SizesHolder& sizes );

      /**
       * \brief Number segments.
       */
      __cuda_callable__
      IndexType getSize() const;

      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

      __cuda_callable__
      IndexType getStorageSize() const;

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
      void forSegments( IndexType first, IndexType last, Function& f, Args... args ) const;

      template< typename Function, typename... Args >
      void forAll( Function& f, Args... args ) const;


      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args );


   protected:

      OffsetsHolder offsets;

};

      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL

#include <TNL/Containers/Segments/CSR.hpp>
