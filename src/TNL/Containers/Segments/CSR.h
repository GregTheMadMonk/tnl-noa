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
      IndexType getSize() const;

      IndexType getStorageSize() const;

      IndexType getGlobalIndex( const Index segmentIdx, const Index localIdx ) const;

      void getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const;

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f' with arguments 'args'
       */
      template< typename Function, typename... Args >
      void forAll( Function& f, Args... args ) const;

      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename... Args >
      void segmentsReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, Args... args );

   protected:

      OffsetsHolder offsets;

};

      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL

#include <TNL/Containers/Segments/CSR.h>
