/***************************************************************************
                          DenseMatrixRowView.h -  description
                             -------------------
    begin                : Jan 3, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {

template< typename SegmentView,
          typename ValuesView >
class DenseMatrixRowView
{
   public:

      using RealType = typename ValuesView::RealType;
      using SegmentViewType = SegmentView;
      using IndexType = typename SegmentViewType::IndexType;
      using ValuesViewType = ValuesView;

      __cuda_callable__
      DenseMatrixRowView( const SegmentViewType& segmentView,
                          const ValuesViewType& values );

      __cuda_callable__
      IndexType getSize() const;

      __cuda_callable__
      const RealType& getValue( const IndexType column ) const;

      __cuda_callable__
      RealType& getValue( const IndexType column );

      __cuda_callable__
      void setElement( const IndexType column,
                       const RealType& value );
   protected:

      SegmentViewType segmentView;

      ValuesViewType values;
};
   } // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/DenseMatrixRowView.hpp>
