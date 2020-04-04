 /***************************************************************************
                          SparseMatrixRowView.h -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
class SparseMatrixRowView
{
   public:

      using RealType = typename ValuesView::RealType;
      using SegmentViewType = SegmentView;
      using IndexType = typename SegmentViewType::IndexType;
      using ValuesViewType = ValuesView;
      using ColumnsIndexesViewType = ColumnsIndexesView;
      using ConstValuesViewType = typename ValuesViewType::ConstViewType;
      using ConstColumnsIndexesViewType = typename ColumnsIndexesViewType::ConstViewType;
      using ConstViewType = SparseMatrixRowView< SegmentView, ConstValuesViewType, ConstColumnsIndexesViewType, isBinary_ >;

      static constexpr bool isBinary() { return isBinary_; };

      __cuda_callable__
      SparseMatrixRowView( const SegmentViewType& segmentView,
                           const ValuesViewType& values,
                           const ColumnsIndexesViewType& columnIndexes );

      __cuda_callable__
      IndexType getSize() const;

      __cuda_callable__
      const IndexType& getColumnIndex( const IndexType localIdx ) const;

      __cuda_callable__
      IndexType& getColumnIndex( const IndexType localIdx );

      __cuda_callable__
      const RealType& getValue( const IndexType localIdx ) const;

      __cuda_callable__
      RealType& getValue( const IndexType localIdx );

      __cuda_callable__
      void setElement( const IndexType localIdx,
                       const IndexType column,
                       const RealType& value );
   protected:

      SegmentViewType segmentView;

      ValuesViewType values;

      ColumnsIndexesViewType columnIndexes;
};
   } // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SparseMatrixRowView.hpp>
