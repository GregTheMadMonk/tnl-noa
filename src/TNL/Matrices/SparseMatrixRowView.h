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

template< typename Real,
          typename SegmentView >
class SparseMatrixRowView
{
   public:

      using RealType = Real;
      using SegmentViewType = SegmentView;
      using DeviceType = typename SegmentViewType::DeviceType;
      using IndexType = typename SegmentViewType::IndexType;
      using ValuesView = Containers::VectorView< RealType, DeviceType, IndexType >;
      using ColumnIndexesView = Containers::VectorView< IndexType, DeviceType, IndexType >;

      __cuda_callable__
      SparseMatrixRowView( const SegmentView& segmentView,
                           const ValuesView& values,
                           const ColumnIndexesView& columnIndexes );

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

      SegmentView segmentView;

      ValuesView values;

      ColumnIndexesView columnIndexes;
};
   } // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SparseMatrixRowView.hpp>
