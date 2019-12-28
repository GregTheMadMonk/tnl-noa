/***************************************************************************
                          SparseMatrixView.h -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Allocators/Default.h>
#include <TNL/Containers/Segments/CSR.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename MatrixType = GeneralMatrix,
          template< typename Device_, typename Index_ > class SegmentsView = Containers::Segments::CSRView >
class SparseMatrixView : public MatrixView< Real, Device, Index >
{
   public:

      using RealType = Real;
      template< typename Device_, typename Index_ >
      using SegmentsViewTemplate = SegmentsView< Device_, Index_ >;
      using SegmentsViewType = SegmentsView< Device, Index >;
      using DeviceType = Device;
      using IndexType = Index;
      using RowsCapacitiesView = Containers::VectorView< IndexType, DeviceType, IndexType >;
      using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;
      using ValuesViewType = typename MatrixView< Real, Device, Index >::ValuesView;
      using ColumnsViewType = Containers::VectorView< IndexType, DeviceType, IndexType >;

      // TODO: remove this - it is here only for compatibility with original matrix implementation
      typedef Containers::Vector< IndexType, DeviceType, IndexType > CompressedRowLengthsVector;
      typedef Containers::VectorView< IndexType, DeviceType, IndexType > CompressedRowLengthsVectorView;
      typedef typename CompressedRowLengthsVectorView::ConstViewType ConstCompressedRowLengthsVectorView;

      static constexpr bool isSymmetric() { return MatrixType::isSymmetric(); };

      __cuda_callable__
      SparseMatrixView();

      __cuda_callable__
      SparseMatrixView( const IndexType rows,
                        const IndexType columns,
                        ValuesViewType& values,
                        ColumnsViewType& columnIndexes,
                        SegmentsViewType& segments );

      __cuda_callable__
      SparseMatrixView( const SparseMatrixView& m ) = default;

      //__cuda_callable__
      //SparseMatrixView( const SparseMatrixView&& m ) = default;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      IndexType getRowLength( const IndexType row ) const;

      __cuda_callable__
      IndexType getRowLengthFast( const IndexType row ) const;

      IndexType getNonZeroRowLength( const IndexType row ) const;

      __cuda_callable__
      IndexType getNonZeroRowLengthFast( const IndexType row ) const;

      IndexType getNumberOfNonzeroMatrixElements() const;

      void reset();

      __cuda_callable__
      bool setElementFast( const IndexType row,
                           const IndexType column,
                           const RealType& value );

      bool setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      __cuda_callable__
      bool addElementFast( const IndexType row,
                           const IndexType column,
                           const RealType& value,
                           const RealType& thisElementMultiplicator = 1.0 );

      bool addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );


      __cuda_callable__
      bool setRowFast( const IndexType row,
                       const IndexType* columnIndexes,
                       const RealType* values,
                       const IndexType elements );

      bool setRow( const IndexType row,
                   const IndexType* columnIndexes,
                   const RealType* values,
                   const IndexType elements );


      __cuda_callable__
      bool addRowFast( const IndexType row,
                       const IndexType* columns,
                       const RealType* values,
                       const IndexType numberOfElements,
                       const RealType& thisElementMultiplicator = 1.0 );

      bool addRow( const IndexType row,
                   const IndexType* columns,
                   const RealType* values,
                   const IndexType numberOfElements,
                   const RealType& thisElementMultiplicator = 1.0 );


      __cuda_callable__
      RealType getElementFast( const IndexType row,
                               const IndexType column ) const;

      RealType getElement( const IndexType row,
                           const IndexType column ) const;

      __cuda_callable__
      void getRowFast( const IndexType row,
                       IndexType* columns,
                       RealType* values ) const;

      template< typename Vector >
      __cuda_callable__
      typename Vector::RealType rowVectorProduct( const IndexType row,
                                                  const Vector& vector ) const;

      /***
       * \brief This method computes outVector = matrixMultiplicator * ( *this ) * inVector + inVectorAddition * inVector
       */
      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType& matrixMultiplicator = 1.0,
                          const RealType& inVectorAddition = 0.0 ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function ) const;

      template< typename Function >
      void forAllRows( Function& function ) const;

      template< typename Vector1, typename Vector2 >
      bool performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      void save( File& file ) const;

      void save( const String& fileName ) const;

      void print( std::ostream& str ) const;

      __cuda_callable__
      IndexType getPaddingIndex() const;

   protected:

      ColumnsViewType columnIndexes;

      SegmentsViewType segments;
};

}  // namespace Conatiners
} // namespace TNL

#include <TNL/Matrices/SparseMatrixView.hpp>
