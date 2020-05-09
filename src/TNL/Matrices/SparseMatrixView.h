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
#include <TNL/Matrices/SparseMatrixRowView.h>

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
      static constexpr bool isSymmetric() { return MatrixType::isSymmetric(); };
      static constexpr bool isBinary() { return MatrixType::isBinary(); };

      using RealType = Real;
      template< typename Device_, typename Index_ >
      using SegmentsViewTemplate = SegmentsView< Device_, Index_ >;
      using SegmentsViewType = SegmentsView< Device, Index >;
      using SegmentViewType = typename SegmentsViewType::SegmentViewType;
      using DeviceType = Device;
      using IndexType = Index;
      using BaseType = MatrixView< Real, Device, Index >;
      using RowsCapacitiesView = Containers::VectorView< IndexType, DeviceType, IndexType >;
      using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;
      using ValuesViewType = typename BaseType::ValuesView;
      using ConstValuesViewType = typename ValuesViewType::ConstViewType;
      using ColumnsIndexesViewType = Containers::VectorView< IndexType, DeviceType, IndexType >;
      using ConstColumnsIndexesViewType = typename ColumnsIndexesViewType::ConstViewType;
      using ViewType = SparseMatrixView< typename std::remove_const< Real >::type, Device, Index, MatrixType, SegmentsViewTemplate >;
      using ConstViewType = SparseMatrixView< typename std::add_const< Real >::type, Device, Index, MatrixType, SegmentsViewTemplate >;
      using RowView = SparseMatrixRowView< SegmentViewType, ValuesViewType, ColumnsIndexesViewType, isBinary() >;
      using ConstRowView = typename RowView::ConstViewType;

      // TODO: remove this - it is here only for compatibility with original matrix implementation
      typedef Containers::Vector< IndexType, DeviceType, IndexType > CompressedRowLengthsVector;
      typedef Containers::VectorView< IndexType, DeviceType, IndexType > CompressedRowLengthsVectorView;
      typedef typename CompressedRowLengthsVectorView::ConstViewType ConstCompressedRowLengthsVectorView;

      __cuda_callable__
      SparseMatrixView();

      __cuda_callable__
      SparseMatrixView( const IndexType rows,
                        const IndexType columns,
                        const ValuesViewType& values,
                        const ColumnsIndexesViewType& columnIndexes,
                        const SegmentsViewType& segments );

      __cuda_callable__
      SparseMatrixView( const SparseMatrixView& m ) = default;

      //__cuda_callable__
      //SparseMatrixView( const SparseMatrixView&& m ) = default;

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      __cuda_callable__
      IndexType getRowCapacity( const IndexType row ) const;

      IndexType getNumberOfNonzeroMatrixElements() const;

      void reset();

      __cuda_callable__
      ConstRowView getRow( const IndexType& rowIdx ) const;

      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      __cuda_callable__
      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      __cuda_callable__
      void addElement( IndexType row,
                       IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      __cuda_callable__
      RealType getElement( IndexType row,
                           IndexType column ) const;

      /*template< typename Vector >
      __cuda_callable__
      typename Vector::RealType rowVectorProduct( const IndexType row,
                                                  const Vector& vector ) const;*/

      /***
       * \brief This method computes outVector = matrixMultiplicator * ( *this ) * inVector + inVectorAddition * inVector
       */
      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType matrixMultiplicator = 1.0,
                          const RealType outVectorMultiplicator = 0.0,
                          const IndexType firstRow = 0,
                          IndexType lastRow = 0 ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function );

      template< typename Function >
      void forAllRows( Function& function ) const;

      template< typename Function >
      void forAllRows( Function& function );

      template< typename Vector1, typename Vector2 >
      bool performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      SparseMatrixView& operator=( const SparseMatrixView& matrix );

      template< typename Matrix >
      bool operator==( const Matrix& m ) const;

      template< typename Matrix >
      bool operator!=( const Matrix& m ) const;

      void save( File& file ) const;

      void save( const String& fileName ) const;

      void print( std::ostream& str ) const;

      __cuda_callable__
      IndexType getPaddingIndex() const;

   protected:

      ColumnsIndexesViewType columnIndexes;

      SegmentsViewType segments;

   private:
      // TODO: this should be probably moved into a detail namespace
      template< typename VectorOrView,
                std::enable_if_t< HasSetSizeMethod< VectorOrView >::value, bool > = true >
      static void set_size_if_resizable( VectorOrView& v, IndexType size )
      {
         v.setSize( size );
      }

      template< typename VectorOrView,
                std::enable_if_t< ! HasSetSizeMethod< VectorOrView >::value, bool > = true >
      static void set_size_if_resizable( VectorOrView& v, IndexType size )
      {
         TNL_ASSERT_EQ( v.getSize(), size, "view has wrong size" );
      }
};

} // namespace Conatiners
} // namespace TNL

#include <TNL/Matrices/SparseMatrixView.hpp>
