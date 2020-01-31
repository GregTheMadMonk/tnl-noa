/***************************************************************************
                          SparseMatrix.h -  description
                             -------------------
    begin                : Nov 29, 2019
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
#include <TNL/Matrices/SparseMatrixView.h>
#include <TNL/Matrices/Dense.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename MatrixType = GeneralMatrix,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments = Containers::Segments::CSR,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class SparseMatrix : public Matrix< Real, Device, Index, RealAllocator >
{
   public:
      static constexpr bool isSymmetric() { return MatrixType::isSymmetric(); };
      static constexpr bool isBinary() { return std::is_same< Real, bool >::value; };

      using RealType = Real;
      template< typename Device_, typename Index_, typename IndexAllocator_ >
      using SegmentsTemplate = Segments< Device_, Index_, IndexAllocator_ >;
      using SegmentsType = Segments< Device, Index, IndexAllocator >;
      template< typename Device_, typename Index_ >
      using SegmentsViewTemplate = typename SegmentsType::template ViewTemplate< Device_, Index >;
      using SegmentViewType = typename SegmentsType::SegmentViewType;
      using DeviceType = Device;
      using IndexType = Index;
      using RealAllocatorType = RealAllocator;
      using IndexAllocatorType = IndexAllocator;
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using RowsCapacitiesType = Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType >;
      using RowsCapacitiesView = Containers::VectorView< IndexType, DeviceType, IndexType >;
      using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;
      using ValuesVectorType = typename Matrix< Real, Device, Index, RealAllocator >::ValuesVectorType;
      using ValuesViewType = typename ValuesVectorType::ViewType;
      using ColumnsIndexesVectorType = Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType >;
      using ColumnsIndexesViewType = typename ColumnsIndexesVectorType::ViewType;
      using ViewType = SparseMatrixView< Real, Device, Index, MatrixType, SegmentsViewTemplate >;
      using ConstViewType = SparseMatrixView< typename std::add_const< Real >::type, Device, Index, MatrixType, SegmentsViewTemplate >;
      using RowView = SparseMatrixRowView< SegmentViewType, ValuesViewType, ColumnsIndexesViewType, isBinary() >;

      // TODO: remove this - it is here only for compatibility with original matrix implementation
      typedef Containers::Vector< IndexType, DeviceType, IndexType > CompressedRowLengthsVector;
      typedef Containers::VectorView< IndexType, DeviceType, IndexType > CompressedRowLengthsVectorView;
      typedef typename CompressedRowLengthsVectorView::ConstViewType ConstCompressedRowLengthsVectorView;

      SparseMatrix( const RealAllocatorType& realAllocator = RealAllocatorType(),
                    const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

      SparseMatrix( const SparseMatrix& m );

      SparseMatrix( const SparseMatrix&& m );

      SparseMatrix( const IndexType rows,
                    const IndexType columns,
                    const RealAllocatorType& realAllocator = RealAllocatorType(),
                    const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

      ViewType getView() const; // TODO: remove const

      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      template< typename RowsCapacitiesVector >
      void setCompressedRowLengths( const RowsCapacitiesVector& rowCapacities );

      // TODO: Remove this when possible
      void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths ) {
         this->setCompressedRowLengths( rowLengths );
      };

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      virtual IndexType getRowLength( const IndexType row ) const { return 0;};

      template< typename Matrix >
      void setLike( const Matrix& matrix );

      IndexType getNumberOfNonzeroMatrixElements() const;

      void reset();

      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator );

      RealType getElement( const IndexType row,
                           const IndexType column ) const;

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

      /*template< typename Real2, typename Index2 >
      void addMatrix( const SparseMatrix< Real2, Segments, Device, Index2 >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const SparseMatrix< Real2, Segments, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );
       */

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

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

      /**
       * \brief Assignment of exactly the same matrix type.
       * @param matrix
       * @return
       */
      SparseMatrix& operator=( const SparseMatrix& matrix );

      /**
       * \brief Assignment of dense matrix
       */
      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder, typename RealAllocator_ >
      SparseMatrix& operator=( const Dense< Real_, Device_, Index_, RowMajorOrder, RealAllocator_ >& matrix );


      /**
       * \brief Assignment of any other matrix type.
       * @param matrix
       * @return
       */
      template< typename RHSMatrix >
      SparseMatrix& operator=( const RHSMatrix& matrix );

      void save( File& file ) const;

      void load( File& file );

      void save( const String& fileName ) const;

      void load( const String& fileName );

      void print( std::ostream& str ) const;

      __cuda_callable__
      IndexType getPaddingIndex() const;

// TODO: restore it and also in Matrix
//   protected:

      ColumnsIndexesVectorType columnIndexes;

      SegmentsType segments;

      IndexAllocator indexAllocator;

      //RealAllocator realAllocator;

      ViewType view;
};

}  // namespace Conatiners
} // namespace TNL

#include <TNL/Matrices/SparseMatrix.hpp>
