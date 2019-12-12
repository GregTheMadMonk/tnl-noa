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
#include <TNL/Allocators/Default.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments,
          typename Device = Devices::Host,
          typename Index = int,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class SparseMatrix : public Matrix< Real, Device, Index, RealAllocator >
{
   public:

      using RealType = Real;
      template< typename Device_, typename Index_, typename IndexAllocator_ >
      using SegmentsTemplate = Segments< Device_, Index_, IndexAllocator_ >;
      using SegmentsType = Segments< Device, Index, IndexAllocator >;
      using DeviceType = Device;
      using IndexType = Index;
      using RealAllocatorType = RealAllocator;
      using IndexAllocatorType = IndexAllocator;
      using RowsCapacitiesType = Containers::Vector< IndexType, DeviceType, IndexType >;
      using RowsCapacitiesView = Containers::VectorView< IndexType, DeviceType, IndexType >;
      using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;
      using ValuesVectorType = typename Matrix< Real, Device, Index, RealAllocator >::ValuesVector;
      using ColumnsVectorType = Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType >;

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

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      IndexType getRowLength( const IndexType row ) const;

      __cuda_callable__
      IndexType getRowLengthFast( const IndexType row ) const;

      IndexType getNonZeroRowLength( const IndexType row ) const;

      __cuda_callable__
      IndexType getNonZeroRowLengthFast( const IndexType row ) const;

      template< typename Real2, template< typename, typename, typename > class Segments2, typename Device2, typename Index2, typename RealAllocator2, typename IndexAllocator2 >
      void setLike( const SparseMatrix< Real2, Segments2, Device2, Index2, RealAllocator2, IndexAllocator2 >& matrix );

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
      void forAllRows( Function& function ) const;

      template< typename Vector1, typename Vector2 >
      bool performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      // copy assignment
      SparseMatrix& operator=( const SparseMatrix& matrix );

      // cross-device copy assignment
      template< typename Real2,
                template< typename, typename, typename > class Segments2,
                typename Device2,
                typename Index2,
                typename RealAllocator2,
                typename IndexAllocator2 >
      SparseMatrix& operator=( const SparseMatrix< Real2, Segments2, Device2, Index2, RealAllocator2, IndexAllocator2 >& matrix );

      void save( File& file ) const;

      void load( File& file );

      void save( const String& fileName ) const;

      void load( const String& fileName );

      void print( std::ostream& str ) const;

      __cuda_callable__
      IndexType getPaddingIndex() const;

// TODO: restore it and also in Matrix
//   protected:

      ColumnsVectorType columnIndexes;

      SegmentsType segments;

      IndexAllocator indexAlloctor;

      RealAllocator realAllocator;


};

}  // namespace Conatiners
} // namespace TNL

#include <TNL/Matrices/SparseMatrix.hpp>
