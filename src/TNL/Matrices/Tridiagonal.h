/***************************************************************************
                          Tridiagonal.h  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/TridiagonalMatrixRowView.h>
#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Matrices/details/TridiagonalMatrixIndexer.h>
#include <TNL/Matrices/TridiagonalMatrixView.h>

namespace TNL {
namespace Matrices {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real > >
class Tridiagonal : public Matrix< Real, Device, Index, RealAllocator >
{
   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;
      using RealAllocatorType = RealAllocator;
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using IndexerType = details::TridiagonalMatrixIndexer< IndexType, RowMajorOrder >;
      using ValuesHolderType = typename BaseType::ValuesHolderType;
      using ValuesViewType = typename ValuesHolderType::ViewType;
      using ViewType = TridiagonalMatrixView< Real, Device, Index, RowMajorOrder >;
      using ConstViewType = TridiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, RowMajorOrder >;
      using RowView = TridiagonalMatrixRowView< ValuesViewType, IndexerType >;

      // TODO: remove this - it is here only for compatibility with original matrix implementation
      typedef Containers::Vector< IndexType, DeviceType, IndexType > CompressedRowLengthsVector;
      typedef Containers::VectorView< IndexType, DeviceType, IndexType > CompressedRowLengthsVectorView;
      typedef typename CompressedRowLengthsVectorView::ConstViewType ConstCompressedRowLengthsVectorView;

      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index >
      using Self = Tridiagonal< _Real, _Device, _Index >;

      static constexpr bool getRowMajorOrder() { return RowMajorOrder; };

      Tridiagonal();

      Tridiagonal( const IndexType rows, const IndexType columns );

      ViewType getView() const; // TODO: remove const

      //ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      void setDimensions( const IndexType rows,
                          const IndexType columns );

      //template< typename Vector >
      void setCompressedRowLengths( const ConstCompressedRowLengthsVectorView rowCapacities );

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
      void setLike( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& m );

      IndexType getNumberOfNonzeroMatrixElements() const;

      void reset();

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
      bool operator == ( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix ) const;

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
      bool operator != ( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix ) const;

      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      void setValue( const RealType& v );

      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      RealType getElement( const IndexType row,
                           const IndexType column ) const;

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

      template< typename Vector >
      __cuda_callable__
      typename Vector::RealType rowVectorProduct( const IndexType row,
                                                  const Vector& vector ) const;

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const;

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
      void addMatrix( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const Tridiagonal< Real2, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      __cuda_callable__
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      // copy assignment
      Tridiagonal& operator=( const Tridiagonal& matrix );

      // cross-device copy assignment
      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
      Tridiagonal& operator=( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix );

      void save( File& file ) const;

      void load( File& file );

      void save( const String& fileName ) const;

      void load( const String& fileName );

      void print( std::ostream& str ) const;

      const IndexerType& getIndexer() const;

      IndexerType& getIndexer();

      __cuda_callable__
      IndexType getPaddingIndex() const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType localIdx ) const;

      IndexerType indexer;

      ViewType view;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Tridiagonal.hpp>
