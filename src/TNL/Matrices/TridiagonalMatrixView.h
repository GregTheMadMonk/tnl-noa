/***************************************************************************
                          TridiagonalMatrixView.h  -  description
                             -------------------
    begin                : Jan 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/MatrixView.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/TridiagonalMatrixRowView.h>
#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Matrices/details/TridiagonalMatrixIndexer.h>

namespace TNL {
namespace Matrices {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value >
class TridiagonalMatrixView : public MatrixView< Real, Device, Index >
{
   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;
      using BaseType = MatrixView< Real, Device, Index >;
      using IndexerType = details::TridiagonalMatrixIndexer< IndexType, RowMajorOrder >;
      using ValuesViewType = typename BaseType::ValuesView;
      using ViewType = TridiagonalMatrixView< Real, Device, Index, RowMajorOrder >;
      using ConstViewType = TridiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, RowMajorOrder >;
      using RowView = TridiagonalMatrixRowView< ValuesViewType, IndexerType >;

      // TODO: remove this - it is here only for compatibility with original matrix implementation
      typedef Containers::Vector< IndexType, DeviceType, IndexType > CompressedRowLengthsVector;
      typedef Containers::VectorView< IndexType, DeviceType, IndexType > CompressedRowLengthsVectorView;
      typedef typename CompressedRowLengthsVectorView::ConstViewType ConstCompressedRowLengthsVectorView;

      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index,
                bool RowMajorOrder_ = std::is_same< Device, Devices::Host >::value >
      using Self = TridiagonalMatrixView< _Real, _Device, _Index, RowMajorOrder_ >;

      TridiagonalMatrixView();

      TridiagonalMatrixView( const ValuesViewType& values, const IndexerType& indexer );

      ViewType getView();

      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      void setDimensions( const IndexType rows,
                          const IndexType columns );

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      IndexType getNumberOfNonzeroMatrixElements() const;

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
      bool operator == ( const TridiagonalMatrixView< Real_, Device_, Index_, RowMajorOrder_ >& matrix ) const;

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
      bool operator != ( const TridiagonalMatrixView< Real_, Device_, Index_, RowMajorOrder_ >& matrix ) const;

      RowView getRow( const IndexType& rowIdx );

      const RowView getRow( const IndexType& rowIdx ) const;

      void setValue( const RealType& v );

      bool setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      bool addElement( const IndexType row,
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

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
      void addMatrix( const TridiagonalMatrixView< Real_, Device_, Index_, RowMajorOrder_ >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const TridiagonalMatrixView< Real2, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      __cuda_callable__
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      void save( File& file ) const;

      void save( const String& fileName ) const;

      void print( std::ostream& str ) const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType localIdx ) const;

      IndexerType indexer;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/TridiagonalMatrixView.hpp>
