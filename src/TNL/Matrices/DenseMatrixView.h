/***************************************************************************
                          DenseMatrixView.h  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrixRowView.h>
#include <TNL/Matrices/MatrixView.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
namespace Matrices {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value >
class DenseMatrixView : public MatrixView< Real, Device, Index >
{
   private:
      // convenient template alias for controlling the selection of copy-assignment operator
      template< typename Device2 >
      using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

      // friend class will be needed for templated assignment operators
      //template< typename Real2, typename Device2, typename Index2 >
      //friend class Dense;

   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;
      using BaseType = Matrix< Real, Device, Index >;
      using ValuesVectorType = typename BaseType::ValuesVectorType;
      using ValuesViewType = typename ValuesVectorType::ViewType;
      using SegmentsType = Containers::Segments::Ellpack< DeviceType, IndexType, typename Allocators::Default< Device >::template Allocator< IndexType >, RowMajorOrder, 1 >;
      using SegmentsViewType = typename SegmentsType::ViewType;
      using SegmentViewType = typename SegmentsType::SegmentViewType;
      using RowView = DenseMatrixRowView< SegmentViewType, ValuesViewType >;
      using ViewType = DenseMatrixView< Real, Device, Index, RowMajorOrder >;
      using ConstViewType = DenseMatrixView< typename std::add_const< Real >::type, Device, Index, RowMajorOrder >;


      // TODO: remove this
      using CompressedRowLengthsVector = typename Matrix< Real, Device, Index >::CompressedRowLengthsVector;
      using ConstCompressedRowLengthsVectorView = typename Matrix< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView;

      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index >
      using Self = DenseMatrixView< _Real, _Device, _Index >;

      __cuda_callable__
      DenseMatrixView();

      __cuda_callable__
      DenseMatrixView( const IndexType rows,
                       const IndexType columns,
                       const ValuesViewType& values );

      __cuda_callable__
      DenseMatrixView( const DenseMatrixView& m ) = default;

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      IndexType getElementsCount() const;

      IndexType getNonzeroElementsCount() const;

      void reset();

      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );


      void setValue( const RealType& v );

      __cuda_callable__
      Real& operator()( const IndexType row,
                        const IndexType column );

      __cuda_callable__
      const Real& operator()( const IndexType row,
                              const IndexType column ) const;

      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      Real getElement( const IndexType row,
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

      template< typename InVector, typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const;

      template< typename Matrix >
      void addMatrix( const Matrix& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Matrix1, typename Matrix2, int tileDim = 32 >
      void getMatrixProduct( const Matrix1& matrix1,
                          const Matrix2& matrix2,
                          const RealType& matrix1Multiplicator = 1.0,
                          const RealType& matrix2Multiplicator = 1.0 );

      template< typename Matrix, int tileDim = 32 >
      void getTransposition( const Matrix& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      DenseMatrixView& operator=( const DenseMatrixView& matrix );

      void save( const String& fileName ) const;

      void save( File& file ) const;

      void print( std::ostream& str ) const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType column ) const;

      //typedef DenseDeviceDependentCode< DeviceType > DeviceDependentCode;
      //friend class DenseDeviceDependentCode< DeviceType >;

      SegmentsViewType segments;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/DenseMatrixView.hpp>
