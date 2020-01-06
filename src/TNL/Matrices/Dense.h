/***************************************************************************
                          Dense.h  -  description
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
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
namespace Matrices {

template< typename Device >
class DenseDeviceDependentCode;

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real > >
class Dense : public Matrix< Real, Device, Index >
{
   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;
      using BaseType = Matrix< Real, Device, Index >;
      using ValuesType = typename BaseType::ValuesVector;
      using ValuesViewType = typename ValuesType::ViewType;
      using SegmentsType = Containers::Segments::Ellpack< DeviceType, IndexType, typename Allocators::Default< Device >::template Allocator< IndexType >, RowMajorOrder, 1 >;
      using SegmentViewType = typename SegmentsType::SegmentViewType;
      using ViewType = DenseMatrixView< Real, Device, Index, RowMajorOrder >;
      using ConstViewType = DenseMatrixView< typename std::add_const< Real >::type, Device, Index, RowMajorOrder >;
      using RowView = DenseMatrixRowView< SegmentViewType, ValuesViewType >;

      // TODO: remove this
      using CompressedRowLengthsVector = typename Matrix< Real, Device, Index >::CompressedRowLengthsVector;
      using ConstCompressedRowLengthsVectorView = typename Matrix< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView;

      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index >
      using Self = Dense< _Real, _Device, _Index >;

      Dense();

      Dense( const IndexType rows, const IndexType columns );
      
      ViewType getView();

      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      void setDimensions( const IndexType rows,
                          const IndexType columns );

      template< typename Matrix >
      void setLike( const Matrix& matrix );

      /****
       * This method is only for the compatibility with the sparse matrices.
       */
      void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      IndexType getNumberOfMatrixElements() const;

      IndexType getNumberOfNonzeroMatrixElements() const;

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

      bool setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      bool addElement( const IndexType row,
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

      // copy assignment
      //Dense& operator=( const Dense& matrix );

      // cross-device copy assignment
      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAlocator_ >
      Dense& operator=( const Dense< Real_, Device_, Index_, RowMajorOrder_, RealAlocator_ >& matrix );

      template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
      bool operator==( const Dense< Real_, Device_, Index_, RowMajorOrder >& matrix ) const;

      template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
      bool operator!=( const Dense< Real_, Device_, Index_, RowMajorOrder >& matrix ) const;
      
      void save( const String& fileName ) const;

      void load( const String& fileName );

      void save( File& file ) const;

      void load( File& file );

      void print( std::ostream& str ) const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType column ) const;

      typedef DenseDeviceDependentCode< DeviceType > DeviceDependentCode;
      friend class DenseDeviceDependentCode< DeviceType >;

      SegmentsType segments;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Dense.hpp>
