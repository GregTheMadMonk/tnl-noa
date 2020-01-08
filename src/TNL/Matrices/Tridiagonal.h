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
#include <TNL/Matrices/TridiagonalRow.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
namespace Matrices {

template< typename Device >
class TridiagonalDeviceDependentCode;

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real > >
class Tridiagonal : public Matrix< Real, Device, Index, RealAllocator >
{
   private:
      // convenient template alias for controlling the selection of copy-assignment operator
      template< typename Device2 >
      using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

      // friend class will be needed for templated assignment operators
      template< typename Real2, typename Device2, typename Index2 >
      friend class Tridiagonal;

   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;
      using RealAllocatorType = RealAllocator;
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using ValuesType = typename BaseType::ValuesVector;
      using ValuesViewType = typename ValuesType::ViewType;
      //using ViewType = TridiagonalMatrixView< Real, Device, Index, RowMajorOrder >;
      //using ConstViewType = TridiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, RowMajorOrder >;
      using RowView = TridiagonalMatrixRowView< SegmentViewType, ValuesViewType >;


      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index >
      using Self = Tridiagonal< _Real, _Device, _Index >;

      Tridiagonal();

      Tridiagonal( const IndexType rows, const IndexType columns );

      ViewType getView();

      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      void setDimensions( const IndexType rows,
                          const IndexType columns );

      void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      template< typename Real2, typename Device2, typename Index2 >
      void setLike( const Tridiagonal< Real2, Device2, Index2 >& m );

      IndexType getNumberOfMatrixElements() const;

      IndexType getNumberOfNonzeroMatrixElements() const;

      IndexType getMaxRowlength() const;

      void reset();

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
      bool operator == ( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_ >& matrix ) const;

      template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
      bool operator != ( const Tridiagonal< Real_, Device_, Index_ >& matrix ) const;

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

      template< typename Real2, typename Index2 >
      void addMatrix( const Tridiagonal< Real2, Device, Index2 >& matrix,
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
      template< typename Real2, typename Device2, typename Index2,
                typename = typename Enabler< Device2 >::type >
      Tridiagonal& operator=( const Tridiagonal< Real2, Device2, Index2 >& matrix );

      void save( File& file ) const;

      void load( File& file );

      void save( const String& fileName ) const;

      void load( const String& fileName );

      void print( std::ostream& str ) const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType column ) const;

      Containers::Vector< RealType, DeviceType, IndexType > values;

      typedef TridiagonalDeviceDependentCode< DeviceType > DeviceDependentCode;
      friend class TridiagonalDeviceDependentCode< DeviceType >;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Tridiagonal.hpp>
