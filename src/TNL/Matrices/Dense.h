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
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/DenseRow.h>
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
   using CompressedRowLengthsVector = typename Matrix< Real, Device, Index >::CompressedRowLengthsVector;
   using ConstCompressedRowLengthsVectorView = typename Matrix< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView;
   using BaseType = Matrix< Real, Device, Index >;
   using MatrixRow = DenseRow< Real, Index >;
   using SegmentsType = Containers::Segments::Ellpack< DeviceType, IndexType, typename Allocators::Default< Device >::template Allocator< IndexType >, RowMajorOrder >;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = Dense< _Real, _Device, _Index >;

   Dense();

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

   [[deprecated]]
   IndexType getRowLength( const IndexType row ) const;

   IndexType getMaxRowLength() const;

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   void reset();

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

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   const MatrixRow getRow( const IndexType rowIndex ) const;

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
   Dense& operator=( const Dense& matrix );

   // cross-device copy assignment
   template< typename Real2, typename Device2, typename Index2,
             typename = typename Enabler< Device2 >::type >
   Dense& operator=( const Dense< Real2, Device2, Index2 >& matrix );

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
