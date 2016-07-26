/***************************************************************************
                          tnlTridiagonalMatrix.h  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/matrices/tnlMatrix.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/matrices/tnlTridiagonalMatrixRow.h>

namespace TNL {

template< typename Device >
class tnlTridiagonalMatrixDeviceDependentCode;

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlTridiagonalMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlMatrix< Real, Device, Index >::CompressedRowsLengthsVector CompressedRowsLengthsVector;
   typedef tnlTridiagonalMatrix< Real, Device, Index > ThisType;
   typedef tnlTridiagonalMatrix< Real, tnlHost, Index > HostType;
   typedef tnlTridiagonalMatrix< Real, tnlCuda, Index > CudaType;
   typedef tnlMatrix< Real, Device, Index > BaseType;
   typedef tnlTridiagonalMatrixRow< Real, Index > MatrixRow;

   tnlTridiagonalMatrix();

   static String getType();

   String getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   IndexType getMaxRowLength() const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& m );

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowlength() const;

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

   void setValue( const RealType& v );

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
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType elements );

   bool setRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType elements );

   __cuda_callable__
   bool addRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType elements,
                    const RealType& thisRowMultiplicator = 1.0 );

   bool addRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType elements,
                const RealType& thisRowMultiplicator = 1.0 );

   __cuda_callable__
   RealType getElementFast( const IndexType row,
                            const IndexType column ) const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   __cuda_callable__
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   const MatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

#ifdef HAVE_NOT_CXX11
   template< typename Real2, typename Index2 >
   void getTransposition( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );
#else
   template< typename Real2, typename Index2 >
   void getTransposition( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );
#endif

   template< typename Vector >
   __cuda_callable__
   void performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( File& file ) const;

   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   void print( std::ostream& str ) const;

   protected:

   __cuda_callable__
   IndexType getElementIndex( const IndexType row,
                              const IndexType column ) const;

   Vectors::Vector< RealType, DeviceType, IndexType > values;

   typedef tnlTridiagonalMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlTridiagonalMatrixDeviceDependentCode< DeviceType >;
};

} // namespace TNL

#include <TNL/matrices/tnlTridiagonalMatrix_impl.h>
