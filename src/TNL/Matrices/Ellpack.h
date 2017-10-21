/***************************************************************************
                          Ellpack.h  -  description
                             -------------------
    begin                : Dec 7, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Sparse.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Matrices {   

template< typename Device >
class EllpackDeviceDependentCode;

template< typename Real, typename Device = Devices::Host, typename Index = int >
class Ellpack : public Sparse< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef Ellpack< Real, Device, Index > ThisType;
   typedef Ellpack< Real, Devices::Host, Index > HostType;
   typedef Ellpack< Real, Devices::Cuda, Index > CudaType;
   typedef Sparse< Real, Device, Index > BaseType;
   typedef typename BaseType::MatrixRow MatrixRow;
   typedef SparseRow< const RealType, const IndexType > ConstMatrixRow;

   Ellpack();

   static String getType();

   String getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths );

   bool setConstantCompressedRowLengths( const IndexType& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const Ellpack< Real2, Device2, Index2 >& matrix );

   void reset();
 
   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const Ellpack< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const Ellpack< Real2, Device2, Index2 >& matrix ) const;

   /*template< typename Matrix >
   bool copyFrom( const Matrix& matrix,
                  const CompressedRowLengthsVector& rowLengths );*/

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

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   ConstMatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const Ellpack< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const Ellpack< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( File& file ) const;

   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   void print( std::ostream& str ) const;

   protected:

   bool allocateElements();

   IndexType rowLengths, alignedRows;

   typedef EllpackDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class EllpackDeviceDependentCode< DeviceType >;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Ellpack_impl.h>
