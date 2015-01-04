/***************************************************************************
                          tnlTridiagonalMatrix.h  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLTRIDIAGONALMATRIX_H_
#define TNLTRIDIAGONALMATRIX_H_

#include <matrices/tnlMatrix.h>
#include <core/vectors/tnlVector.h>
#include <matrices/tnlTridiagonalMatrixRow.h>

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
   typedef typename tnlMatrix< Real, Device, Index >::RowLengthsVector RowLengthsVector;
   typedef tnlTridiagonalMatrix< Real, Device, Index > ThisType;
   typedef tnlTridiagonalMatrix< Real, tnlHost, Index > HostType;
   typedef tnlTridiagonalMatrix< Real, tnlCuda, Index > CudaType;
   typedef tnlMatrix< Real, Device, Index > BaseType;
   typedef tnlTridiagonalMatrixRow< Real, Index > MatrixRow;

   tnlTridiagonalMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

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

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool setElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool addElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value,
                        const RealType& thisElementMultiplicator = 1.0 );

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool setRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType elements );

   bool setRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType elements );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
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

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   RealType getElementFast( const IndexType row,
                            const IndexType column ) const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   MatrixRow getRow( const IndexType rowIndex );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const MatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   void print( ostream& str ) const;

   protected:

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getElementIndex( const IndexType row,
                              const IndexType column ) const;

   tnlVector< RealType, DeviceType, IndexType > values;

   typedef tnlTridiagonalMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlTridiagonalMatrixDeviceDependentCode< DeviceType >;
};

#include <implementation/matrices/tnlTridiagonalMatrix_impl.h>


#endif /* TNLTRIDIAGONALMATRIX_H_ */
