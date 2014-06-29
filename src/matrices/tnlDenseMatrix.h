/***************************************************************************
                          tnlDenseMatrix.h  -  description
                             -------------------
    begin                : Nov 29, 2013
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

#ifndef TNLDENSEMATRIX_H_
#define TNLDENSEMATRIX_H_

#include <core/tnlHost.h>
#include <matrices/tnlMatrix.h>
#include <core/arrays/tnlArray.h>

template< typename Device >
class tnlDenseMatrixDeviceDependentCode;

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlDenseMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlMatrix< Real, Device, Index >::RowLengthsVector RowLengthsVector;
   typedef tnlDenseMatrix< Real, Device, Index > ThisType;
   typedef tnlDenseMatrix< Real, tnlHost, Index > HostType;
   typedef tnlDenseMatrix< Real, tnlCuda, Index > CudaType;


   tnlDenseMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlDenseMatrix< Real2, Device2, Index2 >& matrix );

   /****
    * This method is only for the compatibility with the sparse matrices.
    */
   bool setRowLengths( const RowLengthsVector& rowLengths );

   /****
    * Returns maximal number of the nonzero matrix elements that can be stored
    * in a given row.
    */
   IndexType getRowLength( const IndexType row ) const;

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   void reset();

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
   Real getElementFast( const IndexType row,
                        const IndexType column ) const;

   Real getElement( const IndexType row,
                    const IndexType column ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;

   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector, typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Matrix >
   void addMatrix( const Matrix& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

#ifdef HAVE_NOT_CXX11
   template< typename Matrix1, typename Matrix2, int tileDim >
   void getMatrixProduct( const Matrix1& matrix1,
                       const Matrix2& matrix2,
                       const RealType& matrix1Multiplicator = 1.0,
                       const RealType& matrix2Multiplicator = 1.0 );
#else
   template< typename Matrix1, typename Matrix2, int tileDim = 32 >
   void getMatrixProduct( const Matrix1& matrix1,
                       const Matrix2& matrix2,
                       const RealType& matrix1Multiplicator = 1.0,
                       const RealType& matrix2Multiplicator = 1.0 );
#endif   

#ifdef HAVE_NOT_CXX11
   template< typename Matrix, int tileDim >
   void getTransposition( const Matrix& matrix,
                          const RealType& matrixMultiplicator = 1.0 );
#else
   template< typename Matrix, int tileDim = 32 >
   void getTransposition( const Matrix& matrix,
                          const RealType& matrixMultiplicator = 1.0 );
#endif

   template< typename Vector >
   void performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   void print( ostream& str ) const;

   protected:

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getElementIndex( const IndexType row,
                              const IndexType column ) const;

   typedef tnlDenseMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlDenseMatrixDeviceDependentCode< DeviceType >;

};

#include <implementation/matrices/tnlDenseMatrix_impl.h>

#endif /* TNLDENSEMATRIX_H_ */
