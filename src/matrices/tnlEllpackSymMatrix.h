/***************************************************************************
                          tnlEllpackSymMatrix.h  -  description
                             -------------------
    begin                : Dec 7, 2013
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

#ifndef TNLELLPACKSYMMATRIX_H_
#define TNLELLPACKSYMMATRIX_H_

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Device >
class tnlEllpackSymMatrixDeviceDependentCode;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlEllpackSymMatrix : public tnlSparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::RowLengthsVector RowLengthsVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef tnlEllpackSymMatrix< Real, Device, Index > ThisType;
   typedef tnlEllpackSymMatrix< Real, tnlHost, Index > HostType;
   typedef tnlEllpackSymMatrix< Real, tnlCuda, Index > CudaType;


   tnlEllpackSymMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

   bool setConstantRowLengths( const IndexType& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlEllpackSymMatrix< Real2, Device2, Index2 >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlEllpackSymMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlEllpackSymMatrix< Real2, Device2, Index2 >& matrix ) const;

   /*template< typename Matrix >
   bool copyFrom( const Matrix& matrix,
                  const RowLengthsVector& rowLengths );*/

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
                    const IndexType* columnIndexes,
                    const RealType* values,
                    const IndexType elements );

   bool setRow( const IndexType row,
                const IndexType* columnIndexes,
                const RealType* values,
                const IndexType elements );


#ifdef HAVE_CUDA
   __device__ __host__
#endif
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

   void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;

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

   template< typename InVector,
             typename OutVector >
   void vectorProductHost( const InVector& inVector,
                           OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlEllpackSymMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlEllpackSymMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   void print( ostream& str ) const;

   template< typename InVector,
             typename OutVector >
#ifdef HAVE_CUDA
   __device__
#endif
   void spmvCuda( const InVector& inVector,
                  OutVector& outVector,
                  int rowIdx ) const;

   protected:

   bool allocateElements();

   IndexType rowLengths, alignedRows;

   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlEllpackSymMatrixDeviceDependentCode< DeviceType >;
};

#include <implementation/matrices/tnlEllpackSymMatrix_impl.h>


#endif /* TNLELLPACKSYMMATRIX_H_ */
