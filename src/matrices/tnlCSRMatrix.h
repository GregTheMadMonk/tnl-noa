/***************************************************************************
                          tnlCSRMatrix.h  -  description
                             -------------------
    begin                : Dec 10, 2013
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

#ifndef TNLCSRMATRIX_H_
#define TNLCSRMATRIX_H_

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Real >
class tnlCusparseCSRMatrix;

template< typename Device >
class tnlCSRMatrixDeviceDependentCode;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlCSRMatrix : public tnlSparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >:: RowLengthsVector RowLengthsVector;
   typedef tnlCSRMatrix< Real, Device, Index > ThisType;
   typedef tnlCSRMatrix< Real, tnlHost, Index > HostType;
   typedef tnlCSRMatrix< Real, tnlCuda, Index > CudaType;


   enum SPMVCudaKernel { scalar, vector, hybrid };

   tnlCSRMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlCSRMatrix< Real2, Device2, Index2 >& matrix );

   void reset();

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
   // TODO: add const RealType& multiplicator = 1.0 )

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
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

   void setCudaKernelType( const SPMVCudaKernel kernel );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   SPMVCudaKernel getCudaKernelType() const;

   void setCudaWarpSize( const int warpSize );

   int getCudaWarpSize() const;

   void setHybridModeSplit( const IndexType hybridModeSplit );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getHybridModeSplit() const;

#ifdef HAVE_CUDA

   template< typename InVector,
             typename OutVector,
             int warpSize >
   __device__
   void spmvCudaVectorized( const InVector& inVector,
                            OutVector& outVector,
                            const IndexType warpStart,
                            const IndexType warpEnd,
                            const IndexType inWarpIdx ) const;

   template< typename InVector,
             typename OutVector,
             int warpSize >
   __device__
   void vectorProductCuda( const InVector& inVector,
                           OutVector& outVector,
                           int gridIdx ) const;
#endif

   protected:

   tnlVector< Index, Device, Index > rowPointers;

   SPMVCudaKernel spmvCudaKernel;

   int cudaWarpSize, hybridModeSplit;

   typedef tnlCSRMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlCSRMatrixDeviceDependentCode< DeviceType >;
   friend class tnlCusparseCSRMatrix< RealType >;

};

#include <implementation/matrices/tnlCSRMatrix_impl.h>


#endif /* TNLCSRMATRIX_H_ */
