/***************************************************************************
                          tnlSlicedEllpackGraphMatrix.h  -  description
                             -------------------
    begin                : Dec 8, 2013
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

#ifndef TNLSLICEDELLPACKGRAPHMATRIX_H_
#define TNLSLICEDELLPACKGRAPHMATRIX_H_

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Device >
class tnlSlicedEllpackGraphMatrixDeviceDependentCode;

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int,
          int SliceSize = 32 >
class tnlSlicedEllpackGraphMatrix;

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void tnlSlicedEllpackGraphMatrix_computeMaximalRowLengthInSlices_CudaKernel( tnlSlicedEllpackGraphMatrix< Real, tnlCuda, Index, SliceSize >* matrix,
                                                                                       const typename tnlSlicedEllpackGraphMatrix< Real, tnlCuda, Index, SliceSize >::RowLengthsVector* rowLengths,
                                                                                       int gridIdx );
#endif

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
class tnlSlicedEllpackGraphMatrix : public tnlSparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::RowLengthsVector RowLengthsVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef tnlSlicedEllpackGraphMatrix< Real, Device, Index > ThisType;
   typedef tnlSlicedEllpackGraphMatrix< Real, tnlHost, Index > HostType;
   typedef tnlSlicedEllpackGraphMatrix< Real, tnlCuda, Index > CudaType;


   tnlSlicedEllpackGraphMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlSlicedEllpackGraphMatrix< Real2, Device2, Index2, SliceSize >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlSlicedEllpackGraphMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlSlicedEllpackGraphMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProductHost( const InVector& inVector, OutVector& outVector );

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

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlSlicedEllpackGraphMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlSlicedEllpackGraphMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   Index getRealRowLength( const Index row );

   tnlVector< Index, Device, Index > getRealRowLengths();

   bool testRowLengths( tnlVector< Index, Device, Index >& rowLengths,
                        tnlVector< Index, Device, Index >& sliceRowLengths );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   void print( ostream& str ) const;

   bool help( bool verbose = false );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool rearrangeMatrix( bool verbose = false );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void computePermutationArray();

   protected:

   tnlVector< Index, Device, Index > slicePointers, sliceRowLengths;

   typedef tnlSlicedEllpackGraphMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlSlicedEllpackGraphMatrixDeviceDependentCode< DeviceType >;

   tnlVector< Index, Device, Index > permutationArray;
   tnlVector< Index, Device, Index > colorPointers;
   bool rearranged;
#ifdef HAVE_CUDA
   /*friend __global__ void tnlSlicedEllpackGraphMatrix_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize >( tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >* matrix,
                                                                                      const typename tnlSlicedEllpackGraphMatrix< Real, tnlCuda, Index, SliceSize >::RowLengthsVector* rowLengths,
                                                                                      int gridIdx );
    */
   // TODO: The friend declaration above does not work because of __global__ storage specifier. Therefore we declare the following method as public. Fix this, when possible.

   public:
   __device__ void computeMaximalRowLengthInSlicesCuda( const RowLengthsVector& rowLengths,
                                                        const IndexType sliceIdx );

#endif

};

#include <implementation/matrices/tnlSlicedEllpackGraphMatrix_impl.h>


#endif /* TNLSLICEDELLPACKGRAPHMATRIX_H_ */
