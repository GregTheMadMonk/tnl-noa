/***************************************************************************
                          tnlChunkedEllpackMatrix.h  -  description
                             -------------------
    begin                : Dec 12, 2013
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

#ifndef TNLCHUNKEDELLPACKMATRIX_H_
#define TNLCHUNKEDELLPACKMATRIX_H_

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Device >
class tnlChunkedEllpackMatrixDeviceDependentCode;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlChunkedEllpackMatrix;

#ifdef HAVE_CUDA
#endif

template< typename IndexType >
struct tnlChunkedEllpackSliceInfo
{
   IndexType size;
   IndexType chunkSize;
   IndexType firstRow;
   IndexType pointer;

   static inline tnlString getType()
   { return tnlString( "tnlChunkedEllpackSliceInfo" ); };
};

template< typename Real, typename Device, typename Index >
class tnlChunkedEllpackMatrix : public tnlSparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlChunkedEllpackSliceInfo< IndexType > ChunkedEllpackSliceInfo;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >:: RowLengthsVector RowLengthsVector;

   tnlChunkedEllpackMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   void setNumberOfChunksInSlice( const IndexType chunksInSlice );

   IndexType getNumberOfChunksInSlice() const;

   void setDesiredChunkSize( const IndexType desiredChunkSize );

   IndexType getDesiredChunkSize() const;

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
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename Vector >
   void vectorProduct( const Vector& inVector,
                       Vector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlChunkedEllpackMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlChunkedEllpackMatrix< Real2, Device, Index2 >& matrix,
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

   protected:


   void resolveSliceSizes( const tnlVector< Index, tnlHost, Index >& rowLengths,
                           IndexType& numberOfSlices );

   bool setSlice( const RowLengthsVector& rowLengths,
                  const IndexType sliceIdx,
                  IndexType& elementsToAllocation );

   IndexType chunksInSlice, desiredChunkSize;

   tnlVector< Index, Device, Index > rowToChunkMapping, rowToSliceMapping, rowPointers;

   tnlArray< ChunkedEllpackSliceInfo, Device, Index > slices;

   //IndexType numberOfSlices;

   typedef tnlChunkedEllpackMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlChunkedEllpackMatrixDeviceDependentCode< DeviceType >;
   friend class tnlChunkedEllpackMatrix< RealType, tnlHost, IndexType >;
   friend class tnlChunkedEllpackMatrix< RealType, tnlCuda, IndexType >;

/*#ifdef HAVE_CUDA
   friend void tnlChunkedEllpackMatrix_setSlices_CudaKernel< Real, Index, 256 >( tnlChunkedEllpackMatrix< Real, tnlCuda, Index >* matrix,
                                                                                 const RowLengthsVector* rowLengths,
                                                                                 const Index numberOfSlices,
                                                                                 Index* elementsToAllocation,
                                                                                 const Index gridIdx );
#endif*/


};

#include <implementation/matrices/tnlChunkedEllpackMatrix_impl.h>


#endif /* TNLCHUNKEDELLPACKMATRIX_H_ */
