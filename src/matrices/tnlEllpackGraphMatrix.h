/***************************************************************************
                          tnlEllpackGraphMatrix.h  -  description
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

#ifndef TNLELLPACKGRAPHMATRIX_H_
#define TNLELLPACKGRAPHMATRIX_H_

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Device >
class tnlEllpackGraphMatrixDeviceDependentCode;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlEllpackGraphMatrix : public tnlSparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::RowLengthsVector RowLengthsVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef tnlEllpackGraphMatrix< Real, Device, Index > ThisType;
   typedef tnlEllpackGraphMatrix< Real, tnlHost, Index > HostType;
   typedef tnlEllpackGraphMatrix< Real, tnlCuda, Index > CudaType;


   tnlEllpackGraphMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

   bool setConstantRowLengths( const IndexType& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlEllpackGraphMatrix< Real2, Device2, Index2 >& matrix );

   void reset();

   //template< typename Real2, typename Device2, typename Index2 >
   //bool operator == ( const tnlEllpackGraphMatrix< Real2, Device2, Index2 >& matrix ) const;

   //template< typename Real2, typename Device2, typename Index2 >
   //bool operator != ( const tnlEllpackGraphMatrix< Real2, Device2, Index2 >& matrix ) const;

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

#ifdef HAVE_CUDA
   template< typename InVector,
             typename OutVector >
   void spmvCuda( const InVector& inVector,
                  OutVector& outVector,
                  const int globalIdx,
                  const int color );
#endif

   void computeColorsVector( IndexType* colorsVector );

   void computePermutationArray();

   bool rearrangeMatrix();

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   void print( ostream& str ) const;

   bool help();

   Index getNumberOfColors() const;

   Index getRowsOfColor( IndexType color ) const;

   protected:

   bool allocateElements();

   IndexType rowLengths, alignedRows;

   typedef tnlEllpackGraphMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlEllpackGraphMatrixDeviceDependentCode< DeviceType >;

   tnlVector< Index, Device, Index > permutationArray;
   tnlVector< Index, Device, Index > colorPointers;
   IndexType numberOfColors;
   bool rearranged;
};

#include <implementation/matrices/tnlEllpackGraphMatrix_impl.h>


#endif /* TNLELLPACKGRAPHMATRIX_H_ */
