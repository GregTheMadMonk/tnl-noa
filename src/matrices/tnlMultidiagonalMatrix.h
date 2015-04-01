/***************************************************************************
                          tnlMultidiagonalMatrix.h  -  description
                             -------------------
    begin                : Oct 13, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#ifndef TNLMULTIDIAGONALMATRIX_H_
#define TNLMULTIDIAGONALMATRIX_H_

#include <matrices/tnlMatrix.h>
#include <core/vectors/tnlVector.h>
#include <matrices/tnlMultidiagonalMatrixRow.h>

template< typename Device >
class tnlMultidiagonalMatrixDeviceDependentCode;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlMultidiagonalMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlMatrix< Real, Device, Index >::RowLengthsVector RowLengthsVector;
   typedef tnlMultidiagonalMatrix< Real, Device, Index > ThisType;
   typedef tnlMultidiagonalMatrix< Real, tnlHost, Index > HostType;
   typedef tnlMultidiagonalMatrix< Real, tnlCuda, Index > CudaType;
   typedef tnlMatrix< Real, Device, Index > BaseType;
   typedef tnlMultidiagonalMatrixRow< Real, Index > MatrixRow;


   tnlMultidiagonalMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   IndexType getMaxRowLength() const;

   template< typename Vector >
   bool setDiagonals( const Vector& diagonals );

   const tnlVector< Index, Device, Index >& getDiagonals() const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowlength() const;

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

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
                    const IndexType numberOfElements );

   bool setRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType numberOfElements );


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

   /*void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;*/

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
   void addMatrix( const tnlMultidiagonalMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlMultidiagonalMatrix< Real2, Device, Index2 >& matrix,
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

   bool getElementIndex( const IndexType row,
                         const IndexType column,
                         IndexType& index ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool getElementIndexFast( const IndexType row,
                             const IndexType column,
                             IndexType& index ) const;

   tnlVector< Real, Device, Index > values;

   tnlVector< Index, Device, Index > diagonalsShift;

   typedef tnlMultidiagonalMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlMultidiagonalMatrixDeviceDependentCode< DeviceType >;


};

#include <matrices/tnlMultidiagonalMatrix_impl.h>

#endif /* TNLMULTIDIAGONALMATRIX_H_ */
