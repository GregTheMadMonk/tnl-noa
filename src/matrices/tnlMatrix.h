/***************************************************************************
                          tnlMatrix.h  -  description
                             -------------------
    begin                : Dec 18, 2013
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

#ifndef TNLMATRIX_H_
#define TNLMATRIX_H_

#include <core/tnlObject.h>
#include <core/tnlHost.h>
#include <core/vectors/tnlVector.h>

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlMatrix : public virtual tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlVector< IndexType, DeviceType, IndexType > RowLengthsVector;
   typedef tnlVector< RealType, DeviceType, IndexType > ValuesVector;

   tnlMatrix();

   virtual bool setDimensions( const IndexType rows,
                               const IndexType columns );

   virtual bool setRowLengths( const RowLengthsVector& rowLengths ) = 0;

   virtual IndexType getRowLength( const IndexType row ) const = 0;

   virtual void getRowLengths( tnlVector< IndexType, DeviceType, IndexType >& rowLengths ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlMatrix< Real2, Device2, Index2 >& matrix );

   virtual IndexType getNumberOfMatrixElements() const = 0;

   virtual IndexType getNumberOfNonzeroMatrixElements() const = 0;

   void reset();

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getRows() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getColumns() const;

#ifdef HAVE_CUDA
    __device__ __host__
#endif
    IndexType getNumberOfColors() const;

   /****
    * TODO: The fast variants of the following methods cannot be virtual.
    * If they were, they could not be used in the CUDA kernels. If CUDA allows it
    * in the future and it does not slow down, declare them as virtual here.
    */

   virtual bool setElement( const IndexType row,
                            const IndexType column,
                            const RealType& value ) = 0;

   virtual bool addElement( const IndexType row,
                            const IndexType column,
                            const RealType& value,
                            const RealType& thisElementMultiplicator = 1.0 ) = 0;

   virtual bool setRow( const IndexType row,
                        const IndexType* columns,
                        const RealType* values,
                        const IndexType numberOfElements ) = 0;

   virtual bool addRow( const IndexType row,
                        const IndexType* columns,
                        const RealType* values,
                        const IndexType numberOfElements,
                        const RealType& thisElementMultiplicator = 1.0 ) = 0;

   virtual Real getElement( const IndexType row,
                            const IndexType column ) const = 0;

   virtual void getRow( const IndexType row,
                        IndexType* columns,
                        RealType* values ) const = 0;

   tnlMatrix< RealType, DeviceType, IndexType >& operator = ( const tnlMatrix< RealType, DeviceType, IndexType >& );

   template< typename Matrix >
   bool operator == ( const Matrix& matrix ) const;

   template< typename Matrix >
   bool operator != ( const Matrix& matrix ) const;

   template< typename Matrix >
   bool copyFrom( const Matrix& matrix,
                  const RowLengthsVector& rowLengths );

   virtual bool save( tnlFile& file ) const;

   virtual bool load( tnlFile& file );

   virtual void print( ostream& str ) const;

/*
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void computeColorsVector( tnlVector< Index, Device, Index >& colorsVector );
*/

   bool help( bool verbose = false );

#ifdef  HAVE_CUDA
   __device__ __host__
#endif
   void copyFromHostToCuda( tnlMatrix< Real, tnlHost, Index >& matrix );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void computeColorsVector( tnlVector< Index, Device, Index >& colorsVector );

   protected:

   IndexType rows, columns, numberOfColors;

   public: // TODO: remove this

   ValuesVector values;
};

template< typename Real, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlMatrix< Real, Device, Index >& m )
{
   m.print( str );
   return str;
}

template< typename Matrix,
          typename InVector,
          typename OutVector >
void tnlMatrixVectorProductCuda( const Matrix& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector );


#include <implementation/matrices/tnlMatrix_impl.h>

#endif /* TNLMATRIX_H_ */
