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

#include <core/vectors/tnlVector.h>

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlChunkedEllpackMatrix : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlChunkedEllpackMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   template< typename Vector >
   bool setRowLengths( const Vector& rowLengths );

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfAllocatedElements() const;

   void reset();

   IndexType getRows() const;

   IndexType getColumns() const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   bool setRow( const IndexType row,
                const IndexType* columnIndexes,
                const RealType* values,
                const IndexType elements );

   void setNumberOfChunksInSlice( const IndexType chunksInSlice );

   IndexType getNumberOfChunksInSlice() const;

   void setDesiredChunkSize( const IndexType desiredChunkSize );

   IndexType getDesiredChunkSize() const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );

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

   struct tnlChunkedEllpackSliceInfo
   {
      IndexType size;
      IndexType chunkSize;
      IndexType firstRow;
      IndexType pointer;

      static inline tnlString getType()
      { return tnlString( "tnlChunkedEllpackSliceInfo" ); };
   };

   IndexType rows, columns;

   IndexType chunksInSlice, desiredChunkSize;

   tnlVector< Real, Device, Index > values;

   tnlVector< Index, Device, Index > columnIndexes, chunksToRowsMapping, slicesToRowsMapping, rowPointers;

   tnlArray< tnlChunkedEllpackSliceInfo, Device, Index > slices;

   //IndexType numberOfSlices;

};

#include <implementation/matrices/tnlChunkedEllpackMatrix_impl.h>


#endif /* TNLCHUNKEDELLPACKMATRIX_H_ */
