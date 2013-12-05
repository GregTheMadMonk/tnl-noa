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

#include <core/vectors/tnlVector.h>

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlMultidiagonalMatrix : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   //! Basic constructor
   tnlMultidiagonalMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setDiagonals( const tnlVector< Index, Device, Index >& diagonals );

   const tnlVector< Index, Device, Index >& getDiagonals() const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfAllocatedElements() const;

   void reset();

   IndexType getRows() const;

   IndexType getColumns() const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

   void setValue( const RealType& v );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   bool addToElement( const IndexType row,
                      const IndexType column,
                      const RealType& value,
                      const RealType& thisElementMultiplicator = 1.0 );

   template< typename Vector >
   void vectorProduct( const Vector& inVector,
                       Vector& outVector ) const;

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

   IndexType rows, columns;

   tnlVector< Real, Device, Index > values;

   tnlVector< Index, Device, Index > diagonalsOffsets;

};

#include <implementation/matrices/tnlMultidiagonalMatrix_impl.h>

#endif /* TNLMULTIDIAGONALMATRIX_H_ */
