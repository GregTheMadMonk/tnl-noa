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

   tnlMatrix();

   virtual bool setDimensions( const IndexType rows,
                               const IndexType columns );

   virtual bool setRowLengths( const RowLengthsVector& rowLengths ) = 0;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlMatrix< Real2, Device2, Index2 >& matrix );

   virtual IndexType getNumberOfMatrixElements() const = 0;

   void reset();

   IndexType getRows() const;

   IndexType getColumns() const;

   virtual bool setElement( const IndexType row,
                            const IndexType column,
                            const RealType& value ) = 0;

   virtual bool addElement( const IndexType row,
                            const IndexType column,
                            const RealType& value,
                            const RealType& thisElementMultiplicator = 1.0 ) = 0;

   virtual Real getElement( const IndexType row,
                            const IndexType column ) const = 0;

   virtual bool save( tnlFile& file ) const;

   virtual bool load( tnlFile& file );

   protected:

   IndexType rows, columns;

};

#include <implementation/matrices/tnlMatrix_impl.h>

#endif /* TNLMATRIX_H_ */
