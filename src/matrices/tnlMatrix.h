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

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlMatrix : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   virtual bool setDimensions( const IndexType rows,
                               const IndexType columns );

   IndexType getRows() const;

   IndexType getColumns() const;

   virtual bool setElement( const IndexType row,
                            const IndexType column,
                            const RealType& value );

   virtual bool addElement( const IndexType row,
                            const IndexType column,
                            const RealType& value,
                            const RealType& thisElementMultiplicator = 1.0 );


   virtual bool save( tnlFile& file ) const;

   virtual bool load( tnlFile& file );

   virtual bool save( const tnlString& fileName ) const;

   virtual bool load( const tnlString& fileName );

   protected:

   IndexType rows, columns;

};

#include <implementation/matrices/tnlMatrix.h>

#endif /* TNLMATRIX_H_ */
