/***************************************************************************
                          tnlSparseMatrixRow.h  -  description
                             -------------------
    begin                : Dec 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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


#ifndef TNLSPARSEMATRIXROW_H_
#define TNLSPARSEMATRIXROW_H_

template< typename Real, typename Index >
class tnlSparseMatrixRow
{
   public:

      __cuda_callable__
      tnlSparseMatrixRow();

      __cuda_callable__
      tnlSparseMatrixRow( Index* columns,
                          Real* values,
                          const Index length,
                          const Index step );

      __cuda_callable__
      void bind( Index* columns,
                 Real* values,
                 const Index length,
                 const Index step );

      __cuda_callable__
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );
      
      void print( ostream& str ) const;

   protected:

      Real* values;

      Index* columns;

      Index length, step;
};

template< typename Real, typename Index >
ostream& operator << ( ostream& str, const tnlSparseMatrixRow< Real, Index >& row )
{
   row.print( str );
   return str;
}

#include <matrices/tnlSparseMatrixRow_impl.h>

#endif /* TNLSPARSEMATRIXROW_H_ */
