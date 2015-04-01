/***************************************************************************
                          tnlMultidiagonalMatrixRow.h  -  description
                             -------------------
    begin                : Jan 2, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLMULTIDIAGONALMATRIXROW_H_
#define TNLMULTIDIAGONALMATRIXROW_H_

template< typename Real, typename Index >
class tnlMultidiagonalMatrixRow
{
   public:

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlMultidiagonalMatrixRow();

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlMultidiagonalMatrixRow( Real* values,
                                 Index* diagonals,
                                 const Index maxRowLength,
                                 const Index row,
                                 const Index columns,
                                 const Index step );

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      void bind( Real* values,
                 Index* diagonals,
                 const Index maxRowLength,
                 const Index row,
                 const Index columns,
                 const Index step );

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );

   protected:

      Real* values;

      Index* diagonals;

      Index row, columns, maxRowLength, step;
};

#include <matrices/tnlMultidiagonalMatrixRow_impl.h>


#endif /* TNLMULTIDIAGONALMATRIXROW_H_ */
