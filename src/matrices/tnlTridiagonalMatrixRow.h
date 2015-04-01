/***************************************************************************
                          tnlTridiagonalMatrixRow.h  -  description
                             -------------------
    begin                : Dec 31, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLTRIDIAGONALMATRIXROW_H_
#define TNLTRIDIAGONALMATRIXROW_H_

template< typename Real, typename Index >
class tnlTridiagonalMatrixRow
{
   public:

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlTridiagonalMatrixRow();

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlTridiagonalMatrixRow( Real* values,
                               const Index row,
                               const Index columns,
                               const Index step );

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      void bind( Real* values,
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

      Index row, columns, step;
};

#include <matrices/tnlTridiagonalMatrixRow_impl.h>


#endif /* TNLTRIDIAGONALMATRIXROW_H_ */
