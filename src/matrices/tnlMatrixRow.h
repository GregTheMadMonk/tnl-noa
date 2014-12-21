/***************************************************************************
                          tnlMatrixRow.h  -  description
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


#ifndef TNLMATRIXROW_H_
#define TNLMATRIXROW_H_

template< typename Real, typename Index >
class tnlMatrixRow
{
   public:

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlMatrixRow();

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlMatrixRow( Index* columns,
                    Real* values,
                    const Index length,
                    const Index step );

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      void bind( Index* columns,
                 Real* values,
                 const Index length,
                 const Index step );

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );

   protected:

      Real* values;

      Index* columns;

      Index length, step;
};

#include <implementation/matrices/tnlMatrixRow_impl.h>

#endif /* TNLMATRIXROW_H_ */
