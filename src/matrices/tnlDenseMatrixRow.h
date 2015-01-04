/***************************************************************************
                          tnlDenseMatrixRow.h  -  description
                             -------------------
    begin                : Dec 24, 2014
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

#ifndef TNLDENSEMATRIXROW_H_
#define TNLDENSEMATRIXROW_H_

template< typename Real, typename Index >
class tnlDenseMatrixRow
{
   public:

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlDenseMatrixRow();

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      tnlDenseMatrixRow( Real* values,
                         const Index columns,
                         const Index step );

#ifdef HAVE_CUDA
      __device__ __host__
#endif
      void bind( Real* values,
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

      Index columns, step;
};

#include <implementation/matrices/tnlDenseMatrixRow_impl.h>


#endif /* TNLDENSEMATRIXROW_H_ */
