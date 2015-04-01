/***************************************************************************
                          cuda-reduction.h  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef CUDA_REDUCTION_H_
#define CUDA_REDUCTION_H_

template< typename Operation >
bool reductionOnCudaDevice( const Operation& operation,
                            const typename Operation :: IndexType size,
                            const typename Operation :: RealType* deviceInput1,
                            const typename Operation :: RealType* deviceInput2,
                            typename Operation :: ResultType& result );

#include <core/cuda/cuda-reduction_impl.h>

#endif /* CUDA_REDUCTION_H_ */
