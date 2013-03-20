/***************************************************************************
                          cuda-long-vector-kernels.h  -  description
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

#ifndef CUDALONGVECTORKERNELS_H_
#define CUDALONGVECTORKERNELS_H_

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <iostream>

/***
 * The template calling the final CUDA kernel for the single vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlParallelReductionSum, tnlParallelReductionMin or tnlParallelReductionMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput1 is the pointer to an array storing the data we want
 *        to reduce. This array must stay on the device!.
 * @param deviceInput2 is the pointer to an array storing the coupling data for example
 *        the second vector for the SDOT operation. This array must stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param parameter can be used for example for the passing the parameter p of Lp norm.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 */
template< typename Type, typename ParameterType, typename Index, tnlTupleOperation operation >
bool tnlCUDALongVectorReduction( const Index size,
                                 const Type* deviceInput1,
                                 const Type* deviceInput2,
                                 Type& result,
                                 const ParameterType& parameter,
                                 Type* deviceAux = 0 );
#endif /* CUDALONGVECTORKERNELS_H_ */
