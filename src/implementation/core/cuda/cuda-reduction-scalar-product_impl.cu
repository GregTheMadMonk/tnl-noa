/***************************************************************************
                          cuda-reduction-scalar-product_impl.cu  -  description
                             -------------------
    begin                : Jan 19, 2014
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

#include <core/cuda/reduction-operations.h>
#include <core/cuda/cuda-reduction.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * ScalarProduct
 */
template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< char, int > >
                                   ( const tnlParallelReductionScalarProduct< char, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< int, int > >
                                   ( const tnlParallelReductionScalarProduct< int, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< float, int > >
                                   ( const tnlParallelReductionScalarProduct< float, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< double, int > >
                                   ( const tnlParallelReductionScalarProduct< double, int>& operation,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< long double, int > >
                                   ( const tnlParallelReductionScalarProduct< long double, int>& operation,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< char, long int > >
                                   ( const tnlParallelReductionScalarProduct< char, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< int, long int > >
                                   ( const tnlParallelReductionScalarProduct< int, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< float, long int > >
                                   ( const tnlParallelReductionScalarProduct< float, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< double, long int > >
                                   ( const tnlParallelReductionScalarProduct< double, long int>& operation,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< double, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< long double, long int > >
                                   ( const tnlParallelReductionScalarProduct< long double, long int>& operation,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< long double, long int> :: ResultType& result );

#endif                                     