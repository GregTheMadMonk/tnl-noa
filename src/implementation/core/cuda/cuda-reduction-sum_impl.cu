/***************************************************************************
                          cuda-reduction-sum_impl.cu  -  description
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
 * Sum 
 */

template bool reductionOnCudaDevice< tnlParallelReductionSum< char, int > >
                                   ( const tnlParallelReductionSum< char, int >& operation,
                                     const typename tnlParallelReductionSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< int, int > >
                                   ( const tnlParallelReductionSum< int, int >& operation,
                                     const typename tnlParallelReductionSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< float, int > >
                                   ( const tnlParallelReductionSum< float, int >& operation,
                                     const typename tnlParallelReductionSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< double, int > >
                                   ( const tnlParallelReductionSum< double, int>& operation,
                                     const typename tnlParallelReductionSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< long double, int > >
                                   ( const tnlParallelReductionSum< long double, int>& operation,
                                     const typename tnlParallelReductionSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< char, long int > >
                                   ( const tnlParallelReductionSum< char, long int >& operation,
                                     const typename tnlParallelReductionSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< int, long int > >
                                   ( const tnlParallelReductionSum< int, long int >& operation,
                                     const typename tnlParallelReductionSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< float, long int > >
                                   ( const tnlParallelReductionSum< float, long int >& operation,
                                     const typename tnlParallelReductionSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< double, long int > >
                                   ( const tnlParallelReductionSum< double, long int>& operation,
                                     const typename tnlParallelReductionSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< double, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionSum< long double, long int > >
                                   ( const tnlParallelReductionSum< long double, long int>& operation,
                                     const typename tnlParallelReductionSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionSum< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionSum< long double, long int> :: ResultType& result );
                                     
#endif                                     