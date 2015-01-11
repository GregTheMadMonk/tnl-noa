/***************************************************************************
                          cuda-reduction-diff-abs-sum_impl.cu  -  description
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
 * Diff abs sum
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< char, int > >
                                   ( const tnlParallelReductionDiffAbsSum< char, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< int, int > >
                                   ( const tnlParallelReductionDiffAbsSum< int, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< float, int > >
                                   ( const tnlParallelReductionDiffAbsSum< float, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< double, int > >
                                   ( const tnlParallelReductionDiffAbsSum< double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< long double, int > >
                                   ( const tnlParallelReductionDiffAbsSum< long double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< char, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< char, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< int, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< int, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< float, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< float, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< double, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< double, long int> :: ResultType& result );

/*template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< long double, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< long double, long int> :: ResultType& result );*/

#endif                                     