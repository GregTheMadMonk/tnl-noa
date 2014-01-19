/***************************************************************************
                          cuda-reduction-diff-abs-min_impl.cu  -  description
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
 * Diff abs min
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< char, int > >
                                   ( const tnlParallelReductionDiffAbsMin< char, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< int, int > >
                                   ( const tnlParallelReductionDiffAbsMin< int, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< float, int > >
                                   ( const tnlParallelReductionDiffAbsMin< float, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< double, int > >
                                   ( const tnlParallelReductionDiffAbsMin< double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< long double, int > >
                                   ( const tnlParallelReductionDiffAbsMin< long double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< char, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< char, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< int, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< int, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< float, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< float, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< double, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< double, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< long double, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< long double, long int> :: ResultType& result );

#endif                                     