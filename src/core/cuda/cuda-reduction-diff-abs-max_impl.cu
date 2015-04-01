/***************************************************************************
                          cuda-reduction-diff-abs-max_impl.cu  -  description
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
 * Diff abs max
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< char, int > >
                                   ( const tnlParallelReductionDiffAbsMax< char, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< int, int > >
                                   ( const tnlParallelReductionDiffAbsMax< int, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< float, int > >
                                   ( const tnlParallelReductionDiffAbsMax< float, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< double, int > >
                                   ( const tnlParallelReductionDiffAbsMax< double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< long double, int > >
                                   ( const tnlParallelReductionDiffAbsMax< long double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< char, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< char, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< int, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< int, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< float, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< float, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< double, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< double, long int> :: ResultType& result );

/*template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< long double, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< long double, long int> :: ResultType& result );*/

#endif                                     