/***************************************************************************
                          cuda-reduction-diff-min_impl.cu  -  description
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
 * Diff min
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< char, int > >
                                   ( tnlParallelReductionDiffMin< char, int >& operation,
                                     const typename tnlParallelReductionDiffMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< int, int > >
                                   ( tnlParallelReductionDiffMin< int, int >& operation,
                                     const typename tnlParallelReductionDiffMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< float, int > >
                                   ( tnlParallelReductionDiffMin< float, int >& operation,
                                     const typename tnlParallelReductionDiffMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< double, int > >
                                   ( tnlParallelReductionDiffMin< double, int>& operation,
                                     const typename tnlParallelReductionDiffMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< long double, int > >
                                   ( tnlParallelReductionDiffMin< long double, int>& operation,
                                     const typename tnlParallelReductionDiffMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< char, long int > >
                                   ( tnlParallelReductionDiffMin< char, long int >& operation,
                                     const typename tnlParallelReductionDiffMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< int, long int > >
                                   ( tnlParallelReductionDiffMin< int, long int >& operation,
                                     const typename tnlParallelReductionDiffMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< float, long int > >
                                   ( tnlParallelReductionDiffMin< float, long int >& operation,
                                     const typename tnlParallelReductionDiffMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< double, long int > >
                                   ( tnlParallelReductionDiffMin< double, long int>& operation,
                                     const typename tnlParallelReductionDiffMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< long double, long int > >
                                   ( tnlParallelReductionDiffMin< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMin< long double, long int> :: ResultType& result );
#endif
#endif
#endif                                     