/***************************************************************************
                          cuda-reduction-diff-lp-norm_impl.cu  -  description
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
 * Diff Lp Norm
 */
template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< float, int > >
                                   ( const tnlParallelReductionDiffLpNorm< float, int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< double, int > >
                                   ( const tnlParallelReductionDiffLpNorm< double, int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< long double, int > >
                                   ( const tnlParallelReductionDiffLpNorm< long double, int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< char, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< char, long int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< int, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< int, long int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< float, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< float, long int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< double, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< double, long int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< long double, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< long double, long int> :: ResultType& result );
#endif
#endif
#endif
