/***************************************************************************
                          cuda-reduction-diff-sum_impl.cu  -  description
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
 * Diff sum
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< char, int > >
                                   ( const tnlParallelReductionDiffSum< char, int >& operation,
                                     const typename tnlParallelReductionDiffSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< int, int > >
                                   ( const tnlParallelReductionDiffSum< int, int >& operation,
                                     const typename tnlParallelReductionDiffSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< float, int > >
                                   ( const tnlParallelReductionDiffSum< float, int >& operation,
                                     const typename tnlParallelReductionDiffSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< double, int > >
                                   ( const tnlParallelReductionDiffSum< double, int>& operation,
                                     const typename tnlParallelReductionDiffSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< long double, int > >
                                   ( const tnlParallelReductionDiffSum< long double, int>& operation,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< char, long int > >
                                   ( const tnlParallelReductionDiffSum< char, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< int, long int > >
                                   ( const tnlParallelReductionDiffSum< int, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< float, long int > >
                                   ( const tnlParallelReductionDiffSum< float, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< double, long int > >
                                   ( const tnlParallelReductionDiffSum< double, long int>& operation,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< double, long int> :: ResultType& result );

/*template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< long double, long int > >
                                   ( const tnlParallelReductionDiffSum< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< long double, long int> :: ResultType& result );*/
                                    
#endif                                     