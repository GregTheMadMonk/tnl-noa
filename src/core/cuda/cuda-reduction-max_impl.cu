/***************************************************************************
                          cuda-reduction-max_impl.cu  -  description
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
 * Max
 */

template bool reductionOnCudaDevice< tnlParallelReductionMax< char, int > >
                                   ( const tnlParallelReductionMax< char, int >& operation,
                                     const typename tnlParallelReductionMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< int, int > >
                                   ( const tnlParallelReductionMax< int, int >& operation,
                                     const typename tnlParallelReductionMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< float, int > >
                                   ( const tnlParallelReductionMax< float, int >& operation,
                                     const typename tnlParallelReductionMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< double, int > >
                                   ( const tnlParallelReductionMax< double, int>& operation,
                                     const typename tnlParallelReductionMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< long double, int > >
                                   ( const tnlParallelReductionMax< long double, int>& operation,
                                     const typename tnlParallelReductionMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< char, long int > >
                                   ( const tnlParallelReductionMax< char, long int >& operation,
                                     const typename tnlParallelReductionMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< int, long int > >
                                   ( const tnlParallelReductionMax< int, long int >& operation,
                                     const typename tnlParallelReductionMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< float, long int > >
                                   ( const tnlParallelReductionMax< float, long int >& operation,
                                     const typename tnlParallelReductionMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMax< double, long int > >
                                   ( const tnlParallelReductionMax< double, long int>& operation,
                                     const typename tnlParallelReductionMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< double, long int> :: ResultType& result );

/*template bool reductionOnCudaDevice< tnlParallelReductionMax< long double, long int > >
                                   ( const tnlParallelReductionMax< long double, long int>& operation,
                                     const typename tnlParallelReductionMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMax< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMax< long double, long int> :: ResultType& result );*/

#endif