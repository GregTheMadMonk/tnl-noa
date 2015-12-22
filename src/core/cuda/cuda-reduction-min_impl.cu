/***************************************************************************
                          cuda-reduction-min_impl.cu  -  description
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
 * Min
 */

template bool reductionOnCudaDevice< tnlParallelReductionMin< char, int > >
                                   ( tnlParallelReductionMin< char, int >& operation,
                                     const typename tnlParallelReductionMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMin< int, int > >
                                   ( tnlParallelReductionMin< int, int >& operation,
                                     const typename tnlParallelReductionMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMin< float, int > >
                                   ( tnlParallelReductionMin< float, int >& operation,
                                     const typename tnlParallelReductionMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMin< double, int > >
                                   ( tnlParallelReductionMin< double, int>& operation,
                                     const typename tnlParallelReductionMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionMin< long double, int > >
                                   ( tnlParallelReductionMin< long double, int>& operation,
                                     const typename tnlParallelReductionMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionMin< char, long int > >
                                   ( tnlParallelReductionMin< char, long int >& operation,
                                     const typename tnlParallelReductionMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMin< int, long int > >
                                   ( tnlParallelReductionMin< int, long int >& operation,
                                     const typename tnlParallelReductionMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMin< float, long int > >
                                   ( tnlParallelReductionMin< float, long int >& operation,
                                     const typename tnlParallelReductionMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionMin< double, long int > >
                                   ( tnlParallelReductionMin< double, long int>& operation,
                                     const typename tnlParallelReductionMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionMin< long double, long int > >
                                   ( tnlParallelReductionMin< long double, long int>& operation,
                                     const typename tnlParallelReductionMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionMin< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionMin< long double, long int> :: ResultType& result );
#endif
#endif
#endif