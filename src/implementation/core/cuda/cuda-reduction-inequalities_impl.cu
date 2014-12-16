/***************************************************************************
                          cuda-reduction-inequalities_impl.cu  -  description
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
 * Inequalities
 */
template bool reductionOnCudaDevice< tnlParallelReductionInequalities< char, int > >
                                   ( const tnlParallelReductionInequalities< char, int >& operation,
                                     const typename tnlParallelReductionInequalities< char, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< int, int > >
                                   ( const tnlParallelReductionInequalities< int, int >& operation,
                                     const typename tnlParallelReductionInequalities< int, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< float, int > >
                                   ( const tnlParallelReductionInequalities< float, int >& operation,
                                     const typename tnlParallelReductionInequalities< float, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< double, int > >
                                   ( const tnlParallelReductionInequalities< double, int>& operation,
                                     const typename tnlParallelReductionInequalities< double, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< long double, int > >
                                   ( const tnlParallelReductionInequalities< long double, int>& operation,
                                     const typename tnlParallelReductionInequalities< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< long double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< char, long int > >
                                   ( const tnlParallelReductionInequalities< char, long int >& operation,
                                     const typename tnlParallelReductionInequalities< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< int, long int > >
                                   ( const tnlParallelReductionInequalities< int, long int >& operation,
                                     const typename tnlParallelReductionInequalities< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< float, long int > >
                                   ( const tnlParallelReductionInequalities< float, long int >& operation,
                                     const typename tnlParallelReductionInequalities< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionInequalities< double, long int > >
                                   ( const tnlParallelReductionInequalities< double, long int>& operation,
                                     const typename tnlParallelReductionInequalities< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< double, long int> :: ResultType& result );

/*template bool reductionOnCudaDevice< tnlParallelReductionInequalities< long double, long int > >
                                   ( const tnlParallelReductionInequalities< long double, long int>& operation,
                                     const typename tnlParallelReductionInequalities< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionInequalities< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionInequalities< long double, long int> :: ResultType& result );*/

#endif