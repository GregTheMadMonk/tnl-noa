/***************************************************************************
                          cuda-reduction-abs-max_impl.cu  -  description
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
 * Abs max
 */

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< char, int > >
                                   ( const tnlParallelReductionAbsMax< char, int >& operation,
                                     const typename tnlParallelReductionAbsMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< int, int > >
                                   ( const tnlParallelReductionAbsMax< int, int >& operation,
                                     const typename tnlParallelReductionAbsMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< float, int > >
                                   ( const tnlParallelReductionAbsMax< float, int >& operation,
                                     const typename tnlParallelReductionAbsMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< double, int > >
                                   ( const tnlParallelReductionAbsMax< double, int>& operation,
                                     const typename tnlParallelReductionAbsMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< double, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< long double, int > >
                                   ( const tnlParallelReductionAbsMax< long double, int>& operation,
                                     const typename tnlParallelReductionAbsMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< long double, int> :: ResultType& result );
                                     

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< char, long int > >
                                   ( const tnlParallelReductionAbsMax< char, long int >& operation,
                                     const typename tnlParallelReductionAbsMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< int, long int > >
                                   ( const tnlParallelReductionAbsMax< int, long int >& operation,
                                     const typename tnlParallelReductionAbsMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< float, long int > >
                                   ( const tnlParallelReductionAbsMax< float, long int >& operation,
                                     const typename tnlParallelReductionAbsMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< double, long int > >
                                   ( const tnlParallelReductionAbsMax< double, long int>& operation,
                                     const typename tnlParallelReductionAbsMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< double, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< long double, long int > >
                                   ( const tnlParallelReductionAbsMax< long double, long int>& operation,
                                     const typename tnlParallelReductionAbsMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMax< long double, long int> :: ResultType& result );

#endif                                     