/***************************************************************************
                          cuda-reduction-abs-min_impl.cu  -  description
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
 * Abs min
 */

template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< char, int > >
                                   ( const tnlParallelReductionAbsMin< char, int >& operation,
                                     const typename tnlParallelReductionAbsMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< int, int > >
                                   ( const tnlParallelReductionAbsMin< int, int >& operation,
                                     const typename tnlParallelReductionAbsMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< float, int > >
                                   ( const tnlParallelReductionAbsMin< float, int >& operation,
                                     const typename tnlParallelReductionAbsMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< double, int > >
                                   ( const tnlParallelReductionAbsMin< double, int>& operation,
                                     const typename tnlParallelReductionAbsMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< long double, int > >
                                   ( const tnlParallelReductionAbsMin< long double, int>& operation,
                                     const typename tnlParallelReductionAbsMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< long double, int> :: ResultType& result );
#endif

template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< char, long int > >
                                   ( const tnlParallelReductionAbsMin< char, long int >& operation,
                                     const typename tnlParallelReductionAbsMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< char, long int > :: ResultType& result );

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< int, long int > >
                                   ( const tnlParallelReductionAbsMin< int, long int >& operation,
                                     const typename tnlParallelReductionAbsMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< float, long int > >
                                   ( const tnlParallelReductionAbsMin< float, long int >& operation,
                                     const typename tnlParallelReductionAbsMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< double, long int > >
                                   ( const tnlParallelReductionAbsMin< double, long int>& operation,
                                     const typename tnlParallelReductionAbsMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< long double, long int > >
                                   ( const tnlParallelReductionAbsMin< long double, long int>& operation,
                                     const typename tnlParallelReductionAbsMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsMin< long double, long int> :: ResultType& result );
#endif
#endif                                    
#endif                                     