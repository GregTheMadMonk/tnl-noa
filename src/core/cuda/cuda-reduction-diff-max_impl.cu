/***************************************************************************
                          cuda-reduction-diff-max_impl.cu  -  description
                             -------------------
    begin                : Jan 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */
 
#include <core/cuda/reduction-operations.h>
#include <core/cuda/cuda-reduction.h>
 
#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Diff max
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< char, int > >
                                   ( tnlParallelReductionDiffMax< char, int >& operation,
                                     const typename tnlParallelReductionDiffMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< int, int > >
                                   ( tnlParallelReductionDiffMax< int, int >& operation,
                                     const typename tnlParallelReductionDiffMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< float, int > >
                                   ( tnlParallelReductionDiffMax< float, int >& operation,
                                     const typename tnlParallelReductionDiffMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< double, int > >
                                   ( tnlParallelReductionDiffMax< double, int>& operation,
                                     const typename tnlParallelReductionDiffMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< long double, int > >
                                   ( tnlParallelReductionDiffMax< long double, int>& operation,
                                     const typename tnlParallelReductionDiffMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< char, long int > >
                                   ( tnlParallelReductionDiffMax< char, long int >& operation,
                                     const typename tnlParallelReductionDiffMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< int, long int > >
                                   ( tnlParallelReductionDiffMax< int, long int >& operation,
                                     const typename tnlParallelReductionDiffMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< float, long int > >
                                   ( tnlParallelReductionDiffMax< float, long int >& operation,
                                     const typename tnlParallelReductionDiffMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< double, long int > >
                                   ( tnlParallelReductionDiffMax< double, long int>& operation,
                                     const typename tnlParallelReductionDiffMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< long double, long int > >
                                   ( tnlParallelReductionDiffMax< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffMax< long double, long int> :: ResultType& result );
#endif
#endif
#endif
