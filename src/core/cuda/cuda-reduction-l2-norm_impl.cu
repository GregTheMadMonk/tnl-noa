/***************************************************************************
                          cuda-reduction-l2-norm_impl.cu  -  description
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
 * L2 Norm
 */
template bool reductionOnCudaDevice< tnlParallelReductionL2Norm< float, int > >
                                   ( tnlParallelReductionL2Norm< float, int >& operation,
                                     const typename tnlParallelReductionL2Norm< float, int > :: IndexType size,
                                     const typename tnlParallelReductionL2Norm< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionL2Norm< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionL2Norm< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionL2Norm< double, int > >
                                   ( tnlParallelReductionL2Norm< double, int>& operation,
                                     const typename tnlParallelReductionL2Norm< double, int > :: IndexType size,
                                     const typename tnlParallelReductionL2Norm< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionL2Norm< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionL2Norm< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionL2Norm< long double, int > >
                                   ( tnlParallelReductionL2Norm< long double, int>& operation,
                                     const typename tnlParallelReductionL2Norm< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionL2Norm< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionL2Norm< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionL2Norm< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionL2Norm< int, long int > >
                                   ( tnlParallelReductionL2Norm< int, long int >& operation,
                                     const typename tnlParallelReductionL2Norm< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionL2Norm< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionL2Norm< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionL2Norm< int, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionL2Norm< float, long int > >
                                   ( tnlParallelReductionL2Norm< float, long int >& operation,
                                     const typename tnlParallelReductionL2Norm< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionL2Norm< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionL2Norm< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionL2Norm< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionL2Norm< double, long int > >
                                   ( tnlParallelReductionL2Norm< double, long int>& operation,
                                     const typename tnlParallelReductionL2Norm< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionL2Norm< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionL2Norm< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionL2Norm< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionL2Norm< long double, long int > >
                                   ( tnlParallelReductionL2Norm< long double, long int>& operation,
                                     const typename tnlParallelReductionL2Norm< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionL2Norm< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionL2Norm< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionL2Norm< long double, long int> :: ResultType& result );
#endif
#endif
#endif
