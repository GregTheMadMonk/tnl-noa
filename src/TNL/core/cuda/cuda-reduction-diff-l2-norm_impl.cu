/***************************************************************************
                          cuda-reduction-diff-lp-norm_impl.cu  -  description
                             -------------------
    begin                : Jan 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */
 
#include <TNL/core/cuda/reduction-operations.h>
#include <TNL/core/cuda/cuda-reduction.h>
 
namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Diff L2 Norm
 */
template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< float, int > >
                                   ( tnlParallelReductionDiffL2Norm< float, int >& operation,
                                     const typename tnlParallelReductionDiffL2Norm< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< double, int > >
                                   ( tnlParallelReductionDiffL2Norm< double, int>& operation,
                                     const typename tnlParallelReductionDiffL2Norm< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< long double, int > >
                                   ( tnlParallelReductionDiffL2Norm< long double, int>& operation,
                                     const typename tnlParallelReductionDiffL2Norm< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< char, long int > >
                                   ( tnlParallelReductionDiffL2Norm< char, long int >& operation,
                                     const typename tnlParallelReductionDiffL2Norm< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< int, long int > >
                                   ( tnlParallelReductionDiffL2Norm< int, long int >& operation,
                                     const typename tnlParallelReductionDiffL2Norm< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< float, long int > >
                                   ( tnlParallelReductionDiffL2Norm< float, long int >& operation,
                                     const typename tnlParallelReductionDiffL2Norm< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< double, long int > >
                                   ( tnlParallelReductionDiffL2Norm< double, long int>& operation,
                                     const typename tnlParallelReductionDiffL2Norm< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffL2Norm< long double, long int > >
                                   ( tnlParallelReductionDiffL2Norm< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffL2Norm< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffL2Norm< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffL2Norm< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffL2Norm< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace TNL