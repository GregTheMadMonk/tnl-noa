/***************************************************************************
                          cuda-reduction-diff-abs-sum_impl.cu  -  description
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
 * Diff abs sum
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< char, int > >
                                   ( tnlParallelReductionDiffAbsSum< char, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< int, int > >
                                   ( tnlParallelReductionDiffAbsSum< int, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< float, int > >
                                   ( tnlParallelReductionDiffAbsSum< float, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< double, int > >
                                   ( tnlParallelReductionDiffAbsSum< double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< long double, int > >
                                   ( tnlParallelReductionDiffAbsSum< long double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< char, long int > >
                                   ( tnlParallelReductionDiffAbsSum< char, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< int, long int > >
                                   ( tnlParallelReductionDiffAbsSum< int, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< float, long int > >
                                   ( tnlParallelReductionDiffAbsSum< float, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< double, long int > >
                                   ( tnlParallelReductionDiffAbsSum< double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< long double, long int > >
                                   ( tnlParallelReductionDiffAbsSum< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace TNL