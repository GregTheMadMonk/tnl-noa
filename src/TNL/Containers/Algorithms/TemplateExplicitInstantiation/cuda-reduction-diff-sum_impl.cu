/***************************************************************************
                          cuda-reduction-diff-sum_impl.cu  -  description
                             -------------------
    begin                : Jan 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */
 
#include <TNL/Containers/Algorithms/reduction-operations.h>
#include <TNL/Containers/Algorithms/Reduction.h>
 
namespace TNL {
namespace Containers {
namespace Algorithms {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Diff sum
 */

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< char, int > >
                                   ( tnlParallelReductionDiffSum< char, int >& operation,
                                     const typename tnlParallelReductionDiffSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< int, int > >
                                   ( tnlParallelReductionDiffSum< int, int >& operation,
                                     const typename tnlParallelReductionDiffSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< float, int > >
                                   ( tnlParallelReductionDiffSum< float, int >& operation,
                                     const typename tnlParallelReductionDiffSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< double, int > >
                                   ( tnlParallelReductionDiffSum< double, int>& operation,
                                     const typename tnlParallelReductionDiffSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< long double, int > >
                                   ( tnlParallelReductionDiffSum< long double, int>& operation,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< char, long int > >
                                   ( tnlParallelReductionDiffSum< char, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< int, long int > >
                                   ( tnlParallelReductionDiffSum< int, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< float, long int > >
                                   ( tnlParallelReductionDiffSum< float, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< double, long int > >
                                   ( tnlParallelReductionDiffSum< double, long int>& operation,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< long double, long int > >
                                   ( tnlParallelReductionDiffSum< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionDiffSum< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL