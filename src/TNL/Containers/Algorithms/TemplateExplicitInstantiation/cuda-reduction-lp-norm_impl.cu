/***************************************************************************
                          cuda-reduction-lp-norm_impl.cu  -  description
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
 * Lp Norm
 */
template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< float, int > >
                                   ( tnlParallelReductionLpNorm< float, int >& operation,
                                     const typename tnlParallelReductionLpNorm< float, int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLpNorm< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< double, int > >
                                   ( tnlParallelReductionLpNorm< double, int>& operation,
                                     const typename tnlParallelReductionLpNorm< double, int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLpNorm< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< long double, int > >
                                   ( tnlParallelReductionLpNorm< long double, int>& operation,
                                     const typename tnlParallelReductionLpNorm< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLpNorm< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< int, long int > >
                                   ( tnlParallelReductionLpNorm< int, long int >& operation,
                                     const typename tnlParallelReductionLpNorm< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLpNorm< int, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< float, long int > >
                                   ( tnlParallelReductionLpNorm< float, long int >& operation,
                                     const typename tnlParallelReductionLpNorm< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLpNorm< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< double, long int > >
                                   ( tnlParallelReductionLpNorm< double, long int>& operation,
                                     const typename tnlParallelReductionLpNorm< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLpNorm< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< long double, long int > >
                                   ( tnlParallelReductionLpNorm< long double, long int>& operation,
                                     const typename tnlParallelReductionLpNorm< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLpNorm< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
