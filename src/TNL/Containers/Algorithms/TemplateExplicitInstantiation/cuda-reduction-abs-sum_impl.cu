/***************************************************************************
                          cuda-reduction-abs-sum_impl.cu  -  description
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
 * Abs sum
 */

template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< char, int > >
                                   ( tnlParallelReductionAbsSum< char, int >& operation,
                                     const typename tnlParallelReductionAbsSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< int, int > >
                                   ( tnlParallelReductionAbsSum< int, int >& operation,
                                     const typename tnlParallelReductionAbsSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< float, int > >
                                   ( tnlParallelReductionAbsSum< float, int >& operation,
                                     const typename tnlParallelReductionAbsSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< double, int > >
                                   ( tnlParallelReductionAbsSum< double, int>& operation,
                                     const typename tnlParallelReductionAbsSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< long double, int > >
                                   ( tnlParallelReductionAbsSum< long double, int>& operation,
                                     const typename tnlParallelReductionAbsSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< long double, int> :: ResultType& result );
#endif

template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< char, long int > >
                                   ( tnlParallelReductionAbsSum< char, long int >& operation,
                                     const typename tnlParallelReductionAbsSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< char, long int > :: ResultType& result );

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< int, long int > >
                                   ( tnlParallelReductionAbsSum< int, long int >& operation,
                                     const typename tnlParallelReductionAbsSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< float, long int > >
                                   ( tnlParallelReductionAbsSum< float, long int >& operation,
                                     const typename tnlParallelReductionAbsSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< double, long int > >
                                   ( tnlParallelReductionAbsSum< double, long int>& operation,
                                     const typename tnlParallelReductionAbsSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< long double, long int > >
                                   ( tnlParallelReductionAbsSum< long double, long int>& operation,
                                     const typename tnlParallelReductionAbsSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionAbsSum< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL