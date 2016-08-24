/***************************************************************************
                          cuda-reduction-scalar-product_impl.cu  -  description
                             -------------------
    begin                : Jan 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Algorithms/reduction-operations.h>
#include <TNL/Containers/Algorithms/cuda-reduction.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * ScalarProduct
 */
template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< char, int > >
                                   ( tnlParallelReductionScalarProduct< char, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< int, int > >
                                   ( tnlParallelReductionScalarProduct< int, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< float, int > >
                                   ( tnlParallelReductionScalarProduct< float, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< double, int > >
                                   ( tnlParallelReductionScalarProduct< double, int>& operation,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< long double, int > >
                                   ( tnlParallelReductionScalarProduct< long double, int>& operation,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< char, long int > >
                                   ( tnlParallelReductionScalarProduct< char, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< int, long int > >
                                   ( tnlParallelReductionScalarProduct< int, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< float, long int > >
                                   ( tnlParallelReductionScalarProduct< float, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< double, long int > >
                                   ( tnlParallelReductionScalarProduct< double, long int>& operation,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< long double, long int > >
                                   ( tnlParallelReductionScalarProduct< long double, long int>& operation,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
