/***************************************************************************
                          cuda-reduction-equalities_impl.cu  -  description
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
 * Equalities
 */
template bool reductionOnCudaDevice< tnlParallelReductionEqualities< char, int > >
                                   ( tnlParallelReductionEqualities< char, int >& operation,
                                     const typename tnlParallelReductionEqualities< char, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionEqualities< int, int > >
                                   ( tnlParallelReductionEqualities< int, int >& operation,
                                     const typename tnlParallelReductionEqualities< int, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionEqualities< float, int > >
                                   ( tnlParallelReductionEqualities< float, int >& operation,
                                     const typename tnlParallelReductionEqualities< float, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionEqualities< double, int > >
                                   ( tnlParallelReductionEqualities< double, int>& operation,
                                     const typename tnlParallelReductionEqualities< double, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionEqualities< long double, int > >
                                   ( tnlParallelReductionEqualities< long double, int>& operation,
                                     const typename tnlParallelReductionEqualities< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionEqualities< char, long int > >
                                   ( tnlParallelReductionEqualities< char, long int >& operation,
                                     const typename tnlParallelReductionEqualities< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionEqualities< int, long int > >
                                   ( tnlParallelReductionEqualities< int, long int >& operation,
                                     const typename tnlParallelReductionEqualities< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionEqualities< float, long int > >
                                   ( tnlParallelReductionEqualities< float, long int >& operation,
                                     const typename tnlParallelReductionEqualities< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionEqualities< double, long int > >
                                   ( tnlParallelReductionEqualities< double, long int>& operation,
                                     const typename tnlParallelReductionEqualities< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionEqualities< long double, long int > >
                                   ( tnlParallelReductionEqualities< long double, long int>& operation,
                                     const typename tnlParallelReductionEqualities< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionEqualities< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionEqualities< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL