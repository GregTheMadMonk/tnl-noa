/***************************************************************************
                          cuda-reduction-or_impl.cu  -  description
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
 * Logical OR
 */
template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< char, int > >
                                   ( tnlParallelReductionLogicalOr< char, int >& operation,
                                     const typename tnlParallelReductionLogicalOr< char, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< int, int > >
                                   ( tnlParallelReductionLogicalOr< int, int >& operation,
                                     const typename tnlParallelReductionLogicalOr< int, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< float, int > >
                                   ( tnlParallelReductionLogicalOr< float, int >& operation,
                                     const typename tnlParallelReductionLogicalOr< float, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< double, int > >
                                   ( tnlParallelReductionLogicalOr< double, int>& operation,
                                     const typename tnlParallelReductionLogicalOr< double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< long double, int > >
                                   ( tnlParallelReductionLogicalOr< long double, int>& operation,
                                     const typename tnlParallelReductionLogicalOr< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< char, long int > >
                                   ( tnlParallelReductionLogicalOr< char, long int >& operation,
                                     const typename tnlParallelReductionLogicalOr< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< int, long int > >
                                   ( tnlParallelReductionLogicalOr< int, long int >& operation,
                                     const typename tnlParallelReductionLogicalOr< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< float, long int > >
                                   ( tnlParallelReductionLogicalOr< float, long int >& operation,
                                     const typename tnlParallelReductionLogicalOr< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< double, long int > >
                                   ( tnlParallelReductionLogicalOr< double, long int>& operation,
                                     const typename tnlParallelReductionLogicalOr< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< long double, long int > >
                                   ( tnlParallelReductionLogicalOr< long double, long int>& operation,
                                     const typename tnlParallelReductionLogicalOr< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace TNL