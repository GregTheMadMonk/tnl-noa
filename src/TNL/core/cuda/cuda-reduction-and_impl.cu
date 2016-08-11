/***************************************************************************
                          cuda-reduction-and_impl.cu  -  description
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
 * Logical AND
 */
template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< char, int > >
                                   ( tnlParallelReductionLogicalAnd< char, int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< char, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< char, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< char, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< char, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< int, int > >
                                   ( tnlParallelReductionLogicalAnd< int, int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< int, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< int, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< int, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< int, int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< float, int > >
                                   ( tnlParallelReductionLogicalAnd< float, int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< float, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< float, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< float, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< float, int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< double, int > >
                                   ( tnlParallelReductionLogicalAnd< double, int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< long double, int > >
                                   ( tnlParallelReductionLogicalAnd< long double, int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< long double, int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< long double, int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< char, long int > >
                                   ( tnlParallelReductionLogicalAnd< char, long int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< char, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< char, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< char, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< int, long int > >
                                   ( tnlParallelReductionLogicalAnd< int, long int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< int, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< int, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< int, long int > :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< float, long int > >
                                   ( tnlParallelReductionLogicalAnd< float, long int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< float, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< float, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< float, long int> :: ResultType& result );

template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< double, long int > >
                                   ( tnlParallelReductionLogicalAnd< double, long int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< long double, long int > >
                                   ( tnlParallelReductionLogicalAnd< long double, long int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< long double, long int > :: RealType* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< long double, long int > :: RealType* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< long double, long int> :: ResultType& result );
#endif
#endif
#endif

} // namespace TNL