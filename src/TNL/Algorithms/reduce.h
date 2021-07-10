/***************************************************************************
                          reduce.h  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::pair, std::forward

#include <TNL/Functional.h>  // extension of STL functionals for reduction
#include <TNL/Algorithms/detail/Reduction.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief \e reduce implements [(parallel) reduction](https://en.wikipedia.org/wiki/Reduce_(parallel_pattern)) for vectors and arrays.
 *
 * Reduction can be used for operations having one or more vectors (or arrays) elements is input and returning
 * one number (or element) as output. Some examples of such operations can be vectors/arrays comparison,
 * vector norm, scalar product of two vectors or computing minimum or maximum. If one needs to know even
 * position of the smallest or the largest element, reduction with argument can be used.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Result is a type of the reduction result.
 * \tparam Fetch is a lambda function for fetching the input data.
 * \tparam Reduction is a lambda function performing the reduction.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation.
 * \param zero is the idempotent element for the reduction operation, i.e. element which
 *             does not change the result of the reduction.
 * \return result of the reduction
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * The `reduction` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/SumExampleWithLambda.cpp
 *
 * \par Output
 *
 * \include SumExampleWithLambda.out
 */
template< typename Device,
          typename Index,
          typename Result,
          typename Fetch,
          typename Reduction >
Result reduce( const Index begin,
               const Index end,
               Fetch&& fetch,
               Reduction&& reduction,
               const Result& zero )
{
   return detail::Reduction< Device >::reduce( begin,
                                               end,
                                               std::forward< Fetch >( fetch ),
                                               std::forward< Reduction >( reduction ),
                                               zero );
}

/**
 * \brief Variant of \ref TNL::Algorithms::reduce with functional instead of reduction lambda function.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Fetch is a lambda function for fetching the input data.
 * \tparam Reduction is a functional performing the reduction.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \e Reduction can be one of the following \ref TNL::Plus, \ref TNL::Multiplies, \ref TNL::Min, \ref TNL::Max, \ref TNL::LogicalAnd,
 *    \ref TNL::LogicalOr, \ref TNL::BitAnd or \ref TNL::BitOr.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation.
 * \return result of the reduction
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/SumExampleWithFunctional.cpp
 *
 * \par Output
 *
 * \include SumExampleWithFunctional.out
 */
template< typename Device,
          typename Index,
          typename Fetch,
          typename Reduction >
auto reduce( const Index begin,
             const Index end,
             Fetch&& fetch,
             Reduction&& reduction )
{
   using Result = std::decay_t< decltype( fetch( 0 ) ) >;
   return detail::Reduction< Device >::reduce( begin,
                                               end,
                                               std::forward< Fetch >( fetch ),
                                               std::forward< Reduction >( reduction ),
                                               reduction.template getIdempotent< Result >() );
}

/**
 * \brief Variant of \ref TNL::Algorithms::reduce returning also a position of an element of interest.
 *
 * For example in case of computing minimal or maximal element in array/vector,
 * the position of the element having given value can be obtained. The use of this method
 * is, however, more flexible.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Result is a type of the reduction result.
 * \tparam Reduction is a lambda function performing the reduction.
 * \tparam Fetch is a lambda function for fetching the input data.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation and managing the elements positions.
 * \param zero is the idempotent element for the reduction operation, i.e. element which
 *             does not change the result of the reduction.
 * \return result of the reduction in a form of std::pair< Index, Result> structure. `pair.first'
 *         is the element position and `pair.second` is the reduction result.
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * The `reduction` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/ReductionWithArgument.cpp
 *
 * \par Output
 *
 * \include ReductionWithArgument.out
 */
template< typename Device,
          typename Index,
          typename Result,
          typename Fetch,
          typename Reduction >
std::pair< Result, Index >
reduceWithArgument( const Index begin,
                    const Index end,
                    Fetch&& fetch,
                    Reduction&& reduction,
                    const Result& zero )
{
   return detail::Reduction< Device >::reduceWithArgument( begin,
                                                           end,
                                                           std::forward< Fetch >( fetch ),
                                                           std::forward< Reduction >( reduction ),
                                                           zero );
}

/**
 * \brief Variant of \ref TNL::Algorithms::reduceWithArgument with functional instead of reduction lambda function.
 *
 * For example in case of computing minimal or maximal element in array/vector,
 * the position of the element having given value can be obtained. The use of this method
 * is, however, more flexible.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Result is a type of the reduction result.
 * \tparam Reduction is a functional performing the reduction.
 * \tparam Fetch is a lambda function for fetching the input data.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \e Reduction can be one of \ref TNL::MinWithArg, \ref TNL::MaxWithArg.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation and managing the elements positions.
 * \param zero is the idempotent element for the reduction operation, i.e. element which
 *             does not change the result of the reduction.
 * \return result of the reduction in a form of std::pair< Index, Result> structure. `pair.first'
 *         is the element position and `pair.second` is the reduction result.
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * The `reduction` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/ReductionWithArgumentWithFunctional.cpp
 *
 * \par Output
 *
 * \include ReductionWithArgumentWithFunctional.out
 */
template< typename Device,
          typename Index,
          typename Fetch,
          typename Reduction >
auto
reduceWithArgument( const Index begin,
                    const Index end,
                    Fetch&& fetch,
                    Reduction&& reduction )
{
   using Result = std::decay_t< decltype( fetch( 0 ) ) >;
   return detail::Reduction< Device >::reduceWithArgument( begin,
                                                           end,
                                                           std::forward< Fetch >( fetch ),
                                                           std::forward< Reduction >( reduction ),
                                                           reduction.template getIdempotent< Result >() );
}

} // namespace Algorithms
} // namespace TNL
