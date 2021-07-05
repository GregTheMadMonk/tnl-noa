/***************************************************************************
                          Reduction.h  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::pair
#include <functional>        // reduction functions like std::plus, std::logical_and, std::logical_or etc. - deprecated

#include <TNL/Functional.h>  // replacement of STL functional
#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief Reduction implements [(parallel) reduction](https://en.wikipedia.org/wiki/Reduce_(parallel_pattern)) for vectors and arrays.
 *
 * Reduction can be used for operations having one or more vectors (or arrays) elements is input and returning
 * one number (or element) as output. Some examples of such operations can be vectors/arrays comparison,
 * vector norm, scalar product of two vectors or computing minimum or maximum. If one needs to know even
 * position of the smallest or the largest element, reduction with argument can be used.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 *
 * See \ref Reduction< Devices::Host > and \ref Reduction< Devices::Cuda >.
 */
template< typename Device >
struct Reduction;

template<>
struct Reduction< Devices::Sequential >
{
   using DeviceType = Devices::Sequential;

   /**
    * \brief Computes reduction on CPU sequentially.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam Fetch is a lambda function for fetching the input data.
    * \tparam Reduce is a lambda function performing the reduction.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param fetch is a lambda function fetching the input data.
    * \param reduce is a lambda function defining the reduction operation.
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
    * The `reduce` lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SumExample.cpp
    *
    * \par Output
    *
    * \include SumExample.out
    */
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static constexpr Result
   reduce( const Index begin,
           const Index end,
           Fetch&& fetch,
           Reduce&& reduce,
           const Result& zero = Reduce::template getIdempotent< DeviceType >() );

   /**
    * \brief Computes sequentially reduction on CPU and returns position of an element of interest.
    *
    * For example in case of computing minimal or maximal element in array/vector,
    * the position of the element having given value can be obtained. The use of this method
    * is, however, more flexible.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam Fetch is a lambda function for fetching the input data.
    * \tparam Reduce is a lambda function performing the reduction.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param fetch is a lambda function fetching the input data.
    * \param reduce is a lambda function defining the reduction operation and managing the elements positions.
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
    * The `reduce` lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
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
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static constexpr std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       const Result& zero = Reduce::template getIdempotent< DeviceType >() );
};

template<>
struct Reduction< Devices::Host >
{
   using DeviceType = Devices::Host;

   /**
    * \brief Computes reduction on CPU.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam Fetch is a lambda function for fetching the input data.
    * \tparam Reduce is a lambda function performing the reduction.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param fetch is a lambda function fetching the input data.
    * \param reduce is a lambda function defining the reduction operation.
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
    * The `reduce` lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SumExample.cpp
    *
    * \par Output
    *
    * \include SumExample.out
    */
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static Result
   reduce( const Index begin,
           const Index end,
           Fetch&& fetch,
           Reduce&& reduce,
           const Result& zero = Reduce::template getIdempotent< DeviceType >() );

   /*template< typename Index,
             typename Fetch,
             typename Reduce_ >
   static auto
   reduce( const Index begin,
           const Index end,
           Fetch&& fetch,
           Reduce_&& reduce_ ) -> decltype( fetch( ( Index ) 0 ) )
   {
      using Result = decltype( fetch( ( Index ) 0 ) );
      return reduce( begin, end, fetch, reduce_, std::remove_reference< Reduce_ >::type::template getIdempotent< Result >() );
   };*/


   /**
    * \brief Computes reduction on CPU and returns position of an element of interest.
    *
    * For example in case of computing minimal or maximal element in array/vector,
    * the position of the element having given value can be obtained. The use of this method
    * is, however, more flexible.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam ReductionOperation is a lambda function performing the reduction.
    * \tparam DataFetcher is a lambda function for fetching the input data.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param fetch is a lambda function fetching the input data.
    * \param reduce is a lambda function defining the reduction operation and managing the elements positions.
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
    * The `reduce` lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
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
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       const Result& zero = Reduce::template getIdempotent< DeviceType >() );
};

template<>
struct Reduction< Devices::Cuda >
{
   using DeviceType = Devices::Cuda;

   /**
    * \brief Computes reduction on GPU.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam Fetch is a lambda function for fetching the input data.
    * \tparam Reduce is a lambda function performing the reduction.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param fetch is a lambda function fetching the input data.
    * \param reduce is a lambda function defining the reduction operation.
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
    * The `reduce` lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SumExample.cpp
    *
    * \par Output
    *
    * \include SumExample.out
    */
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static Result
   reduce( const Index begin,
           const Index end,
           Fetch&& fetch,
           Reduce&& reduce,
           const Result& zero = Reduce::template getIdempotent< DeviceType >() );

   /**
    * \brief Computes reduction on GPU and returns position of an element of interest.
    *
    * For example in case of computing minimal or maximal element in array/vector,
    * the position of the element having given value can be obtained. The use of this method
    * is, however, more flexible.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam Fetch is a lambda function for fetching the input data.
    * \tparam Reduce is a lambda function performing the reduction.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param fetch is a lambda function fetching the input data.
    * \param reduce is a lambda function defining the reduction operation and managing the elements positions.
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
    * The `reduce` lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
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
   template< typename Index,
             typename Result,
             typename Fetch,
             typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       const Result& zero = Reduce::template getIdempotent< DeviceType >() );
};

/**
 * \brief Reduction implements [(parallel) reduction](https://en.wikipedia.org/wiki/Reduce_(parallel_pattern)) for vectors and arrays.
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
 * \tparam Reduce is a lambda function performing the reduction.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduce is a lambda function defining the reduction operation.
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
 * The `reduce` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
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
          typename Reduce >
Result reduce( const Index begin,
               const Index end,
               Fetch&& fetch,
               Reduce&& reduce,
               const Result& zero )
{
    return Reduction< Device >::reduce( begin, end, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ), zero );
}

/**
 * \brief Variant of \ref TNL::Algorithms::reduce with functional instead of reduction lambda function.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Fetch is a lambda function for fetching the input data.
 * \tparam Reduce is a functional performing the reduction.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \e Reduce can be one of the following \ref TNL::Plus, \ref TNL::Multiplies, \ref TNL::Min, \ref TNL::Max, \ref TNL::LogicalAnd,
 *    \ref TNL::LogicalOr, \ref TNL::BitAnd or \ref TNL::BitOr.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduce is a lambda function defining the reduction operation.
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
          typename Reduce >
auto reduce( const Index begin,
             const Index end,
             Fetch&& fetch,
             Reduce&& reduce ) -> decltype( Reduction< Device >::reduce( begin, end, std::forward< Fetch >( fetch ), std::forward< Reduce >( reduce ),
                                                   std::remove_reference< Reduce >::type::template getIdempotent< decltype( fetch( ( Index ) 0 ) ) >() ) )
{
   using Result = decltype( fetch( ( Index ) 0 ) );
   return Reduction< Device >::reduce( begin,
                                       end,
                                       std::forward< Fetch >( fetch ),
                                       std::forward< Reduce >( reduce ),
                                       std::remove_reference< Reduce >::type::template getIdempotent< Result >() );
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
 * \tparam Reduce is a lambda function performing the reduction.
 * \tparam Fetch is a lambda function for fetching the input data.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduce is a lambda function defining the reduction operation and managing the elements positions.
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
 * The `reduce` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
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
          typename Reduce >
std::pair< Result, Index >
reduceWithArgument( const Index begin,
                    const Index end,
                    Fetch&& fetch,
                    Reduce&& reduce,
                    const Result& zero )
{
    return Reduction< Device >::reduceWithArgument( begin,
                                                    end,
                                                    std::forward< Fetch >( fetch ),
                                                    std::forward< Reduce >( reduce ),
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
 * \tparam Reduce is a functional performing the reduction.
 * \tparam Fetch is a lambda function for fetching the input data.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential, \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \e Reduce can be one of \ref TNL::MinWithArg, \ref TNL::MaxWithArg.
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduce is a lambda function defining the reduction operation and managing the elements positions.
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
 * The `reduce` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduce = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
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
          typename Reduce >
auto
reduceWithArgument( const Index begin,
                    const Index end,
                    Fetch&& fetch,
                    Reduce&& reduce ) -> decltype( Reduction< Device >::reduceWithArgument( begin, end, fetch, reduce,
                                                   std::remove_reference< Reduce >::type::template getIdempotent<  decltype( fetch( ( Index ) 0 ) ) >() ) )
{
   using Result = decltype( fetch( ( Index ) 0 ) );
   return Reduction< Device >::reduceWithArgument( begin,
                                                   end,
                                                   std::forward< Fetch >( fetch ),
                                                   std::forward< Reduce >( reduce ),
                                                   std::remove_reference< Reduce >::type::template getIdempotent< Result >() );
}


} // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Reduction.hpp>
