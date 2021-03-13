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
#include <functional>  // reduction functions like std::plus, std::logical_and, std::logical_or etc.

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
   /**
    * \brief Computes reduction on CPU sequentialy.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam ReductionOperation is a lambda function performing the reduction.
    * \tparam DataFetcher is a lambda function for fetching the input data.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param reduction is a lambda function defining the reduction operation.
    * \param dataFetcher is a lambda function fetching the input data.
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    * \return result of the reduction
    *
    * The dataFetcher lambda function takes one argument which is index of the element to be fetched:
    *
    * ```
    * auto dataFetcher1 = [=] __cuda_callable__ ( Index i ) { return ... };
    * ```
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
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
             typename ReductionOperation,
             typename DataFetcher >
   static constexpr Result
   reduce( const Index begin,
           const Index end,
           const ReductionOperation& reduction,
           DataFetcher& dataFetcher,
           const Result& zero );

   /**
    * \brief Computes sequentially reduction on CPU and returns position of an element of interest.
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
    * \param reduction is a lambda function defining the reduction operation and managing the elements positions.
    * \param dataFetcher is a lambda function fetching the input data.
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    * \return result of the reduction in a form of std::pair< Index, Result> structure. `pair.first'
    *         is the element position and `pair.second` is the reduction result.
    *
    * The dataFetcher lambda function takes one argument which is index of the element to be fetched:
    *
    * ```
    * auto dataFetcher1 = [=] __cuda_callable__ ( Index i ) { return ... };
    * ```
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
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
   template< typename Index,
             typename Result,
             typename ReductionOperation,
             typename DataFetcher >
   static constexpr std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       const ReductionOperation& reduction,
                       DataFetcher& dataFetcher,
                       const Result& zero );
};

template<>
struct Reduction< Devices::Host >
{
   /**
    * \brief Computes reduction on CPU.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam ReductionOperation is a lambda function performing the reduction.
    * \tparam DataFetcher is a lambda function for fetching the input data.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param reduction is a lambda function defining the reduction operation.
    * \param dataFetcher is a lambda function fetching the input data.
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    * \return result of the reduction
    *
    * The dataFetcher lambda function takes one argument which is index of the element to be fetched:
    *
    * ```
    * auto dataFetcher1 = [=] __cuda_callable__ ( Index i ) { return ... };
    * ```
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
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
             typename ReductionOperation,
             typename DataFetcher >
   static Result
   reduce( const Index begin,
           const Index end,
           const ReductionOperation& reduction,
           DataFetcher& dataFetcher,
           const Result& zero );

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
    * \param reduction is a lambda function defining the reduction operation and managing the elements positions.
    * \param dataFetcher is a lambda function fetching the input data.
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    * \return result of the reduction in a form of std::pair< Index, Result> structure. `pair.first'
    *         is the element position and `pair.second` is the reduction result.
    * 
    * The dataFetcher lambda function takes one argument which is index of the element to be fetched:
    * 
    * ```
    * auto dataFetcher1 = [=] __cuda_callable__ ( Index i ) { return ... };
    * ```
    * 
    * The reduction lambda function takes two variables which are supposed to be reduced:
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
   template< typename Index,
             typename Result,
             typename ReductionOperation,
             typename DataFetcher >
   static std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       const ReductionOperation& reduction,
                       DataFetcher& dataFetcher,
                       const Result& zero );
};

template<>
struct Reduction< Devices::Cuda >
{
   /**
    * \brief Computes reduction on GPU.
    *
    * \tparam Index is a type for indexing.
    * \tparam Result is a type of the reduction result.
    * \tparam ReductionOperation is a lambda function performing the reduction.
    * \tparam DataFetcher is a lambda function for fetching the input data.
    *
    * \param begin defines range [begin, end) of indexes which will be used for the reduction.
    * \param end defines range [begin, end) of indexes which will be used for the reduction.
    * \param reduction is a lambda function defining the reduction operation.
    * \param dataFetcher is a lambda function fetching the input data.
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    * \return result of the reduction
    *
    * The dataFetcher lambda function takes one argument which is index of the element to be fetched:
    *
    * ```
    * auto dataFetcher1 = [=] __cuda_callable__ ( Index i ) { return ... };
    * ```
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
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
             typename ReductionOperation,
             typename DataFetcher >
   static Result
   reduce( const Index begin,
           const Index end,
           const ReductionOperation& reduction,
           DataFetcher& dataFetcher,
           const Result& zero );

   /**
    * \brief Computes reduction on GPU and returns position of an element of interest.
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
    * \param reduction is a lambda function defining the reduction operation and managing the elements positions.
    * \param dataFetcher is a lambda function fetching the input data.
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    * \return result of the reduction in a form of std::pair< Index, Result> structure. `pair.first'
    *         is the element position and `pair.second` is the reduction result.
    *
    * The dataFetcher lambda function takes one argument which is index of the element to be fetched:
    *
    * ```
    * auto dataFetcher1 = [=] __cuda_callable__ ( Index i ) { return ... };
    * ```
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
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
   template< typename Index,
             typename Result,
             typename ReductionOperation,
             typename DataFetcher >
   static std::pair< Result, Index >
   reduceWithArgument( const Index begin,
                       const Index end,
                       const ReductionOperation& reduction,
                       DataFetcher& dataFetcher,
                       const Result& zero );
};

} // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Reduction.hpp>
