/***************************************************************************
                          scan.h  -  description
                             -------------------
    begin                : Jul 11, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::forward

#include <TNL/Algorithms/detail/Scan.h>
#include <TNL/Functional.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief Computes an inclusive scan (or prefix sum) of an array in-place.
 *
 * [Inclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$s_1, \ldots, s_n\f$ defined as
 *
 * \f[
 * s_i = \sum_{j=1}^i a_i.
 * \f]
 *
 * \tparam Array type of the array to be scanned
 * \tparam Reduction type of the reduction functor
 *
 * \param array input array, the result of scan is stored in the same array
 * \param begin the first element in the array to be scanned
 * \param end the last element in the array to be scanned
 * \param reduction functor implementing the reduction operation
 * \param zero is the idempotent element for the reduction operation, i.e. element which
 *             does not change the result of the reduction.
 *
 * The reduction functor takes two variables to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/inplaceInclusiveScanExample.cpp
 *
 * \par Output
 *
 * \include inplaceInclusiveScanExample.out
 */
template< typename Array,
          typename Reduction >
void
inplaceInclusiveScan( Array& array,
                      typename Array::IndexType begin,
                      typename Array::IndexType end,
                      Reduction&& reduction,
                      typename Array::ValueType zero )
{
   using Scan = detail::Scan< typename Array::DeviceType, detail::ScanType::Inclusive >;
   Scan::perform( array, begin, end, std::forward< Reduction >( reduction ), zero );
}

/**
 * \brief Overload of \ref inplaceInclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The idempotent value is taken as `reduction.template getIdempotent< typename Array::ValueType >()`.
 * See \ref inplaceInclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `array.getSize()`.
 */
template< typename Array,
          typename Reduction = TNL::Plus >
void
inplaceInclusiveScan( Array& array,
                      typename Array::IndexType begin = 0,
                      typename Array::IndexType end = 0,
                      Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = array.getSize();
   constexpr typename Array::ValueType zero = Reduction::template getIdempotent< typename Array::ValueType >();
   inplaceInclusiveScan( array, begin, end, std::forward< Reduction >( reduction ), zero );
}

/**
 * \brief Computes an exclusive scan (or prefix sum) of an array in-place.
 *
 * [Exclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$\sigma_1, \ldots, \sigma_n\f$ defined as
 *
 * \f[
 * \sigma_i = \sum_{j=1}^{i-1} a_i.
 * \f]
 *
 * \tparam Array type of the array to be scanned
 * \tparam Reduction type of the reduction functor
 *
 * \param array input array, the result of scan is stored in the same array
 * \param begin the first element in the array to be scanned
 * \param end the last element in the array to be scanned
 * \param reduction functor implementing the reduction operation
 * \param zero is the idempotent element for the reduction operation, i.e. element which
 *             does not change the result of the reduction.
 *
 * The reduction functor takes two variables to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/inplaceExclusiveScanExample.cpp
 *
 * \par Output
 *
 * \include inplaceExclusiveScanExample.out
 */
template< typename Array,
          typename Reduction >
void
inplaceExclusiveScan( Array& array,
                      typename Array::IndexType begin,
                      typename Array::IndexType end,
                      Reduction&& reduction,
                      typename Array::ValueType zero )
{
   using Scan = detail::Scan< typename Array::DeviceType, detail::ScanType::Exclusive >;
   Scan::perform( array, begin, end, std::forward< Reduction >( reduction ), zero );
}

/**
 * \brief Overload of \ref inplaceExclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The idempotent value is taken as `reduction.template getIdempotent< typename Array::ValueType >()`.
 * See \ref inplaceExclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `array.getSize()`.
 */
template< typename Array,
          typename Reduction = TNL::Plus >
void
inplaceExclusiveScan( Array& array,
                      typename Array::IndexType begin = 0,
                      typename Array::IndexType end = 0,
                      Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = array.getSize();
   constexpr typename Array::ValueType zero = Reduction::template getIdempotent< typename Array::ValueType >();
   inplaceExclusiveScan( array, begin, end, std::forward< Reduction >( reduction ), zero );
}

} // namespace Algorithms
} // namespace TNL
