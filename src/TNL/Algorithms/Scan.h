/***************************************************************************
                          Scan.h  -  description
                             -------------------
    begin                : May 9, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief Scan (or prefix sum) type - inclusive or exclusive.
 *
 * See \ref TNL::Algorithms::Scan.
 */
enum class ScanType {
   Exclusive,
   Inclusive
};

/**
 * \brief Computes scan (or prefix sum) on a vector.
 *
 * [Scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum) operation turns a sequence
 * \f$a_1, \ldots, a_n\f$ into a sequence \f$s_1, \ldots, s_n\f$ defined as
 *
 * \f[
 * s_i = \sum_{j=1}^i a_i.
 * \f]
 * Exclusive scan (or prefix sum) is defined as
 *
 * \f[
 * \sigma_i = \sum_{j=1}^{i-1} a_i.
 * \f]
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Type parameter says if inclusive or exclusive is scan is to be computed.
 *
 * See \ref Scan< Devices::Host, Type > and \ref Scan< Devices::Cuda, Type >.
 */
template< typename Device,
          ScanType Type = ScanType::Inclusive >
struct Scan;

/**
 * \brief Computes segmented scan (or prefix sum) on a vector.
 *
 * Segmented scan is a modification of common scan. In this case the sequence of
 * numbers in hand is divided into segments like this, for example
 *
 * ```
 * [1,3,5][2,4,6,9][3,5],[3,6,9,12,15]
 * ```
 *
 * and we want to compute inclusive or exclusive scan of each segment. For inclusive segmented prefix sum we get
 *
 * ```
 * [1,4,9][2,6,12,21][3,8][3,9,18,30,45]
 * ```
 *
 * and for exclusive segmented prefix sum it is
 *
 * ```
 * [0,1,4][0,2,6,12][0,3][0,3,9,18,30]
 * ```
 *
 * In addition to common scan, we need to encode the segments of the input sequence.
 * It is done by auxiliary flags array (it can be array of booleans) having `1` at the
 * beginning of each segment and `0` on all other positions. In our example, it would be like this:
 *
 * ```
 * [1,0,0,1,0,0,0,1,0,1,0,0, 0, 0]
 * [1,3,5,2,4,6,9,3,5,3,6,9,12,15]
 *
 * ```
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Type parameter says if inclusive or exclusive is scan is to be computed.
 *
 * See \ref Scan< Devices::Host, Type > and \ref Scan< Devices::Cuda, Type >.
 *
 * **Note: Segmented scan is not implemented for CUDA yet.**
 */
template< typename Device,
          ScanType Type = ScanType::Inclusive >
struct SegmentedScan;


template< ScanType Type >
struct Scan< Devices::Sequential, Type >
{
   /**
    * \brief Computes scan (prefix sum) sequentially.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/ScanExample.cpp
    *
    * \par Output
    *
    * \include ScanExample.out
    */
   template< typename Vector,
             typename Reduction >
   static void
   perform( Vector& v,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::ValueType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::ValueType zero );
};

template< ScanType Type >
struct Scan< Devices::Host, Type >
{
   /**
    * \brief Computes scan (prefix sum) using OpenMP.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/ScanExample.cpp
    *
    * \par Output
    *
    * \include ScanExample.out
    */
   template< typename Vector,
             typename Reduction >
   static void
   perform( Vector& v,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::ValueType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::ValueType zero );
};

template< ScanType Type >
struct Scan< Devices::Cuda, Type >
{
   /**
    * \brief Computes scan (prefix sum) on GPU.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/ScanExample.cpp
    *
    * \par Output
    *
    * \include ScanExample.out
    */
   template< typename Vector,
             typename Reduction >
   static void
   perform( Vector& v,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType zero );

   template< typename Vector,
             typename Reduction >
   static auto
   performFirstPhase( Vector& v,
                      const typename Vector::IndexType begin,
                      const typename Vector::IndexType end,
                      const Reduction& reduction,
                      const typename Vector::ValueType zero );

   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( Vector& v,
                       const BlockShifts& blockShifts,
                       const typename Vector::IndexType begin,
                       const typename Vector::IndexType end,
                       const Reduction& reduction,
                       const typename Vector::ValueType zero );
};

template< ScanType Type >
struct SegmentedScan< Devices::Sequential, Type >
{
   /**
    * \brief Computes segmented scan (prefix sum) sequentially.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    * \tparam Flags array type containing zeros and ones defining the segments begining
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param flags is an array with zeros and ones defining the segments begining
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SegmentedScanExample.cpp
    *
    * \par Output
    *
    * \include SegmentedScanExample.out
    */
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType zero );
};

template< ScanType Type >
struct SegmentedScan< Devices::Host, Type >
{
   /**
    * \brief Computes segmented scan (prefix sum) using OpenMP.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    * \tparam Flags array type containing zeros and ones defining the segments begining
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param flags is an array with zeros and ones defining the segments begining
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SegmentedScanExample.cpp
    *
    * \par Output
    *
    * \include SegmentedScanExample.out
    */
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType zero );
};

template< ScanType Type >
struct SegmentedScan< Devices::Cuda, Type >
{
   /**
    * \brief Computes segmented scan (prefix sum) on GPU.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    * \tparam Flags array type containing zeros and ones defining the segments begining
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param flags is an array with zeros and ones defining the segments begining
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param zero is the idempotent element for the reduction operation, i.e. element which
    *             does not change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SegmentedScanExample.cpp
    *
    * \par Output
    *
    * \include SegmentedScanExample.out
    *
    * **Note: Segmented scan is not implemented for CUDA yet.**
    */
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType zero );
};

} // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Scan.hpp>
