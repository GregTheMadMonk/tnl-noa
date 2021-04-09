/***************************************************************************
                          staticFor.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>
#include <type_traits>

namespace TNL {
namespace Algorithms {

namespace detail {

// special dispatch for `begin >= end` (i.e. empty loop)
template< typename Index, Index begin, Index end,  typename Func >
constexpr std::enable_if_t< (begin >= end) >
static_for_dispatch( Func &&f )
{}

#if __cplusplus >= 201703L

// C++17 version using fold expression
template< typename Index, Index begin,  typename Func, Index... idx >
constexpr void static_for_impl( Func &&f, std::integer_sequence< Index, idx... > )
{
   ( f( std::integral_constant<Index, begin + idx>{} ), ... );
}

// general dispatch for `begin < end`
template< typename Index, Index begin, Index end,  typename Func >
constexpr std::enable_if_t< (begin < end) >
static_for_dispatch( Func &&f )
{
   static_for_impl< Index, begin >(
         std::forward< Func >( f ),
         std::make_integer_sequence< Index, end - begin >{}
   );
}

#else

// C++14 version using recursive folding
// (We avoid manual folding with std::integer_sequence, because it cannot be
// empty, so it would be rather weird. Folding is done by bisection to limit
// the recursion depth.)

// special dispatch for 1 iteration
template< typename Index, Index begin, Index end,  typename Func >
constexpr std::enable_if_t< (begin < end && end - begin == 1) >
static_for_dispatch( Func &&f )
{
   f( std::integral_constant< Index, begin >{} );
}

// general dispatch for at least 2 iterations
template< typename Index, Index begin, Index end,  typename Func >
constexpr std::enable_if_t< (begin < end && end - begin >= 2) >
static_for_dispatch( Func &&f )
{
   constexpr Index mid = begin + (end - begin) / 2;
   static_for_dispatch< Index, begin, mid >( std::forward< Func >( f ) );
   static_for_dispatch< Index, mid, end >( std::forward< Func >( f ) );
}

#endif

} // namespace detail

/**
 * \brief Generic loop with constant bounds and indices usable in constant
 * expressions.
 *
 * \e staticFor is a generic C++14/C++17 implementation of a static for-loop
 * using \e constexpr functions and template metaprogramming. It is equivalent
 * to executing a function $f(i)$ for arguments $i$ from the integral range
 * `[begin, end)`, but with the type \ref std::integral_constant rather than
 * `int` or `std::size_t` representing the indices. Hence, each index has its
 * own distinct C++ type and the \e value of the index can be deduced from the
 * type.
 *
 * Also note that thanks to `constexpr`, the argument $i$ can be used in
 * constant expressions and the \e staticFor function can be used from the host
 * code as well as CUDA kernels (TNL requires the `--expt-relaxed-constexpr`
 * parameter when compiled by `nvcc`).
 *
 * \tparam Index is the type of the loop indices.
 * \tparam begin is the left bound of the iteration range `[begin, end)`.
 * \tparam end is the right bound of the iteration range `[begin, end)`.
 * \tparam Func is the type of the functor (it is usually deduced from the
 *    argument used in the function call).
 *
 * \param f is the functor to be called in each iteration.
 *
 * \par Example
 * \include Algorithms/staticForExample.cpp
 * \par Output
 * \include staticForExample.out
 */
template< typename Index, Index begin, Index end,  typename Func >
constexpr void staticFor( Func&& f )
{
   detail::static_for_dispatch< Index, begin, end >( std::forward< Func >( f ) );
}

} // namespace Algorithms
} // namespace TNL
