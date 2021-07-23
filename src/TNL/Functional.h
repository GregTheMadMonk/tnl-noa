/***************************************************************************
                          Functional.h  -  description
                             -------------------
    begin                : Juyl 1, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <functional>
#include <limits>

#include <TNL/Math.h>

namespace TNL {

/**
 * \brief Extension of \ref std::plus<void> for use with \ref TNL::Algorithms::reduce.
 */
struct Plus : public std::plus< void >
{
   template< typename T >
   static constexpr T getIdentity() { return 0; }
};

/**
 * \brief Extension of \ref std::multiplies<void> for use with \ref TNL::Algorithms::reduce.
 */
struct Multiplies : public std::multiplies< void >
{
   template< typename T >
   static constexpr T getIdentity() { return 1; }
};

/**
 * \brief Function object implementing `min(x, y)` for use with \ref TNL::Algorithms::reduce.
 */
struct Min
{
   template< typename T >
   static constexpr T getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::max();
   }

   template< typename Value >
   constexpr Value operator()( const Value& lhs, const Value& rhs ) const
   {
      // use argument-dependent lookup and make TNL::min available for unqualified calls
      using TNL::min;
      return min( lhs, rhs );
   }
};

/**
 * \brief Function object implementing `max(x, y)` for use with \ref TNL::Algorithms::reduce.
 */
struct Max
{
   template< typename T >
   static constexpr T getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::lowest();
   }

   template< typename Value >
   constexpr Value operator()( const Value& lhs, const Value& rhs ) const
   {
      // use argument-dependent lookup and make TNL::max available for unqualified calls
      using TNL::max;
      return max( lhs, rhs );
   }
};

/**
 * \brief Extension of \ref std::min<void> for use with \ref TNL::Algorithms::reduceWithArgument.
 */
struct MinWithArg
{
   template< typename T >
   static constexpr T getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::max();
   }

   template< typename Value, typename Index >
   constexpr void operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx ) const
   {
      if( lhs > rhs )
      {
         lhs = rhs;
         lhsIdx = rhsIdx;
      }
      else if( lhs == rhs && rhsIdx < lhsIdx )
      {
         lhsIdx = rhsIdx;
      }
   }
};

/**
 * \brief Extension of \ref std::max<void> for use with \ref TNL::Algorithms::reduceWithArgument.
 */
struct MaxWithArg
{
   template< typename T >
   static constexpr T getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::lowest();
   }

   template< typename Value, typename Index >
   constexpr void operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx ) const
   {
      if( lhs < rhs )
      {
         lhs = rhs;
         lhsIdx = rhsIdx;
      }
      else if( lhs == rhs && rhsIdx < lhsIdx )
      {
         lhsIdx = rhsIdx;
      }
   }
};

/**
 * \brief Extension of \ref std::logical_and<void> for use with \ref TNL::Algorithms::reduce.
 */
struct LogicalAnd : public std::logical_and< void >
{
   template< typename T >
   static constexpr T getIdentity()
   {
      static_assert( std::numeric_limits< T >::is_specialized,
                     "std::numeric_limits is not specialized for the requested type" );
      return std::numeric_limits< T >::max();
   }
};

/**
 * \brief Extension of \ref std::logical_or<void> for use with \ref TNL::Algorithms::reduce.
 */
struct LogicalOr : public std::logical_or< void >
{
   template< typename T >
   static constexpr T getIdentity() { return 0; }
};

/**
 * \brief Extension of \ref std::bit_and<void> for use with \ref TNL::Algorithms::reduce.
 */
struct BitAnd : public std::bit_and< void >
{
   template< typename T >
   static constexpr T getIdentity() { return ~static_cast< T >( 0 ); }
};

/**
 * \brief Extension of \ref std::bit_or<void> for use with \ref TNL::Algorithms::reduce.
 */
struct BitOr : public std::bit_or< void >
{
   template< typename T >
   static constexpr T getIdentity() { return 0; }
};

} // namespace TNL
