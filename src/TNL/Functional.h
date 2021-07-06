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
#include <algorithm>
#include <limits>

namespace TNL {

/**
 * \brief Extension of std::plus for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct Plus : public std::plus< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) 0; };
};

/**
 * \brief Extension of std::multiplies for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct Multiplies : public std::multiplies< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) 1; };
};

/**
 * \brief Extension of std::min for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct Min
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::max(); };

   template< typename Value >
   constexpr Value operator()( const Value& lhs, const Value& rhs ) const { return lhs < rhs ? lhs : rhs; }
};

/**
 * \brief Extension of std::max for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct Max
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::min(); };

   template< typename Value >
   constexpr Value operator()( const Value& lhs, const Value& rhs ) const { return lhs > rhs ? lhs : rhs; }
};

/**
 * \brief Extension of std::min for use with \ref TNL::Algorithms::reduceWithArgument.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct MinWithArg
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::max(); };

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
 * \brief Extension of std::max for use with \ref TNL::Algorithms::reduceWithArgument.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct MaxWithArg
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::min(); };

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
 * \brief Extension of std::logical_and for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct LogicalAnd : public std::logical_and< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) true; };
};

/**
 * \brief Extension of std::logical_or for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct LogicalOr : public std::logical_or< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) false; };
};

/**
 * \brief Extension of std::bit_and for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct BitAnd : public std::bit_and< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ~static_cast< T >( 0 ); };
};

/**
 * \brief Extension of std::bit_or for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
struct BitOr : public std::bit_or< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return static_cast< T >( 0 ); };
};

} // namespace TNL
