/***************************************************************************
                          Functional.h  -  description
                             -------------------
    begin                : Juyl 1, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <limits>

namespace TNL {

/**
 * \brief Replacement of std::plus which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct Plus
{
   using ValueType = Value;

   static constexpr Value getIdempotent() { return ( Value ) 0; };

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs + rhs; }
};

/**
 * \brief Replacement of std::plus which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct Plus< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) 0; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs + rhs; }
};

/**
 * \brief Replacement of std::multiplies which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct Multiplies
{
   using ValueType = Value;

   static constexpr ValueType idempotent = 1;

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs * rhs; }
};

/**
 * \brief Replacement of std::multiplies which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct Multiplies< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) 1; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs * rhs; }
};

/**
 * \brief Replacement of std::min which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct Min
{
   using ValueType = Value;

   static constexpr ValueType idempotent = std::numeric_limits< Value >::max();

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs < rhs ? lhs : rhs; }
};

/**
 * \brief Replacement of std::min which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct Min< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::max(); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs < rhs ? lhs : rhs; }
};


/**
 * \brief Replacement of std::max which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct Max
{
   using ValueType = Value;

   static constexpr ValueType idempotent = std::numeric_limits< Value >::min();

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs > rhs ? lhs : rhs; }
};

/**
 * \brief Replacement of std::max which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct Max< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::min(); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs > rhs ? lhs : rhs; }
};

/**
 * \brief Replacement of std::min which is optimized for use with \ref TNL::Algorithms::reduceWithArgument.
 *
 * \tparam Value is data type.
 */
template< typename Value = void, typename Index = void >
struct MinWithArg
{
   using ValueType = Value;

   static constexpr ValueType idempotent = std::numeric_limits< Value >::max();

   constexpr void operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx )
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
 * \brief Replacement of std::min which is optimized for use with \ref TNL::Algorithms::reduceWithArgument.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct MinWithArg< void, void >
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::max(); };

   template< typename Value, typename Index >
   constexpr void operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx )
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
 * \brief Replacement of std::max which is optimized for use with \ref TNL::Algorithms::reduceWithArgument.
 *
 * \tparam Value is data type.
 */
template< typename Value = void, typename Index = void >
struct MaxWithArg
{
   using ValueType = Value;

   static constexpr ValueType idempotent = std::numeric_limits< Value >::min();

   constexpr void operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx )
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
 * \brief Replacement of std::max which is optimized for use with \ref TNL::Algorithms::reduceWithArgument.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct MaxWithArg< void, void >
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::min(); };

   template< typename Value, typename Index >
   constexpr void operator()( Value& lhs, const Value& rhs, Index& lhsIdx, const Index& rhsIdx )
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
 * \brief Replacement of std::logical_and which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct LogicalAnd
{
   using ValueType = Value;

   static constexpr ValueType idempotent = ( Value ) true;

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs && rhs; }
};

/**
 * \brief Replacement of std::logical_and which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct LogicalAnd< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) true; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs && rhs; }
};

/**
 * \brief Replacement of std::logical_or which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct LogicalOr
{
   using ValueType = Value;

   static constexpr ValueType idempotent = ( Value ) false;

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs || rhs; }
};

/**
 * \brief Replacement of std::logical_or which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct LogicalOr< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) false; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs || rhs; }
};


/**
 * \brief Replacement of std::bit_and which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct BitAnd
{
   using ValueType = Value;

   static constexpr ValueType idempotent = ~static_cast< ValueType >( 0 );

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs & rhs; }
};

/**
 * \brief Replacement of std::bit_and which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct BitAnd< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ~static_cast< T >( 0 ); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs & rhs; }
};

/**
 * \brief Replacement of std::bit_or which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * \tparam Value is data type.
 */
template< typename Value = void >
struct BitOr
{
   using ValueType = Value;

   static constexpr ValueType idempotent =  static_cast< ValueType >( 0 );

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs | rhs; }
};

/**
 * \brief Replacement of std::bit_or which is optimized for use with \ref TNL::Algorithms::reduce.
 *
 * This is specialization for void type. The real type is deduced just when operator() is evoked.
 */
template<>
struct BitOr< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return static_cast< T >( 0 ); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs | rhs; }
};

} // namespace TNL
