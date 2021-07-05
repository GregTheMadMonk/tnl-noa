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

/*template< typename Value,
          int size = sizeof( Value ) >
struct AllBitsTrue
{
   static constexpr Value aux = AllBitsTrue< Value, size - 1 >::value << 8;
   static constexpr Value value = ( Value ) aux | 0xff;
};

template< typename Value >
struct AllBitsTrue< Value, 1 >
{
   static constexpr Value value = ( Value ) 0xff;
};

template< typename Value,
          int size = sizeof( Value ) >
struct AllBitsFalse
{
   static constexpr Value aux = AllBitsFalse< Value, size - 1 >::value << 8;
   static constexpr Value value = ( Value ) aux | 0x00;
};

template< typename Value >
struct AllBitsFalse< Value, 1 >
{
   static constexpr Value value = ( Value ) 0x00;
};*/


template< typename Value = void >
struct Plus
{
   using ValueType = Value;

   static constexpr Value getIdempotent() { return ( Value ) 0; };

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs + rhs; }
};

template<>
struct Plus< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) 0; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs + rhs; }
};

template< typename Value = void >
struct Multiplies
{
   using ValueType = Value;

   static constexpr ValueType idempotent = 1;

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs * rhs; }
};

template<>
struct Multiplies< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) 1; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs * rhs; }
};


template< typename Value = void >
struct Min
{
   using ValueType = Value;

   static constexpr ValueType idempotent = std::numeric_limits< Value >::max();

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs < rhs ? lhs : rhs; }
};

template<>
struct Min< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::max(); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs < rhs ? lhs : rhs; }
};


template< typename Value = void >
struct Max
{
   using ValueType = Value;

   static constexpr ValueType idempotent = std::numeric_limits< Value >::min();

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs > rhs ? lhs : rhs; }
};

template<>
struct Max< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return std::numeric_limits< T >::min(); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs > rhs ? lhs : rhs; }
};

template< typename Value = void >
struct LogicalAnd
{
   using ValueType = Value;

   static constexpr ValueType idempotent = ( Value ) true;

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs && rhs; }
};

template<>
struct LogicalAnd< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) true; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs && rhs; }
};

template< typename Value = void >
struct LogicalOr
{
   using ValueType = Value;

   static constexpr ValueType idempotent = ( Value ) false;

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs || rhs; }
};
template<>
struct LogicalOr< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ( T ) false; };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs || rhs; }
};


template< typename Value = void >
struct BitAnd
{
   using ValueType = Value;

   static constexpr ValueType idempotent = ~static_cast< ValueType >( 0 );

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs & rhs; }
};

template<>
struct BitAnd< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return ~static_cast< T >( 0 ); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs & rhs; }
};

template< typename Value = void >
struct BitOr
{
   using ValueType = Value;

   static constexpr ValueType idempotent =  static_cast< ValueType >( 0 );

   constexpr Value operator()( const Value& lhs, const Value& rhs ) { return lhs | rhs; }
};

template<>
struct BitOr< void >
{
   template< typename T >
   static constexpr T getIdempotent() { return static_cast< T >( 0 ); };

   template< typename T >
   constexpr T operator()( const T& lhs, const T& rhs ) { return lhs | rhs; }
};


} // namespace TNL
