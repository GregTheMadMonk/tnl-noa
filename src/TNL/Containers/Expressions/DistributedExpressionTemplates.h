/***************************************************************************
                          DistributedExpressionTemplates.h  -  description
                             -------------------
    begin                : Jun 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once
#include <utility>

#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Containers/Expressions/DistributedComparison.h>
#include <TNL/Containers/Expressions/DistributedVerticalOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

////
// Distributed unary expression template
template< typename T1,
          template< typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct DistributedUnaryExpressionTemplate
{};

template< typename T1,
          template< typename > class Operation,
          ExpressionVariableType T1Type >
struct IsExpressionTemplate< DistributedUnaryExpressionTemplate< T1, Operation, T1Type > >
: std::true_type
{};

////
// Distributed binary expression template
template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct DistributedBinaryExpressionTemplate
{};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type,
          ExpressionVariableType T2Type >
struct IsExpressionTemplate< DistributedBinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation< typename T1::RealType, typename T2::RealType >::
                              evaluate( std::declval<T1>()[0], std::declval<T2>()[0] ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using CommunicatorType = typename T1::CommunicatorType;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< typename T1::ConstLocalViewType,
                                                        typename T2::ConstLocalViewType,
                                                        Operation >;

   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value,
                  "Attempt to mix operands which have different DeviceType." );
   static_assert( IsStaticArrayType< T1 >::value == IsStaticArrayType< T2 >::value,
                  "Attempt to mix static and non-static operands in binary expression templates." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b )
   {
      TNL_ASSERT_EQ( op1.getSize(), op2.getSize(),
                     "Attempt to mix operands with different sizes." );
      TNL_ASSERT_EQ( op1.getLocalRange(), op2.getLocalRange(),
                     "Distributed expressions are supported only on vectors which are distributed the same way." );
      TNL_ASSERT_EQ( op1.getCommunicationGroup(), op2.getCommunicationGroup(),
                     "Distributed expressions are supported only on vectors within the same communication group." );
   }

   RealType getElement( const IndexType i ) const
   {
      return getConstLocalView().getElement( i );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType getSize() const
   {
      return op1.getSize();
   }

   LocalRangeType getLocalRange() const
   {
      return op1.getLocalRange();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return op1.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2.getConstLocalView() );
   }

protected:
   const T1& op1;
   const T2& op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using RealType = decltype( Operation< typename T1::RealType, T2 >::
                              evaluate( std::declval<T1>()[0], std::declval<T2>() ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using CommunicatorType = typename T1::CommunicatorType;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< typename T1::ConstLocalViewType, T2, Operation >;

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      return getConstLocalView().getElement( i );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType getSize() const
   {
      return op1.getSize();
   }

   LocalRangeType getLocalRange() const
   {
      return op1.getLocalRange();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return op1.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2 );
   }

protected:
   const T1& op1;
   const T2& op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation< T1, typename T2::RealType >::
                              evaluate( std::declval<T1>(), std::declval<T2>()[0] ) );
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using CommunicatorType = typename T2::CommunicatorType;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   using LocalRangeType = typename T2::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< T1, typename T2::ConstLocalViewType, Operation >;

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      return getConstLocalView().getElement( i );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType getSize() const
   {
      return op2.getSize();
   }

   LocalRangeType getLocalRange() const
   {
      return op2.getLocalRange();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return op2.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( op1, op2.getConstLocalView() );
   }

protected:
   const T1& op1;
   const T2& op2;
};

////
// Distributed unary expression template
template< typename T1,
          template< typename > class Operation >
struct DistributedUnaryExpressionTemplate< T1, Operation, VectorExpressionVariable >
{
   using RealType = decltype( Operation< typename T1::RealType >::
                              evaluate( std::declval<T1>()[0] ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using CommunicatorType = typename T1::CommunicatorType;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = UnaryExpressionTemplate< typename T1::ConstLocalViewType, Operation >;

   DistributedUnaryExpressionTemplate( const T1& a )
   : operand( a ) {}

   RealType getElement( const IndexType i ) const
   {
      return getConstLocalView().getElement( i );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType getSize() const
   {
      return operand.getSize();
   }

   LocalRangeType getLocalRange() const
   {
      return operand.getLocalRange();
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return operand.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( operand.getConstLocalView() );
   }

protected:
   const T1& operand;
};

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator<<( std::ostream& str, const DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

template< typename T,
          template< typename > class Operation >
std::ostream& operator<<( std::ostream& str, const DistributedUnaryExpressionTemplate< T, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

////
// Operators are supposed to be in the same namespace as the expression templates

#ifndef DOXYGEN_ONLY

////
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator+( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator+( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
           const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator+( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator+( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator+( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator+( const DistributedUnaryExpressionTemplate< L1,LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Addition >( a, b );
}

////
// Binary expression subtraction
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator-( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator-( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
           const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator-( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator-( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator-( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator-( const DistributedUnaryExpressionTemplate< L1,LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator*( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator*( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
           const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator*( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator*( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator*( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator*( const DistributedUnaryExpressionTemplate< L1,LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Multiplication >( a, b );
}

////
// Binary expression division
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator/( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
operator/( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
           const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
operator/( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator/( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator/( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator/( const DistributedUnaryExpressionTemplate< L1,LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Division >( a, b );
}

////
// Comparison operator ==
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator==( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator==( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator==( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
            const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator==( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator==( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator==( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator==( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::EQ( a, b );
}

////
// Comparison operator !=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator!=( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator!=( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator!=( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
            const DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator!=( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
            const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator!=( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator!=( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator!=( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator!=( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::NE( a, b );
}

////
// Comparison operator <
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator<( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
           const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator<( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LT( a, b );
}

////
// Comparison operator <=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator<=( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<=( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<=( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
            const DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<=( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
            const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator<=( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator<=( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator<=( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator<=( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::LE( a, b );
}

////
// Comparison operator >
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator>( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
           const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
           const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
           const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator>( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator>( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GT( a, b );
}

////
// Comparison operator >=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator>=( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>=( const DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
            const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>=( const DistributedUnaryExpressionTemplate< T1, Operation >& a,
            const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
bool
operator>=( const typename DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
            const DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation >
bool
operator>=( const typename DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
            const DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
bool
operator>=( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
            const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
bool
operator>=( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
            const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedComparison< std::decay_t<decltype(a)>, std::decay_t<decltype(b)> >::GE( a, b );
}

////
// Unary operations

////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
operator-( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
operator-( const DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Minus >( a );
}

////
// Scalar product
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator,( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedExpressionSum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator,( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return DistributedExpressionSum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
operator,( const DistributedUnaryExpressionTemplate< L1, LOperation >& a,
           const DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return DistributedExpressionSum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
operator,( const DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return DistributedExpressionSum( a * b );
}

#endif // DOXYGEN_ONLY

} // namespace Expressions
} // namespace Containers

////
// All operations are supposed to be in namespace TNL

#ifndef DOXYGEN_ONLY

////
// Binary expression min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
min( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
min( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Min >( a, b );
}

////
// Binary expression max
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
auto
max( const typename Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& a,
     const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation >
auto
max( const typename Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >::RealType& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, std::decay_t<decltype(b)>, Containers::Expressions::Max >( a, b );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
abs( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
abs( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Abs >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
auto
pow( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, Real, Containers::Expressions::Pow >( a, exp );
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
auto
pow( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   return Containers::Expressions::DistributedBinaryExpressionTemplate< std::decay_t<decltype(a)>, Real, Containers::Expressions::Pow >( a, exp );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
exp( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
exp( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Exp >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sqrt( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sqrt( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cbrt( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cbrt( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cbrt >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log10( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log10( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
log2( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
log2( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Log2 >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cos( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cos( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
tan( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
tan( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tan >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
asin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
asin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
acos( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
acos( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
atan( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
atan( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sinh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sinh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
cosh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
cosh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
tanh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
tanh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Tanh >( a );
}

////
// Asinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
asinh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
asinh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Asinh >( a );
}

////
// Acosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
acosh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
acosh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Acosh >( a );
}

////
// Atanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
atanh( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
atanh( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Atanh >( a );
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
floor( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
floor( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
ceil( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
ceil( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Ceil >( a );
}

////
// Sign
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sign( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sign >( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sign( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, Containers::Expressions::Sign >( a );
}

////
// Cast
template< typename ResultType,
          typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          // workaround: templated type alias cannot be declared at block level
          template<typename> class CastOperation = Containers::Expressions::Cast< ResultType >::template Operation >
auto
cast( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, CastOperation >( a );
}

template< typename ResultType,
          typename L1,
          template< typename > class LOperation,
          // workaround: templated type alias cannot be declared at block level
          template<typename> class CastOperation = Containers::Expressions::Cast< ResultType >::template Operation >
auto
cast( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::DistributedUnaryExpressionTemplate< std::decay_t<decltype(a)>, CastOperation >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
min( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionMin( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
min( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
argMin( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionArgMin( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
argMin( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionArgMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
max( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionMax( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
max( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
argMax( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionArgMax( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
argMax( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionArgMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
sum( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionSum( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
sum( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionSum( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
maxNorm( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return max( abs( a ) );
}

template< typename L1,
          template< typename > class LOperation >
auto
maxNorm( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return max( abs( a ) );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
l1Norm( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionL1Norm( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
l1Norm( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionL1Norm( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
l2Norm( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return TNL::sqrt( DistributedExpressionL2Norm( a ) );
}

template< typename L1,
          template< typename > class LOperation >
auto
l2Norm( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return TNL::sqrt( DistributedExpressionL2Norm( a ) );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
auto
lpNorm( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& p )
// since (1.0 / p) has type double, TNL::pow returns double
-> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   return TNL::pow( DistributedExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
auto
lpNorm( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a, const Real& p )
// since (1.0 / p) has type double, TNL::pow returns double
-> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   return TNL::pow( DistributedExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
product( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionProduct( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
product( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionProduct( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
logicalOr( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionLogicalOr( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
logicalOr( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionLogicalOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
logicalAnd( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionLogicalAnd( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
logicalAnd( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionLogicalAnd( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
binaryOr( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionBinaryOr( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
binaryOr( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionBinaryOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
auto
binaryAnd( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return DistributedExpressionBinaryAnd( a );
}

template< typename L1,
          template< typename > class LOperation >
auto
binaryAnd( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a )
{
   return DistributedExpressionBinaryAnd( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
dot( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
dot( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
auto
dot( const Containers::Expressions::DistributedUnaryExpressionTemplate< L1, LOperation >& a,
     const Containers::Expressions::DistributedBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return (a, b);
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
auto
dot( const Containers::Expressions::DistributedBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::DistributedUnaryExpressionTemplate< R1, ROperation >& b )
{
   return (a, b);
}

#endif // DOXYGEN_ONLY


////
// Evaluation with reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Reduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, fetch, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Reduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, fetch, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, fetch, zero );
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Reduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), reduction, fetch, zero );
}

} // namespace TNL
