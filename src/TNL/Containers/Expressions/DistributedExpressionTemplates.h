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
#include <memory>

#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Containers/Expressions/DistributedComparison.h>
#include <TNL/Containers/Expressions/DistributedVerticalOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

////
// Distributed unary expression template
template< typename T1,
          typename Operation >
struct DistributedUnaryExpressionTemplate;

template< typename T1,
          typename Operation >
struct HasEnabledDistributedExpressionTemplates< DistributedUnaryExpressionTemplate< T1, Operation > >
: std::true_type
{};

////
// Distributed binary expression template
template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct DistributedBinaryExpressionTemplate
{};

template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type,
          ExpressionVariableType T2Type >
struct HasEnabledDistributedExpressionTemplates< DistributedBinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};

template< typename T1,
          typename T2,
          typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0], std::declval<T2>()[0] ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< typename T1::ConstLocalViewType,
                                                        typename T2::ConstLocalViewType,
                                                        Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not enabled for the left operand." );
   static_assert( HasEnabledDistributedExpressionTemplates< T2 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not enabled for the right operand." );
   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value,
                  "Attempt to mix operands which have different DeviceType." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b )
   {
      TNL_ASSERT_EQ( op1.getSize(), op2.getSize(),
                     "Attempt to mix operands with different sizes." );
      TNL_ASSERT_EQ( op1.getLocalRange(), op2.getLocalRange(),
                     "Distributed expressions are supported only on vectors which are distributed the same way." );
      TNL_ASSERT_EQ( op1.getGhosts(), op2.getGhosts(),
                     "Distributed expressions are supported only on vectors which are distributed the same way." );
      TNL_ASSERT_EQ( op1.getCommunicationGroup(), op2.getCommunicationGroup(),
                     "Distributed expressions are supported only on vectors within the same communication group." );
   }

   RealType getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
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

   IndexType getGhosts() const
   {
      return op1.getGhosts();
   }

   MPI_Comm getCommunicationGroup() const
   {
      return op1.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2.getConstLocalView() );
   }

   ConstLocalViewType getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1.getConstLocalViewWithGhosts(), op2.getConstLocalViewWithGhosts() );
   }

   std::shared_ptr< SynchronizerType > getSynchronizer() const
   {
      return op1.getSynchronizer();
   }

   int getValuesPerElement() const
   {
      return op1.getValuesPerElement();
   }

   void waitForSynchronization() const
   {
      op1.waitForSynchronization();
      op2.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2& op2;
};

template< typename T1,
          typename T2,
          typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0], std::declval<T2>() ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< typename T1::ConstLocalViewType, T2, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not enabled for the left operand." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
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

   IndexType getGhosts() const
   {
      return op1.getGhosts();
   }

   MPI_Comm getCommunicationGroup() const
   {
      return op1.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2 );
   }

   ConstLocalViewType getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1.getConstLocalViewWithGhosts(), op2 );
   }

   std::shared_ptr< SynchronizerType > getSynchronizer() const
   {
      return op1.getSynchronizer();
   }

   int getValuesPerElement() const
   {
      return op1.getValuesPerElement();
   }

   void waitForSynchronization() const
   {
      op1.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2& op2;
};

template< typename T1,
          typename T2,
          typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>(), std::declval<T2>()[0] ) );
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using LocalRangeType = typename T2::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< T1, typename T2::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T2::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T2 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not enabled for the right operand." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
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

   IndexType getGhosts() const
   {
      return op2.getGhosts();
   }

   MPI_Comm getCommunicationGroup() const
   {
      return op2.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( op1, op2.getConstLocalView() );
   }

   ConstLocalViewType getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1, op2.getConstLocalViewWithGhosts() );
   }

   std::shared_ptr< SynchronizerType > getSynchronizer() const
   {
      return op2.getSynchronizer();
   }

   int getValuesPerElement() const
   {
      return op2.getValuesPerElement();
   }

   void waitForSynchronization() const
   {
      op2.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2& op2;
};

////
// Distributed unary expression template
template< typename T1,
          typename Operation >
struct DistributedUnaryExpressionTemplate
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0] ) );
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = UnaryExpressionTemplate< typename T1::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed unary expression templates - distributed expression templates are not enabled for the operand." );

   DistributedUnaryExpressionTemplate( const T1& a )
   : operand( a ) {}

   RealType getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
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

   IndexType getGhosts() const
   {
      return operand.getGhosts();
   }

   MPI_Comm getCommunicationGroup() const
   {
      return operand.getCommunicationGroup();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return ConstLocalViewType( operand.getConstLocalView() );
   }

   ConstLocalViewType getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( operand.getConstLocalViewWithGhosts() );
   }

   std::shared_ptr< SynchronizerType > getSynchronizer() const
   {
      return operand.getSynchronizer();
   }

   int getValuesPerElement() const
   {
      return operand.getValuesPerElement();
   }

   void waitForSynchronization() const
   {
      operand.waitForSynchronization();
   }

protected:
   const T1& operand;
};

#ifndef DOXYGEN_ONLY

////
// Binary expressions addition
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
operator+( const ET1& a, const ET2& b )
{
   return DistributedBinaryExpressionTemplate< ET1, ET2, Addition >( a, b );
}

////
// Binary expression subtraction
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
operator-( const ET1& a, const ET2& b )
{
   return DistributedBinaryExpressionTemplate< ET1, ET2, Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
operator*( const ET1& a, const ET2& b )
{
   return DistributedBinaryExpressionTemplate< ET1, ET2, Multiplication >( a, b );
}

////
// Binary expression division
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
operator/( const ET1& a, const ET2& b )
{
   return DistributedBinaryExpressionTemplate< ET1, ET2, Division >( a, b );
}

////
// Comparison operator ==
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
bool
operator==( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::EQ( a, b );
}

////
// Comparison operator !=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
bool
operator!=( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::NE( a, b );
}

////
// Comparison operator <
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
bool
operator<( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::LT( a, b );
}

////
// Comparison operator <=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
bool
operator<=( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::LE( a, b );
}

////
// Comparison operator >
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
bool
operator>( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::GT( a, b );
}

////
// Comparison operator >=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
bool
operator>=( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::GE( a, b );
}

////
// Scalar product
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
operator,( const ET1& a, const ET2& b )
{
   return DistributedExpressionSum( a * b );
}

template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
dot( const ET1& a, const ET2& b )
{
   return (a, b);
}

////
// Unary expression minus
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
operator-( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Minus >( a );
}

////
// Binary expression min
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
min( const ET1& a, const ET2& b )
{
   return DistributedBinaryExpressionTemplate< ET1, ET2, Min >( a, b );
}

////
// Binary expression max
template< typename ET1, typename ET2,
          typename..., typename = EnableIfDistributedBinaryExpression_t< ET1, ET2 >, typename = void, typename = void >
auto
max( const ET1& a, const ET2& b )
{
   return DistributedBinaryExpressionTemplate< ET1, ET2, Max >( a, b );
}

////
// Abs
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
abs( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Abs >( a );
}

////
// Pow
template< typename ET1, typename Real,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
pow( const ET1& a, const Real& exp )
{
   return DistributedBinaryExpressionTemplate< ET1, Real, Pow >( a, exp );
}

////
// Exp
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
exp( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Exp >( a );
}

////
// Sqrt
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
sqrt( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Sqrt >( a );
}

////
// Cbrt
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
cbrt( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Cbrt >( a );
}

////
// Log
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
log( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Log >( a );
}

////
// Log10
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
log10( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Log10 >( a );
}

////
// Log2
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
log2( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Log2 >( a );
}

////
// Sin
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
sin( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Sin >( a );
}

////
// Cos
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
cos( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Cos >( a );
}

////
// Tan
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
tan( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Tan >( a );
}

////
// Asin
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
asin( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Asin >( a );
}

////
// Acos
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
acos( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Acos >( a );
}

////
// Atan
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
atan( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Atan >( a );
}

////
// Sinh
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
sinh( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Sinh >( a );
}

////
// Cosh
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
cosh( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Cosh >( a );
}

////
// Tanh
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
tanh( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Tanh >( a );
}

////
// Asinh
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
asinh( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Asinh >( a );
}

////
// Acosh
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
acosh( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Acosh >( a );
}

////
// Atanh
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
atanh( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Atanh >( a );
}

////
// Floor
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
floor( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Floor >( a );
}

////
// Ceil
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
ceil( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Ceil >( a );
}

////
// Sign
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
sign( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, Sign >( a );
}

////
// Cast
template< typename ResultType,
          typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >,
          // workaround: templated type alias cannot be declared at block level
          typename CastOperation = typename Cast< ResultType >::Operation,
          typename = void, typename = void >
auto
cast( const ET1& a )
{
   return DistributedUnaryExpressionTemplate< ET1, CastOperation >( a );
}

////
// Vertical operations
template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
min( const ET1& a )
{
   return DistributedExpressionMin( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
argMin( const ET1& a )
{
   return DistributedExpressionArgMin( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
max( const ET1& a )
{
   return DistributedExpressionMax( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
argMax( const ET1& a )
{
   return DistributedExpressionArgMax( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
sum( const ET1& a )
{
   return DistributedExpressionSum( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
maxNorm( const ET1& a )
{
   return max( abs( a ) );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
l1Norm( const ET1& a )
{
   return sum( abs( a ) );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
l2Norm( const ET1& a )
{
   using TNL::sqrt;
   return sqrt( sum( a * a ) );
}

template< typename ET1,
          typename Real,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
lpNorm( const ET1& a, const Real& p )
// since (1.0 / p) has type double, TNL::pow returns double
-> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   using TNL::pow;
   return pow( sum( pow( abs( a ), p ) ), 1.0 / p );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
product( const ET1& a )
{
   return DistributedExpressionProduct( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
logicalOr( const ET1& a )
{
   return DistributedExpressionLogicalOr( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
logicalAnd( const ET1& a )
{
   return DistributedExpressionLogicalAnd( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
binaryOr( const ET1& a )
{
   return DistributedExpressionBinaryOr( a );
}

template< typename ET1,
          typename..., typename = EnableIfDistributedUnaryExpression_t< ET1 >, typename = void, typename = void >
auto
binaryAnd( const ET1& a )
{
   return DistributedExpressionBinaryAnd( a );
}

////
// Output stream
template< typename T1,
          typename T2,
          typename Operation >
std::ostream& operator<<( std::ostream& str, const DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   const auto localRange = expression.getLocalRange();
   str << "[ ";
   for( int i = localRange.getBegin(); i < localRange.getEnd() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( localRange.getEnd() - 1 );
   if( expression.getGhosts() > 0 ) {
      str << " | ";
      const auto localView = expression.getConstLocalViewWithGhosts();
      for( int i = localRange.getSize(); i < localView.getSize() - 1; i++ )
         str << localView.getElement( i ) << ", ";
      str << localView.getElement( localView.getSize() - 1 );
   }
   str << " ]";
   return str;
}

template< typename T,
          typename Operation >
std::ostream& operator<<( std::ostream& str, const DistributedUnaryExpressionTemplate< T, Operation >& expression )
{
   const auto localRange = expression.getLocalRange();
   str << "[ ";
   for( int i = localRange.getBegin(); i < localRange.getEnd() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( localRange.getEnd() - 1 );
   if( expression.getGhosts() > 0 ) {
      str << " | ";
      const auto localView = expression.getConstLocalViewWithGhosts();
      for( int i = localRange.getSize(); i < localView.getSize() - 1; i++ )
         str << localView.getElement( i ) << ", ";
      str << localView.getElement( localView.getSize() - 1 );
   }
   str << " ]";
   return str;
}

#endif // DOXYGEN_ONLY

} // namespace Expressions

// Make all operators visible in the TNL::Containers namespace to be considered
// even for DistributedVector and DistributedVectorView
using Expressions::operator+;
using Expressions::operator-;
using Expressions::operator*;
using Expressions::operator/;
using Expressions::operator,;
using Expressions::operator==;
using Expressions::operator!=;
using Expressions::operator<;
using Expressions::operator<=;
using Expressions::operator>;
using Expressions::operator>=;

// Make all functions visible in the TNL::Containers namespace
using Expressions::dot;
using Expressions::min;
using Expressions::max;
using Expressions::abs;
using Expressions::pow;
using Expressions::exp;
using Expressions::sqrt;
using Expressions::cbrt;
using Expressions::log;
using Expressions::log10;
using Expressions::log2;
using Expressions::sin;
using Expressions::cos;
using Expressions::tan;
using Expressions::asin;
using Expressions::acos;
using Expressions::atan;
using Expressions::sinh;
using Expressions::cosh;
using Expressions::tanh;
using Expressions::asinh;
using Expressions::acosh;
using Expressions::atanh;
using Expressions::floor;
using Expressions::ceil;
using Expressions::sign;
using Expressions::cast;
using Expressions::argMin;
using Expressions::argMax;
using Expressions::sum;
using Expressions::maxNorm;
using Expressions::l1Norm;
using Expressions::l2Norm;
using Expressions::lpNorm;
using Expressions::product;
using Expressions::logicalAnd;
using Expressions::logicalOr;
using Expressions::binaryAnd;
using Expressions::binaryOr;

} // namespace Containers

// Make all functions visible in the main TNL namespace
using Containers::dot;
using Containers::min;
using Containers::max;
using Containers::abs;
using Containers::pow;
using Containers::exp;
using Containers::sqrt;
using Containers::cbrt;
using Containers::log;
using Containers::log10;
using Containers::log2;
using Containers::sin;
using Containers::cos;
using Containers::tan;
using Containers::asin;
using Containers::acos;
using Containers::atan;
using Containers::sinh;
using Containers::cosh;
using Containers::tanh;
using Containers::asinh;
using Containers::acosh;
using Containers::atanh;
using Containers::floor;
using Containers::ceil;
using Containers::sign;
using Containers::cast;
using Containers::argMin;
using Containers::argMax;
using Containers::sum;
using Containers::maxNorm;
using Containers::l1Norm;
using Containers::l2Norm;
using Containers::lpNorm;
using Containers::product;
using Containers::logicalAnd;
using Containers::logicalOr;
using Containers::binaryAnd;
using Containers::binaryOr;

////
// Evaluation with reduction
template< typename Vector,
   typename T1,
   typename T2,
   typename Operation,
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
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector,
   typename T1,
   typename Operation,
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
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), fetch, reduction, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   typename Operation,
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
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector,
   typename T1,
   typename Operation,
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
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), fetch, reduction, zero );
}

////
// Addition and reduction
template< typename Vector,
   typename T1,
   typename T2,
   typename Operation,
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
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector,
   typename T1,
   typename Operation,
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
   return Algorithms::Reduction< DeviceType >::reduce( lhs.getSize(), fetch, reduction, zero );
}

} // namespace TNL
