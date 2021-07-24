/***************************************************************************
                          ExpressionTemplates.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <utility>

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Expressions/TypeTraits.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/Comparison.h>
#include <TNL/Containers/Expressions/HorizontalOperations.h>
#include <TNL/Algorithms/reduce.h>

namespace TNL {
namespace Containers {
namespace Expressions {

template< typename T1,
          typename Operation >
struct UnaryExpressionTemplate;

template< typename T1,
          typename Operation >
struct HasEnabledExpressionTemplates< UnaryExpressionTemplate< T1, Operation > >
: std::true_type
{};

template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct BinaryExpressionTemplate;

template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type,
          ExpressionVariableType T2Type >
struct HasEnabledExpressionTemplates< BinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};


////
// Non-static binary expression template
template< typename T1,
          typename T2,
          typename Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0], std::declval<T2>()[0] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using ConstViewType = BinaryExpressionTemplate;

   static_assert( HasEnabledExpressionTemplates< T1 >::value,
                  "Invalid operand in binary expression templates - expression templates are not enabled for the left operand." );
   static_assert( HasEnabledExpressionTemplates< T2 >::value,
                  "Invalid operand in binary expression templates - expression templates are not enabled for the right operand." );
   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value,
                  "Attempt to mix operands which have different DeviceType." );

   BinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a.getConstView() ), op2( b.getConstView() )
   {
      TNL_ASSERT_EQ( op1.getSize(), op2.getSize(),
                     "Attempt to mix operands with different sizes." );
   }

   RealType getElement( const IndexType i ) const
   {
      return Operation::evaluate( op1.getElement( i ), op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   RealType operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return op1.getSize();
   }

   ConstViewType getConstView() const
   {
      return *this;
   }

protected:
   const typename T1::ConstViewType op1;
   const typename T2::ConstViewType op2;
};

template< typename T1,
          typename T2,
          typename Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0], std::declval<T2>() ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using ConstViewType = BinaryExpressionTemplate;

   static_assert( HasEnabledExpressionTemplates< T1 >::value,
                  "Invalid operand in binary expression templates - expression templates are not enabled for the left operand." );

   BinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a.getConstView() ), op2( b ) {}

   RealType getElement( const IndexType i ) const
   {
      return Operation::evaluate( op1.getElement( i ), op2 );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   RealType operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return op1.getSize();
   }

   ConstViewType getConstView() const
   {
      return *this;
   }

protected:
   const typename T1::ConstViewType op1;
   const T2 op2;
};

template< typename T1,
          typename T2,
          typename Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>(), std::declval<T2>()[0] ) );
   using ValueType = RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using ConstViewType = BinaryExpressionTemplate;

   static_assert( HasEnabledExpressionTemplates< T2 >::value,
                  "Invalid operand in binary expression templates - expression templates are not enabled for the right operand." );

   BinaryExpressionTemplate( const T1& a, const T2& b )
   : op1( a ), op2( b.getConstView() ) {}

   RealType getElement( const IndexType i ) const
   {
      return Operation::evaluate( op1, op2.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   RealType operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return op2.getSize();
   }

   ConstViewType getConstView() const
   {
      return *this;
   }

protected:
   const T1 op1;
   const typename T2::ConstViewType op2;
};

////
// Non-static unary expression template
template< typename T1,
          typename Operation >
struct UnaryExpressionTemplate
{
   using RealType = decltype( Operation::evaluate( std::declval<T1>()[0] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using ConstViewType = UnaryExpressionTemplate;

   static_assert( HasEnabledExpressionTemplates< T1 >::value,
                  "Invalid operand in unary expression templates - expression templates are not enabled for the operand." );

   UnaryExpressionTemplate( const T1& a )
   : operand( a.getConstView() ) {}

   RealType getElement( const IndexType i ) const
   {
      return Operation::evaluate( operand.getElement( i ) );
   }

   __cuda_callable__
   RealType operator[]( const IndexType i ) const
   {
      return Operation::evaluate( operand[ i ] );
   }

   __cuda_callable__
   RealType operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType getSize() const
   {
      return operand.getSize();
   }

   ConstViewType getConstView() const
   {
      return *this;
   }

protected:
   const typename T1::ConstViewType operand;
};

#ifndef DOXYGEN_ONLY

////
// Binary expressions addition
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
operator+( const ET1& a, const ET2& b )
{
   return BinaryExpressionTemplate< ET1, ET2, Addition >( a, b );
}

////
// Binary expression subtraction
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
operator-( const ET1& a, const ET2& b )
{
   return BinaryExpressionTemplate< ET1, ET2, Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
operator*( const ET1& a, const ET2& b )
{
   return BinaryExpressionTemplate< ET1, ET2, Multiplication >( a, b );
}

////
// Binary expression division
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
operator/( const ET1& a, const ET2& b )
{
   return BinaryExpressionTemplate< ET1, ET2, Division >( a, b );
}

////
// Comparison operator ==
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
bool
operator==( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::EQ( a, b );
}

////
// Comparison operator !=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
bool
operator!=( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::NE( a, b );
}

////
// Comparison operator <
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
bool
operator<( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::LT( a, b );
}

////
// Comparison operator <=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
bool
operator<=( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::LE( a, b );
}

////
// Comparison operator >
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
bool
operator>( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::GT( a, b );
}

////
// Comparison operator >=
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
bool
operator>=( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::GE( a, b );
}

////
// Scalar product
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
operator,( const ET1& a, const ET2& b )
{
   return Algorithms::reduce( a * b, TNL::Plus{} );
}

template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
dot( const ET1& a, const ET2& b )
{
   return (a, b);
}

////
// Unary expression minus
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
operator-( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Minus >( a );
}

////
// Binary expression min
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
min( const ET1& a, const ET2& b )
{
   return BinaryExpressionTemplate< ET1, ET2, Min >( a, b );
}

////
// Binary expression max
template< typename ET1, typename ET2,
          typename..., typename = EnableIfBinaryExpression_t< ET1, ET2 >, typename = void >
auto
max( const ET1& a, const ET2& b )
{
   return BinaryExpressionTemplate< ET1, ET2, Max >( a, b );
}

////
// Abs
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
abs( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Abs >( a );
}

////
// Pow
template< typename ET1, typename Real,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
pow( const ET1& a, const Real& exp )
{
   return BinaryExpressionTemplate< ET1, Real, Pow >( a, exp );
}

////
// Exp
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
exp( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Exp >( a );
}

////
// Sqrt
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
sqrt( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Sqrt >( a );
}

////
// Cbrt
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
cbrt( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Cbrt >( a );
}

////
// Log
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
log( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Log >( a );
}

////
// Log10
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
log10( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Log10 >( a );
}

////
// Log2
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
log2( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Log2 >( a );
}

////
// Sin
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
sin( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Sin >( a );
}

////
// Cos
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
cos( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Cos >( a );
}

////
// Tan
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
tan( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Tan >( a );
}

////
// Asin
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
asin( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Asin >( a );
}

////
// Acos
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
acos( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Acos >( a );
}

////
// Atan
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
atan( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Atan >( a );
}

////
// Sinh
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
sinh( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Sinh >( a );
}

////
// Cosh
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
cosh( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Cosh >( a );
}

////
// Tanh
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
tanh( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Tanh >( a );
}

////
// Asinh
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
asinh( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Asinh >( a );
}

////
// Acosh
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
acosh( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Acosh >( a );
}

////
// Atanh
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
atanh( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Atanh >( a );
}

////
// Floor
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
floor( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Floor >( a );
}

////
// Ceil
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
ceil( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Ceil >( a );
}

////
// Sign
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
sign( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, Sign >( a );
}

////
// Cast
template< typename ResultType,
          typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >,
          // workaround: templated type alias cannot be declared at block level
          typename CastOperation = typename Cast< ResultType >::Operation,
          typename = void >
auto
cast( const ET1& a )
{
   return UnaryExpressionTemplate< ET1, CastOperation >( a );
}

////
// Vertical operations
template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
min( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Min{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
argMin( const ET1& a )
{
   return Algorithms::reduceWithArgument( a, TNL::MinWithArg{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
max( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Max{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
argMax( const ET1& a )
{
   return Algorithms::reduceWithArgument( a, TNL::MaxWithArg{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
sum( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Plus{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
maxNorm( const ET1& a )
{
   return max( abs( a ) );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
l1Norm( const ET1& a )
{
   return sum( abs( a ) );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
l2Norm( const ET1& a )
{
   using TNL::sqrt;
   return sqrt( sum( a * a ) );
}

template< typename ET1,
          typename Real,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
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
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
product( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Multiplies{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
logicalAnd( const ET1& a )
{
   return Algorithms::reduce( a, TNL::LogicalAnd{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
logicalOr( const ET1& a )
{
   return Algorithms::reduce( a, TNL::LogicalOr{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
binaryAnd( const ET1& a )
{
   return Algorithms::reduce( a, TNL::BitAnd{} );
}

template< typename ET1,
          typename..., typename = EnableIfUnaryExpression_t< ET1 >, typename = void >
auto
binaryOr( const ET1& a )
{
   return Algorithms::reduce( a, TNL::BitOr{} );
}

#endif // DOXYGEN_ONLY

////
// Output stream
template< typename T1,
          typename T2,
          typename Operation >
std::ostream& operator<<( std::ostream& str, const BinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

template< typename T,
          typename Operation >
std::ostream& operator<<( std::ostream& str, const UnaryExpressionTemplate< T, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

} // namespace Expressions

// Make all operators visible in the TNL::Containers namespace to be considered
// even for Vector and VectorView
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
   const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector,
   typename T1,
   typename Operation,
   typename Reduction,
   typename Result >
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
   const Reduction& reduction,
   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> RealType { return ( lhs_data[ i ] = expression[ i ] ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, lhs.getSize(), fetch, reduction, zero );
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
   const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector,
   typename T1,
   typename Operation,
   typename Reduction,
   typename Result >
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, lhs.getSize(), fetch, reduction, zero );
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
   const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector,
   typename T1,
   typename Operation,
   typename Reduction,
   typename Result >
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, lhs.getSize(), fetch, reduction, zero );
}

} // namespace TNL
