/***************************************************************************
                          StaticExpressionTemplates.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Expressions/StaticComparison.h>
#include <TNL/Containers/Expressions/IsStatic.h>
#include <TNL/Containers/Expressions/VerticalOperations.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {


template< typename T1,
          template< typename > class Operation,
          typename Parameter = void,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value >
struct StaticUnaryExpressionTemplate
{
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct StaticBinaryExpressionTemplate
{
};


template< typename T1,
          template< typename > class Operation,
          typename Parameter >
struct IsStaticType< StaticUnaryExpressionTemplate< T1, Operation, Parameter > >
{
   static constexpr bool value = StaticUnaryExpressionTemplate< T1, Operation, Parameter >::isStatic();
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct IsStaticType< StaticBinaryExpressionTemplate< T1, T2, Operation > >
{
   static constexpr bool value = StaticBinaryExpressionTemplate< T1, T2, Operation >::isStatic();
};


////
// Static binary expression template
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorVariable, VectorVariable >
{
   static_assert( IsStaticType< T1 >::value, "Left-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( IsStaticType< T2 >::value, "Right-hand side operand of static expression is not static, i.e. based on static vector." );
   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;
   static_assert( IsStaticType< T1 >::value == IsStaticType< T2 >::value, "Attempt to mix static and non-static operands in binary expression templates" );
   static_assert( T1::getSize() == T2::getSize(), "Attempt to mix static operands with different sizes." );

   static constexpr bool isStatic() { return true; }

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   static StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1 &op1;
      const T2 &op2;
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorVariable, ArithmeticVariable  >
{
   static_assert( IsStaticType< T1 >::value, "Left-hand side operand of static expression is not static, i.e. based on static vector." );

   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   static constexpr bool isStatic() { return true; }

   static constexpr int getSize() { return T1::getSize(); };


   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1 &op1;
      const T2 &op2;

};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorVariable  >
{
   static_assert( IsStaticType< T2 >::value, "Right-hand side operand of static expression is not static, i.e. based on static vector." );

   using RealType = typename T2::RealType;
   using IsExpressionTemplate = bool;

   static constexpr bool isStatic() { return true; }

   static constexpr int getSize() { return T2::getSize(); };


   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1& op1;
      const T2& op2;
};

////
// Static unary expression template
//
// Parameter type serves mainly for pow( base, exp ). Here exp is parameter we need
// to pass to pow.
template< typename T1,
          template< typename > class Operation,
          typename Parameter >
struct StaticUnaryExpressionTemplate< T1, Operation, Parameter, VectorVariable >
{
   static_assert( IsStaticType< T1 >::value, "Operand of static expression is not static, i.e. based on static vector." );

   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   static constexpr bool isStatic() { return true; }

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticUnaryExpressionTemplate( const T1& a, const Parameter& p )
   : operand( a ), parameter( p ) {}

   __cuda_callable__
   static StaticUnaryExpressionTemplate evaluate( const T1& a )
   {
      return StaticUnaryExpressionTemplate( a );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ], parameter );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ], parameter );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   void set( const Parameter& p ) { parameter = p; }

   const Parameter& get() { return parameter; }

   protected:
      const T1& operand;
      Parameter parameter;
};

////
// Static unary expression template with no parameter
template< typename T1,
          template< typename > class Operation >
struct StaticUnaryExpressionTemplate< T1, Operation, void, VectorVariable >
{
   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   static constexpr bool isStatic() { return true; }

   static constexpr int getSize() { return T1::getSize(); };

   __cuda_callable__
   StaticUnaryExpressionTemplate( const T1& a ): operand( a ){}

   __cuda_callable__
   static StaticUnaryExpressionTemplate evaluate( const T1& a )
   {
      return StaticUnaryExpressionTemplate( a );
   }

   RealType getElement( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
      TNL_ASSERT_LT( i, this->getSize(), "Asking for element with index larger than expression size." );
      return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   RealType x() const
   {
      return (*this)[ 0 ];
   }

   __cuda_callable__
   RealType y() const
   {
      return (*this)[ 1 ];
   }

   __cuda_callable__
   RealType z() const
   {
      return (*this)[ 2 ];
   }

   protected:
      const T1& operand;
};

////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}

template< typename T,
          template< typename > class Operation,
          typename Parameter >
std::ostream& operator << ( std::ostream& str, const Containers::Expressions::StaticUnaryExpressionTemplate< T, Operation, Parameter >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}
      } //namespace Expressions
   } //namespace Containers

////
// All operations are supposed to be in namespace TNL

////
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   Containers::Expressions::Addition >
operator + ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Addition >( a, b );
}

////
// Binary expression subtraction
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   Containers::Expressions::Subtraction >
operator - ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Subtraction >( a, b );
}

////
// Binary expression multiplication
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   Containers::Expressions::Multiplication >
operator * ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Multiplication >( a, b );
}

////
// Binary expression division
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Division >
operator + ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   Containers::Expressions::Division >
operator / ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Division >( a, b );
}


////
// Binary expression min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
      const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::Min >
min( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   Containers::Expressions::Min >
min( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Min >
min( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Min >
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Min >( a, b );
}

////
// Binary expression max
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   Containers::Expressions::Max >
operator + ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
   Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
   Containers::Expressions::Max >
max( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType,
      Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1,LOperation >& a,
     const Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >,
      Containers::Expressions::Max >( a, b );
}

////
// Comparison operator ==
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator == ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::StaticComparison< Left, Right >::EQ( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator == ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::EQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::EQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::EQ( a, b );
}

////
// Comparison operator !=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator != ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::StaticComparison< Left, Right >::NE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator != ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::NE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::NE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::NE( a, b );
}

////
// Comparison operator <
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator < ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::StaticComparison< Left, Right >::LT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator < ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::LT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::LT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::LT( a, b );
}

////
// Comparison operator <=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator <= ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::StaticComparison< Left, Right >::LE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator <= ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::LE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::LE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::LE( a, b );
}

////
// Comparison operator >
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator > ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::StaticComparison< Left, Right >::GT( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator > ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
             const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::GT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::GT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::GT( a, b );
}

////
// Comparison operator >=
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   using Right = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   using Right = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   return Containers::Expressions::StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator >= ( const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& a,
              const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& b )
{
   using Left = typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >;
   return Containers::Expressions::StaticComparison< Left, Right >::GE( a, b );
}

template< typename T1,
          template< typename > class Operation,
          typename Parameter >
__cuda_callable__
bool
operator >= ( const typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType& a,
              const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& b )
{
   using Left = typename Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >::RealType;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::GE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   using Left = Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >;
   using Right = Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >;
   return Containers::Expressions::StaticComparison< Left, Right >::GE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   using Left = Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using Right = Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation, RParameter >;
   return Containers::Expressions::StaticComparison< Left, Right >::GE( a, b );
}

////
// Unary operations

////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Minus >
operator -( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Minus >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Abs >
operator -( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Minus >( a );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   auto e = Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a, const Real& exp )
{
   auto e = Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sin >
floor( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Floor >
floor( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Ceil >
ceil( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Ceil >
sin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Acos >
cos( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Acos >
acos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Cos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Atan >
tan( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Atan >
atan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Tanh >
cosh( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Tanh >
tanh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log >
log( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Log >
log( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >,
      Containers::Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
min( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionMin( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionMin( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionMax( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionMax( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
sum( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionSum( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
sum( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionSum( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
__cuda_callable__
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
lpNorm( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& p )
{
   if( p == 1.0 )
      return StaticExpressionLpNorm( a, p );
   if( p == 2.0 )
      return TNL::sqrt( StaticExpressionLpNorm( a, p ) );
   return TNL::pow( StaticExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename Real >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
lpNorm( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a, const Real& p )
{
   if( p == 1.0 )
      return StaticExpressionLpNorm( a, p );
   if( p == 2.0 )
      return TNL::sqrt( StaticExpressionLpNorm( a, p ) );
   return TNL::pow( StaticExpressionLpNorm( a, p ), 1.0 / p );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
product( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionProduct( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
product( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionProduct( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
logicalOr( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionLogicalOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
logicalOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionLogicalOr( a );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
binaryOr( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return StaticExpressionBinaryOr( a );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
binaryOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a )
{
   return StaticExpressionBinaryOr( a );
}

////
// Scalar product
// TODO: Declaration with decltype does not work with g++ 8.3.0 though I think that it should
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
//auto
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
operator,( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
//auto
typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType
operator,( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
//auto
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
operator,( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
           const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
//auto
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
operator,( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
//auto
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
dot( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
//auto
typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType
dot( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename LParameter,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
//auto
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >::RealType
dot( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, LParameter >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation,
          typename RParameter >
__cuda_callable__
//auto
typename Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
dot( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
//-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

////
// Evaluation with reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Parameter,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result evaluateAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

////
// Addition with reduction
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, aux );
   }
   return result;
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Parameter,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduce( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, aux );
   }
   return result;
}

////
// Addition with reduction of abs
template< typename Vector,
   typename T1,
   typename T2,
   template< typename, typename > class Operation,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, TNL::abs( aux ) );
   }
   return result;
}

template< typename Vector,
   typename T1,
   template< typename > class Operation,
   typename Parameter,
   typename Reduction,
   typename VolatileReduction,
   typename Result >
__cuda_callable__
Result addAndReduceAbs( Vector& lhs,
   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation, Parameter >& expression,
   Reduction& reduction,
   VolatileReduction& volatileReduction,
   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      reduction( result, TNL::abs( aux ) );
   }
   return result;
}

} // namespace TNL
