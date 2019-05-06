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
#include <TNL/Containers/Expressions/Comparison.h>
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
   static constexpr bool isStatic() { return true; }

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   static StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return op1.getSize();
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

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   __cuda_callable__
   int getSize() const
   {
       return op1.getSize();
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

   __cuda_callable__
   StaticBinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   __cuda_callable__
   StaticBinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return StaticBinaryExpressionTemplate( a, b );
   }

   RealType getElement( const int i ) const
   {
       return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
       return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return op2.getSize();
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
       return Operation< typename T1::RealType >::evaluate( operand[ i ], parameter );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand[ i ], parameter );
   }

   __cuda_callable__
   int getSize() const
   {
       return operand.getSize();
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

   __cuda_callable__
   StaticUnaryExpressionTemplate( const T1& a ): operand( a ){}

   __cuda_callable__
   static StaticUnaryExpressionTemplate evaluate( const T1& a )
   {
      return StaticUnaryExpressionTemplate( a );
   }

   RealType getElement( const int i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType >::evaluate( operand[ i ] );
   }

   __cuda_callable__
   int getSize() const
   {
       return operand.getSize();
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
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
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

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Addition >
operator + ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
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
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
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

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Subtraction >
operator - ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
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
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
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

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Multiplication >
operator * ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
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
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
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

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Division >
operator / ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
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
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
      const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
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

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Min >
min( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
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
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
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

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Containers::Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Containers::Expressions::StaticBinaryExpressionTemplate<
   Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Containers::Expressions::Max >
max( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticBinaryExpressionTemplate<
      Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
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
   return Containers::Expressions::StaticComparisonEQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticComparisonEQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonEQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator == ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Containers::Expressions::StaticComparisonEQ( a, b );
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
   return Containers::Expressions::StaticComparisonNE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticComparisonNE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonNE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator != ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonNE( a, b );
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
   return Containers::Expressions::StaticComparisonLT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticComparisonLT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonLT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator < ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonLT( a, b );
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
   return Containers::Expressions::StaticComparisonLE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticComparisonLE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonLE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonLE( a, b );
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
   return Containers::Expressions::StaticComparisonGT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticComparisonGT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonGT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator > ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonGT( a, b );
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
   return Containers::Expressions::StaticComparisonGE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Containers::Expressions::StaticComparisonGE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonGE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Containers::Expressions::StaticComparisonGE( a, b );
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Abs >
operator -( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Abs >
abs( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sin >
sin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cos >
cos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Tan >
tan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sqrt >
sqrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cbrt >
cbrt( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          typename Real >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Pow >
pow( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   auto e = Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Floor >
floor( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Ceil >
sin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Asin >
asin( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Acos >
acos( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Atan >
atan( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Sinh >
sinh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Cosh >
cosh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Tanh >
tanh( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log >
log( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log10 >
log10( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Log2 >
log2( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          template< typename > class LOperation >
__cuda_callable__
const Containers::Expressions::StaticUnaryExpressionTemplate<
   Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Containers::Expressions::Exp >
exp( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Containers::Expressions::StaticUnaryExpressionTemplate<
      Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
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
          typename Parameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
min( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
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
          typename Parameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
max( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
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
          typename Parameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
sum( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
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
   return StaticExpressionLpNorm( a, p );
}

template< typename L1,
          template< typename > class LOperation,
          typename Parameter,
          typename Real >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
lpNorm( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >& a, const Real& p )
{
   return StaticExpressionLpNorm( a, p );
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
          typename Parameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
product( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
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
          typename Parameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
logicalOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
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
          typename Parameter >
__cuda_callable__
typename Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >::RealType
binaryOr( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation, Parameter >& a )
{
   return StaticExpressionBinaryOr( a );
}

////
// Scalar product
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
auto
operator,( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
auto
operator,( const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
           const typename Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
auto
operator,( const Containers::Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
           const typename Containers::Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
auto
operator,( const Containers::Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
           const typename Containers::Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
-> decltype( TNL::sum( a * b ) )
{
   return TNL::sum( a * b );
}

} // namespace TNL
