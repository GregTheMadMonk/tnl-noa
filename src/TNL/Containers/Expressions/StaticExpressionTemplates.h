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
// Binary expressions addition
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Addition >
operator + ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Addition >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Addition >
operator + ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Addition >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Addition >
operator + ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Addition >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Expressions::Addition >
operator + ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
      Expressions::Addition >( a, b );
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
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Subtraction >
operator - ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Subtraction >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Subtraction >
operator - ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Subtraction >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Subtraction >
operator - ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Subtraction >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Expressions::Subtraction >
operator - ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
      Expressions::Subtraction >( a, b );
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
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Multiplication >
operator * ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Multiplication >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Multiplication >
operator * ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Multiplication >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Multiplication >
operator * ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Multiplication >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Expressions::Multiplication >
operator * ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
      Expressions::Multiplication >( a, b );
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
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Division >
operator / ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Division >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Division >
operator / ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Division >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Division >
operator / ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Division >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Expressions::Division >
operator / ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
      Expressions::Division >( a, b );
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
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Min >
min ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
      const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Min >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Min >
min( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Min >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Min >
min( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Min >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Expressions::Min >
min( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
      Expressions::Min >( a, b );
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
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Max >
max( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Max >( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
   typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
   Expressions::Max >
max( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
     const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >,
      typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType,
      Expressions::Max >( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
   Expressions::Max >
max( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
     const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >,
      Expressions::Max >( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
const Expressions::StaticBinaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
   Expressions::Max >
max( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
     const typename Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::StaticBinaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >,
      Expressions::Max >( a, b );
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
operator == ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator == ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator == ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator == ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::StaticUnaryExpressionTemplate< R1,ROperation >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
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
operator != ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator != ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator != ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator != ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::StaticComparisonNE( a, b );
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
operator < ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator < ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator < ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator < ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::StaticComparisonLT( a, b );
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
operator <= ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator <= ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator <= ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::StaticComparisonLE( a, b );
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
operator > ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator > ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator > ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
             const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator > ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
             const typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::StaticComparisonGT( a, b );
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
operator >= ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
__cuda_callable__
bool
operator >= ( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >::RealType& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< typename L1,
          template< typename > class LOperation,
          typename R1,
          typename R2,
          template< typename, typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a,
              const typename Expressions::StaticBinaryExpressionTemplate< R1, R2, ROperation >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename R1,
          template< typename > class ROperation >
__cuda_callable__
bool
operator >= ( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a,
              const typename Expressions::StaticUnaryExpressionTemplate< R1, ROperation >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

////
// Unary operations


////
// Minus
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Minus >
operator -( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Minus >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Abs >
operator -( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Minus >( a );
}

////
// Abs
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Abs >
abs( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Abs >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Abs >
abs( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Abs >( a );
}

////
// Sin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sin >
sin( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Sin >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Sin >
sin( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Sin >( a );
}

////
// Cos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Cos >
cos( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Cos >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Cos >
cos( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cos >( a );
}

////
// Tan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Tan >
tan( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Tan >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Tan >
tan( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Tan >( a );
}

////
// Sqrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sqrt >
sqrt( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Sqrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Sqrt >
sqrt( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Sqrt >( a );
}

////
// Cbrt
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Cbrt >
cbrt( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Cbrt >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Cbrt >
cbrt( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cbrt >( a );
}

////
// Pow
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation,
          typename Real >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Pow >
pow( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a, const Real& exp )
{
   auto e = Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

template< typename L1,
          template< typename > class LOperation,
          typename Real >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Pow >
pow( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a, const Real& exp )
{
   auto e = Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Pow >( a );
   e.parameter.set( exp );
   return e;
}

////
// Floor
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sin >
floor( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Floor >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Floor >
floor( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Floor >( a );
}

////
// Ceil
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Ceil >
ceil( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Ceil >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Ceil >
sin( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Ceil >( a );
}

////
// Asin
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Asin >
asin( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Asin >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Asin >
asin( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Asin >( a );
}

////
// Acos
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Acos >
cos( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Acos >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Acos >
acos( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cos >( a );
}

////
// Atan
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Atan >
tan( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Atan >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Atan >
atan( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Atan >( a );
}

////
// Sinh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Sinh >
sinh( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Sinh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Sinh >
sinh( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Sinh >( a );
}

////
// Cosh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Cosh >
cosh( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Cosh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Cosh >
cosh( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Cosh >( a );
}

////
// Tanh
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Tanh >
cosh( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Tanh >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Tanh >
tanh( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Tanh >( a );
}

////
// Log
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Log >
log( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Log >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Log >
log( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Log >( a );
}

////
// Log10
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Log10 >
log10( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Log10 >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Log10 >
log10( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Log10 >( a );
}

////
// Log2
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Log2 >
log2( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Log2 >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Log2 >
log2( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Log2 >( a );
}

////
// Exp
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
   Expressions::Exp >
exp( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >,
      Expressions::Exp >( a );
}

template< typename L1,
          template< typename > class LOperation >
__cuda_callable__
const Expressions::StaticUnaryExpressionTemplate<
   Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
   Expressions::Exp >
exp( const Expressions::StaticUnaryExpressionTemplate< L1, LOperation >& a )
{
   return Expressions::StaticUnaryExpressionTemplate<
      Expressions::StaticUnaryExpressionTemplate< L1, LOperation >,
      Expressions::Exp >( a );
}

////
// Vertical operations - min
template< typename L1,
          typename L2,
          template< typename, typename > class LOperation >
typename Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >::RealType
min( const Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >& a )
{
/*   using ExpressionType = Expressions::StaticBinaryExpressionTemplate< L1, L2, LOperation >;
   using RealType = typename ExpressionType::RealType;
   using IndexType = typename ExpressionType::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = TNL::min( a, b ); };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = TNL::min( a, b ); };
   return Reduction< DeviceType >::reduce( v1.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );*/
}


////
// Output stream
template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
std::ostream& operator << ( std::ostream& str, const StaticBinaryExpressionTemplate< T1, T2, Operation >& expression )
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
std::ostream& operator << ( std::ostream& str, const StaticUnaryExpressionTemplate< T, Operation, Parameter >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}
      } //namespace Expressions
   } //namespace Containers
} // namespace TNL
