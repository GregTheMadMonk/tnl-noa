/***************************************************************************
                          ExpressionVariableType.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

namespace TNL {
   namespace Containers {

template< int Size, typename Real >
class StaticVector;

      namespace Expressions {

enum ExpressionVariableType { ArithmeticVariable, VectorVariable, OtherVariable };


/**
 * SFINAE for checking if T has getSize method
 */
template< typename T >
class IsExpressionTemplate
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( typename C::IsExpressionTemplate );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< typename std::remove_reference< T >::type >(0) ) == sizeof( YesType ) );
};


template< typename T >
struct IsVectorType
{
   static constexpr bool value = false;
};

template< int Size,
          typename Real >
struct IsVectorType< StaticVector< Size, Real > >
{
   static constexpr bool value = true;
};

template< typename T,
          bool IsArithmetic = std::is_arithmetic< T >::value,
          bool IsVector = IsVectorType< T >::value || IsExpressionTemplate< T >::value >
struct ExpressionVariableTypeGetter
{
   static constexpr ExpressionVariableType value = OtherVariable;
};

template< typename T >
struct  ExpressionVariableTypeGetter< T, true, false >
{
   static constexpr ExpressionVariableType value = ArithmeticVariable;
};

template< typename T >
struct ExpressionVariableTypeGetter< T, false, true >
{
   static constexpr ExpressionVariableType value = VectorVariable;
};

      } //namespace Expressions
   } //namespace Containers
} //namespace TNL
