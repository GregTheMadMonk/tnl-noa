/***************************************************************************
                          ExpressionVariableType.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Devices/Host.h>

namespace TNL {
namespace Containers {
namespace Expressions {

enum ExpressionVariableType { ArithmeticVariable, VectorExpressionVariable, OtherVariable };

template< typename T,
          bool IsArithmetic = std::is_arithmetic< T >::value,
          bool IsVector = HasSubscriptOperator< T >::value >
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
   static constexpr ExpressionVariableType value = VectorExpressionVariable;
};

////
// Non-static expression templates might be passed on GPU, for example. In this
// case, we cannot store ET operands using references but we need to make copies.
template< typename T,
          typename Device >
struct OperandType
{
   using type = std::add_const_t< std::remove_reference_t< T > >;
};

template< typename T >
struct OperandType< T, Devices::Host >
{
   using type = std::add_const_t< std::add_lvalue_reference_t< T > >;
};

} // namespace Expressions
} // namespace Containers
} // namespace TNL
