/***************************************************************************
                          ExpressionVariableType.h  -  description
                             -------------------
    begin                : Apr 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Expressions/TypeTraits.h>

namespace TNL {
namespace Containers {
namespace Expressions {

enum ExpressionVariableType { ArithmeticVariable, VectorExpressionVariable, OtherVariable };

template< typename T, typename V = T >
constexpr ExpressionVariableType
getExpressionVariableType()
{
   if( std::is_arithmetic< std::decay_t< T > >::value )
      return ArithmeticVariable;
   // vectors must be considered as an arithmetic type when used as RealType in another vector
   if( IsArithmeticSubtype< T, V >::value )
      return ArithmeticVariable;
   if( HasEnabledExpressionTemplates< T >::value ||
       HasEnabledStaticExpressionTemplates< T >::value ||
       HasEnabledDistributedExpressionTemplates< T >::value
   )
      return VectorExpressionVariable;
   if( IsArrayType< T >::value || IsStaticArrayType< T >::value )
      return VectorExpressionVariable;
   return OtherVariable;
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
