/***************************************************************************
                          TypeTraits.h  -  description
                             -------------------
    begin                : Jul 26, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

namespace TNL {
namespace Containers {
namespace Expressions {

template< typename T >
struct IsExpressionTemplate : std::false_type
{};

template< typename T >
struct IsNumericExpression
: std::integral_constant< bool,
      std::is_arithmetic< T >::value ||
      IsExpressionTemplate< T >::value >
{};

} //namespace Expressions
} //namespace Containers
} //namespace TNL
