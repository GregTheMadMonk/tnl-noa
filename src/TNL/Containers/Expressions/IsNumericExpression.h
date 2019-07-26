/***************************************************************************
                          IsNumericExpression.h  -  description
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
struct IsNumericExpression : std::is_arithmetic< T >
{};

} //namespace Expressions
} //namespace Containers
} //namespace TNL
