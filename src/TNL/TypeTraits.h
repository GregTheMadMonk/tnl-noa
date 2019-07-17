/***************************************************************************
                          TypeTraits.h  -  description
                             -------------------
    begin                : Jun 25, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once
#include <type_traits>

namespace TNL {

// TODO: remove - not used anywhere
template< typename T >
struct ViewTypeGetter
{
   using Type = T;
};

template< typename T >
struct IsStatic
{
   static constexpr bool Value = std::is_arithmetic< T >::value;
};

} //namespace TNL
