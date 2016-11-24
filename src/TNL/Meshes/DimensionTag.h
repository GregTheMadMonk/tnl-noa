/***************************************************************************
                          DimensionTag.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>

namespace TNL {
namespace Meshes {

/***
 * This tag or integer wrapper is necessary for C++ templates specializations.
 * As the C++ standard says:
 *
 *   A partially specialized non-type argument expression shall not involve
 *   a template parameter of the partial specialization except when the argument
 *   expression is a simple identifier.
 *
 * Therefore one cannot specialize the mesh layers just by integers saying the mesh
 * layer dimension but instead this tag must be used. This makes the code more difficult
 * to read and we would like to avoid it if it is possible sometime.
 * On the other hand, DimensionTag is also used for method overloading when
 * asking for different mesh entities. In this case it makes sense and it cannot be
 * replaced.
 */

template< int Dimension >
class DimensionTag
{
   static_assert( Dimension >= 0, "The dimension cannot be negative." );

public:
   static constexpr int value = Dimension;

   using Decrement = DimensionTag< Dimension - 1 >;
   using Increment = DimensionTag< Dimension + 1 >;
};

template<>
class DimensionTag< 0 >
{
public:
   static const int value = 0;

   using Increment = DimensionTag< 1 >;
};

} // namespace Meshes
} // namespace TNL
