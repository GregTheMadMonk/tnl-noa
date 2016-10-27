/***************************************************************************
                          MeshDimensionTag.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

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
 * layer dimensions but instead this tag must be used. This makes the code more difficult
 * to read and we would like to avoid it if it is possible sometime.
 * On the other hand, MeshDimensionTag is also used for method overloading when
 * asking for different mesh entities. In this case it makes sense and it cannot be
 * replaced.
 */

template< int Dimension >
class MeshDimensionTag
{
   static_assert( Dimensions >= 0, "The value of the dimensions cannot be negative." );

public:
   static constexpr int value = Dimensions;

   using Decrement = MeshDimensionsTag< Dimensions - 1 >;
   using Increment = MeshDimensionsTag< Dimensions + 1 >;
};

template<>
class MeshDimensionTag< 0 >
{
public:
   static const int value = 0;

   using Increment = MeshDimensionsTag< 1 >;
};

} // namespace Meshes
} // namespace TNL
