/***************************************************************************
                          tnlDimensionsTag.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>

namespace TNL {

/***
 * This tag or integer wrapper is necessary for C++ templates specializations.
 * As th C++ standard says:
 *
 *   A partially specialized non-type argument expression shall not involve
 *   a template parameter of the partial specialization except when the argument
 *   expression is a simple identifier.
 *
 * Therefore one cannot specialize the mesh layers just by integers saying the mesh
 * layer dimensions but instead this tag must be used. This makes the code more difficult
 * to read and we would like to avoid it if it is possible sometime.
 * On the other hand, tnlDimensionTag is also used for method overloading when
 * asking for different mesh entities. In this case it makes sense and it cannot be
 * replaced.
 */

template< int Dimensions >
class tnlDimensionsTag
{
   public:

      static const int value = Dimensions;

      typedef tnlDimensionsTag< Dimensions - 1 > Decrement;

      static_assert( value >= 0, "The value of the dimensions cannot be negative." );
};

template<>
class tnlDimensionsTag< 0 >
{
   public:
 
      static const int value = 0;
};

} // namespace TNL
