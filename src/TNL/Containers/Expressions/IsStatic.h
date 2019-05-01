/***************************************************************************
                          IsStatic.h  -  description
                             -------------------
    begin                : May 1, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Containers {
      namespace Expressions {


template< typename T >
struct IsStaticType
{
   static constexpr bool value = false;
};

template< int Size,
          typename Real >
struct IsStaticType< StaticVector< Size, Real > >
{
   static constexpr bool value = true;
};

      } //namespace Expressions
   } //namespace Containers
} //namespace TNL