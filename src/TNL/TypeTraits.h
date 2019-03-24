/***************************************************************************
                          TypeTraits.h  -  description
                             -------------------
    begin                : Mar 24, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

   template< typename T >
   struct isArray
   {
      static constexpr bool value = false;
   };

} // namespace TNL