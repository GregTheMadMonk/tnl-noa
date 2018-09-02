/***************************************************************************
                          UnsupportedDimension.h  -  description
                             -------------------
    begin                : Aug 14, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

namespace TNL {
namespace Exceptions {

struct UnsupportedDimension
{
   UnsupportedDimension( int Dimension )
   : Dimension( Dimension )
   {
   }
   
   const char* what() const throw()
   {
      return "This dimension is not supported (yet).";
   }
   
   int Dimension;
};

} // namespace Exceptions
} // namespace TNL
