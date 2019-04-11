/***************************************************************************
                          ArrayWrongSize.h  -  description
                             -------------------
    begin                : Mar 8, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <string>
#include <stdexcept>
#include <TNL/String.h>

namespace TNL {
namespace Exceptions {

class ArrayWrongSize
   : public std::runtime_error
{
public:
   ArrayWrongSize( std::size_t size, const String& mesg = "" )
   : std::runtime_error( "Wrong array size " + convertToString( size ) + ". " + mesg )
   {}
};

} // namespace Exceptions
} // namespace TNL
