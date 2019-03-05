/***************************************************************************
                          FileCloseError.h  -  description
                             -------------------
    begin                : Mar 5, 2019
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

class FileCloseError
   : public std::runtime_error
{
public:
   FileCloseError( const String& fileName )
   : std::runtime_error( "An error occurred when closing file " + fileName + "." )
   {}
};

} // namespace Exceptions
} // namespace TNL
