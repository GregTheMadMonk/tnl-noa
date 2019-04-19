/***************************************************************************
                          NotImplementedError.h  -  description
                             -------------------
    begin                : Apr 12, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdexcept>

namespace TNL {
namespace Exceptions {

struct NotImplementedError
   : public std::runtime_error
{
   NotImplementedError( std::string msg = "Something is not implemented." )
   : std::runtime_error( msg )
   {}
};

} // namespace Exceptions
} // namespace TNL
