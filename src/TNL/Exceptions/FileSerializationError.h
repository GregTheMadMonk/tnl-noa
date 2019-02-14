/***************************************************************************
                          FileSerializationError.h  -  description
                             -------------------
    begin                : Nov 17, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <string>
#include <stdexcept>

namespace TNL {
namespace Exceptions {

class FileSerializationError
   : public std::runtime_error
{
public:
   FileSerializationError( const std::string& objectType, const std::string& fileName )
   : std::runtime_error( "Failed to serialize object of type '" + objectType + "' into file '" + fileName + "'." )
   {}
};

} // namespace Exceptions
} // namespace TNL
