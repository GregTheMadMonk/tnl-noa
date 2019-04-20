/***************************************************************************
                          FileDeserializationError.h  -  description
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

class FileDeserializationError
   : public std::runtime_error
{
public:
   FileDeserializationError( const std::string& fileName, const std::string& details )
   : std::runtime_error( "Failed to deserialize an object from the file '" + fileName + "': " + details )
   {}
};

} // namespace Exceptions
} // namespace TNL
