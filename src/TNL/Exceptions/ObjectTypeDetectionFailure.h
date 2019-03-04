/***************************************************************************
                          ObjectTypeDetectionFailure.h  -  description
                             -------------------
    begin                : Mar 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <string>
#include <stdexcept>

namespace TNL {
namespace Exceptions {

class ObjectTypeDetectionFailure
   : public std::runtime_error
{
public:
   ObjectTypeDetectionFailure( const String& fileName, const String& objectType )
   : std::runtime_error( "Failed to detect " + objectType + " in file " + fileName + "." )
   {}
};

} // namespace Exceptions
} // namespace TNL
