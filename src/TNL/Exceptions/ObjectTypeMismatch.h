/***************************************************************************
                          ObjectTypeMismatch.h  -  description
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

class ObjectTypeMismatch
   : public std::runtime_error
{
public:
   ObjectTypeMismatch( const String& expected, const String& detected )
   : std::runtime_error( "Object type mismatch. Expected object type is " + expected + " but " + detected + " was detcted." )
   {}
};

} // namespace Exceptions
} // namespace TNL
