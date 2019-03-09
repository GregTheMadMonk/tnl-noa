/***************************************************************************
                          MeshFunctionDataMismatch.h  -  description
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

class MeshFunctionDataMismatch
   : public std::runtime_error
{
public:
   MeshFunctionDataMismatch( std::size_t size, const String& mesg = "" )
   : std::runtime_error( "Mesh function data size " + convertToString( size ) + " mismatch." + mesg )
   {}
};

} // namespace Exceptions
} // namespace TNL
