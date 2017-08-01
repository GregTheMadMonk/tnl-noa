/***************************************************************************
                          MICSupportMissing.h  -  description
                             -------------------
    begin                : Jul 31, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdexcept>

namespace TNL {
namespace Exceptions {

struct MICSupportMissing
   : public std::runtime_error
{
   MICSupportMissing()
   : std::runtime_error( "MIC support is missing, but the program called a function which needs it. "
                         "Please recompile the program with MIC support." )
   {}
};

} // namespace Exceptions
} // namespace TNL
