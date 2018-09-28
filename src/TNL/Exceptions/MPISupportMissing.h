/***************************************************************************
                          MPISupportMissing.h  -  description
                             -------------------
    begin                : Jun 11, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <stdexcept>

namespace TNL {
namespace Exceptions {

struct MPISupportMissing
   : public std::runtime_error
{
   MPISupportMissing()
   : std::runtime_error( "MPI support is missing, but the program called a function which needs it. "
                         "Please recompile the program with MPI support." )
   {}
};

} // namespace Exceptions
} // namespace TNL
