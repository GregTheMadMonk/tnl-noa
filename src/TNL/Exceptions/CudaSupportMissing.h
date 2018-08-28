/***************************************************************************
                          CudaSupportMissing.h  -  description
                             -------------------
    begin                : Jun 18, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdexcept>

namespace TNL {
namespace Exceptions {

struct CudaSupportMissing
   : public std::runtime_error
{
   CudaSupportMissing()
   : std::runtime_error( "CUDA support is missing, but the program called a function which needs it. "
                         "Please recompile the program with CUDA support." )
   {}
};

} // namespace Exceptions
} // namespace TNL
