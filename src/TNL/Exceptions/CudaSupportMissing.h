/***************************************************************************
                          CudaSupportMissing.h  -  description
                             -------------------
    begin                : Jun 18, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#include <stdexcept>

#pragma once

namespace TNL {
namespace Exceptions {

class CudaSupportMissing
   : public std::runtime_error
{
public:

   CudaSupportMissing()
   : std::runtime_error( "missing CUDA support" )
   {}

   const char* what() const throw()
   {
      return "The program called a function, which needs a CUDA support. Please recompile the program with CUDA support.";
   }
};

} // namespace Exceptions
} // namespace TNL
