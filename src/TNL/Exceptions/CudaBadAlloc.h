/***************************************************************************
                          CudaBadAlloc.h  -  description
                             -------------------
    begin                : Jun 18, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#include <new>

#pragma once

namespace TNL {
namespace Exceptions {

struct CudaBadAlloc
   : public std::bad_alloc
{
   const char* what() const throw()
   {
      return "Failed to allocate memory on the CUDA device: "
             "most likely there is not enough space on the device memory.";
   }
};

} // namespace Exceptions
} // namespace TNL
