/***************************************************************************
                          CudaBadAlloc.h  -  description
                             -------------------
    begin                : Jun 18, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <new>

namespace TNL {
namespace Exceptions {

struct CudaBadAlloc
   : public std::bad_alloc
{
   CudaBadAlloc()
   {
#ifdef HAVE_CUDA
      // Make sure to clear the CUDA error, otherwise the exception handler
      // might throw another exception with the same error.
      cudaGetLastError();
#endif
   }

   const char* what() const throw()
   {
      return "Failed to allocate memory on the CUDA device: "
             "most likely there is not enough space on the device memory.";
   }
};

} // namespace Exceptions
} // namespace TNL
