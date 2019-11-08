/***************************************************************************
                          CheckDevice.h  -  description
                             -------------------
    begin                : Aug 18, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Exceptions/CudaRuntimeError.h>

namespace TNL {
namespace Cuda {

#ifdef HAVE_CUDA
   /****
    * I do not know why, but it is more reliable to pass the error code instead
    * of calling cudaGetLastError() inside the function.
    * We recommend to use macro 'TNL_CHECK_CUDA_DEVICE' defined bellow.
    */
   inline void checkDevice( const char* file_name, int line, cudaError error )
   {
      if( error != cudaSuccess )
         throw Exceptions::CudaRuntimeError( error, file_name, line );
   }
#else
   inline void checkDevice() {}
#endif

} // namespace Cuda
} // namespace TNL

#ifdef HAVE_CUDA
#define TNL_CHECK_CUDA_DEVICE ::TNL::Cuda::checkDevice( __FILE__, __LINE__, cudaGetLastError() )
#else
#define TNL_CHECK_CUDA_DEVICE ::TNL::Cuda::checkDevice()
#endif
