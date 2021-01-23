/***************************************************************************
                          MPI/Wrappers.h  -  description
                             -------------------
    begin                : Apr 23, 2005
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Cuda/CheckDevice.h>

#include "Utils.h"

namespace TNL {
namespace MPI {

inline void selectGPU()
{
#ifdef HAVE_MPI
#ifdef HAVE_CUDA
   int gpuCount;
   cudaGetDeviceCount(&gpuCount);

   const int local_rank = getRankOnNode();
   const int gpuNumber = local_rank % gpuCount;

   cudaSetDevice(gpuNumber);
   TNL_CHECK_CUDA_DEVICE;
#endif
#endif
}

} // namespace MPI
} // namespace TNL
