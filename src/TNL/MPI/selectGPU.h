/***************************************************************************
                          MPI/selectGPU.h  -  description
                             -------------------
    begin                : Apr 23, 2005
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cstdlib>  // std::getenv

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

   // write debug output before calling cudaSetDevice
   const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
   if( !cuda_visible_devices )
      cuda_visible_devices = "";
   std::cout << "Rank " << GetRank() << ": rank on node is " << local_rank
             << ", using GPU id " << gpuNumber << " of " << gpuCount
             << ", CUDA_VISIBLE_DEVICES=" << cuda_visible_devices
             << std::endl;

   cudaSetDevice(gpuNumber);
   TNL_CHECK_CUDA_DEVICE;
#endif
#endif
}

} // namespace MPI
} // namespace TNL
