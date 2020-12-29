/***************************************************************************
                          MPI/Wrappers.h  -  description
                             -------------------
    begin                : Apr 23, 2005
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cstring>

#include <TNL/Cuda/CheckDevice.h>

namespace TNL {
namespace MPI {
namespace {

#ifdef HAVE_MPI
#ifdef HAVE_CUDA
   typedef struct __attribute__((__packed__)) {
      char name[MPI_MAX_PROCESSOR_NAME];
   } procName;
#endif
#endif

inline void selectGPU()
{
#ifdef HAVE_MPI
#ifdef HAVE_CUDA
   int size;
   MPI_Comm_size( MPI_COMM_WORLD, &size );
   int rank;
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   int gpuCount;
   cudaGetDeviceCount( &gpuCount );

   procName names[size];

   int i=0;
   int len;
   MPI_Get_processor_name(names[rank].name, &len);

   for(i=0;i<size;i++)
      std::memcpy(names[i].name,names[rank].name,len+1);

   MPI_Alltoall( (void*)names ,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
      (void*)names,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
               MPI_COMM_WORLD);

   int nodeRank=0;
   for(i=0;i<rank;i++)
   {
      if(std::strcmp(names[rank].name,names[i].name)==0)
         nodeRank++;
   }

   const int gpuNumber = nodeRank % gpuCount;

   cudaSetDevice(gpuNumber);
   TNL_CHECK_CUDA_DEVICE;

   //std::cout<<"Node: " << rank << " gpu: " << gpuNumber << std::endl;
#endif
#endif
}

} // namespace <unnamed>
} // namespace MPI
} // namespace TNL
