/***************************************************************************
                          Cuda.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {

class Cuda
{
public:
   static inline void configSetup( Config::ConfigDescription& config, const String& prefix = "" )
   {
#ifdef HAVE_CUDA
      config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation.", 0 );
#else
      config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation (not supported on this system).", 0 );
#endif
   }

   static inline bool setup( const Config::ParameterContainer& parameters,
                             const String& prefix = "" )
   {
#ifdef HAVE_CUDA
      int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
      if( cudaSetDevice( cudaDevice ) != cudaSuccess )
      {
         std::cerr << "I cannot activate CUDA device number " << cudaDevice << "." << std::endl;
         return false;
      }
#endif
      return true;
   }
};

} // namespace Devices
} // namespace TNL
