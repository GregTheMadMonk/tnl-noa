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
   static inline void configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   static inline bool setup( const Config::ParameterContainer& parameters,
                             const String& prefix = "" );
};

} // namespace Devices
} // namespace TNL

#include <TNL/Devices/Cuda_impl.h>
