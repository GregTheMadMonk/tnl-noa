/***************************************************************************
                          tnlDefaultBasicTypesChecker.h  -  description
                             -------------------
    begin                : Feb 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Config {   

class tnlDefaultBasicTypesChecker
{
   public:

   static bool checkSupportedRealTypes( const String& realType,
                                        const Config::ParameterContainer& parameters );

   static bool checkSupportedIndexTypes( const String& indexType,
                                         const Config::ParameterContainer& parameters );

   static bool checkSupportedDevices( const String& device,
                                      const Config::ParameterContainer& parameters );
};

} // namespace Config
} // namespace TNL

#include <TNL/implementation/config/tnlDefaultBasicTypesChecker_impl.h>

