/***************************************************************************
                          tnlDefaultBasicTypesChecker.h  -  description
                             -------------------
    begin                : Feb 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/tnlString.h>
#include <TNL/config/tnlParameterContainer.h>

namespace TNL {

class tnlDefaultBasicTypesChecker
{
   public:

   static bool checkSupportedRealTypes( const tnlString& realType,
                                        const tnlParameterContainer& parameters );

   static bool checkSupportedIndexTypes( const tnlString& indexType,
                                         const tnlParameterContainer& parameters );

   static bool checkSupportedDevices( const tnlString& device,
                                      const tnlParameterContainer& parameters );
};

} // namespace TNL

#include <TNL/implementation/config/tnlDefaultBasicTypesChecker_impl.h>

