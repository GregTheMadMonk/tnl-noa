/***************************************************************************
                          tnlDefaultBasicTypesChecker.h  -  description
                             -------------------
    begin                : Feb 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLDEFAULTBASICTYPESCHECKER_H_
#define TNLDEFAULTBASICTYPESCHECKER_H_

#include <core/tnlString.h>
#include <config/tnlParameterContainer.h>

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

#include <implementation/config/tnlDefaultBasicTypesChecker_impl.h>

#endif /* TNLDEFAULTBASICTYPESCHECKER_H_ */
