/***************************************************************************
                          tnlDefaultBasicTypesChecker.h  -  description
                             -------------------
    begin                : Feb 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

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
