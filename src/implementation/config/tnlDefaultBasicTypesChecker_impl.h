/***************************************************************************
                          tnlDefaultBasicTypesChecker_impl.h  -  description
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

#ifndef TNLDEFAULTBASICTYPESCHECKER_IMPL_H_
#define TNLDEFAULTBASICTYPESCHECKER_IMPL_H_

bool tnlDefaultBasicTypesChecker :: checkSupportedRealTypes( const tnlString& realType,
                                                             const tnlParameterContainer& parameters )
{
   return true;
}

bool tnlDefaultBasicTypesChecker :: checkSupportedIndexTypes( const tnlString& indexType,
                                                              const tnlParameterContainer& parameters )
{
   return true;
}

bool tnlDefaultBasicTypesChecker :: checkSupportedDevices( const tnlString& device,
                                                           const tnlParameterContainer& parameters )
{
   return true;
}


#endif /* TNLDEFAULTBASICTYPESCHECKER_IMPL_H_ */
