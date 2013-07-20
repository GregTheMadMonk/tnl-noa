/***************************************************************************
                          tnlHost.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLHOSTL_H_
#define TNLHOSTL_H_

#include <core/tnlHost.h>

tnlString tnlHost :: getDeviceType()
{
   return tnlString( "tnlHost" );
};

tnlDeviceEnum tnlHost :: getDevice()
{
   return tnlHostDevice;
};

#endif /* TNLHOST_H_ */
