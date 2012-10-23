/***************************************************************************
                          typename.h  -  description
                             -------------------
    begin                : Oct 22, 2012
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

#ifndef TNLDEVICE_H_
#define TNLDEVICE_H_

#include <core/tnlString.h>
#include <core/tnlString.h>

enum tnlDeviceEnum { tnlHostDevice, tnlCudaDevice };

class tnlHost
{   
   public:

   static tnlString getDeviceType()
   {
      return tnlString( "tnlHost" );
   }

   static tnlDeviceEnum getDevice()
   {
      return tnlHostDevice;
   };
};

class tnlCuda
{
   public:

   static tnlString getDeviceType()
   {
      return tnlString( "tnlCuda" );      
   }

   static tnlDeviceEnum getDevice()
   {
      return tnlCudaDevice;
   };

};


#endif /* TNLDEVICE_H_ */
