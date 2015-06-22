/***************************************************************************
                          tnlCudaDeviceInfo.h  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef TNLCUDADEVICEINFO_H
#define	TNLCUDADEVICEINFO_H

#include <core/tnlCuda.h>

class tnlCudaDeviceInfo
{
   public:
      
      static int getNumberOfDevices();
      
      static tnlString getDeviceName( int deviceNum );
      
      static int getArchitectureMajor( int deviceNum );
      
      static int getArchitectureMinor( int deviceNum );
      
      static int getClockRate( int deviceNum );
      
      static int getGlobalMemory( int deviceNum );

      static int getMemoryClockRate( int deviceNum );

      static bool getECCEnabled( int deviceNum );

      static int getCudaMultiprocessors( int deviceNum );      
      
      static int getCudaCoresPerMultiprocessors( int deviceNum );      
      
      static int getCudaCores( int deviceNum );      
      
};




#endif	/* TNLCUDADEVICEINFO_H */

