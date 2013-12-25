/***************************************************************************
                          tnlCuda.h  -  description
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

#ifndef TNLCUDA_H_
#define TNLCUDA_H_

#include <iostream>
#include <unistd.h>
#include <core/tnlDevice.h>
#include <core/tnlString.h>
#include <core/tnlAssert.h>

class tnlCuda
{
   public:

   static tnlString getDeviceType();

   static tnlDeviceEnum getDevice();

   static int getMaxGridSize();

   static void setMaxGridSize( int newMaxGridSize );

   static int getMaxBlockSize();

   static void setMaxBlockSize( int newMaxBlockSize );

   static int getGPUTransferBufferSize();

   static bool checkDevice( const char* file_name, int line );

   static size_t getFreeMemory();

   protected:

   static int maxGridSize, maxBlockSize;
};

#define checkCudaDevice tnlCuda::checkDevice( __FILE__, __LINE__ )

#define tnlCudaSupportMissingMessage \
   std::cerr << "The CUDA support is missing in the source file " << __FILE__ << " at line " << __LINE__ << ". Please set WITH_CUDA=yes in the install script. " << std::endl;

#endif /* TNLCUDA_H_ */
