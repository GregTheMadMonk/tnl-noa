/***************************************************************************
                          tnlCuda_impl.h  -  description
                             -------------------
    begin                : Jul 11, 2013
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

#ifndef TNLCUDA_IMPL_H_
#define TNLCUDA_IMPL_H_

inline tnlString tnlCuda :: getDeviceType()
{
   return tnlString( "tnlCuda" );
}

inline tnlDeviceEnum tnlCuda :: getDevice()
{
   return tnlCudaDevice;
};

inline int tnlCuda :: getMaxGridSize()
{
   return maxGridSize;
}

inline void tnlCuda :: setMaxGridSize( int newMaxGridSize )
{
   maxGridSize = newMaxGridSize;
}

inline int tnlCuda :: getMaxBlockSize()
{
   return maxBlockSize;
}

inline void tnlCuda :: setMaxBlockSize( int newMaxBlockSize )
{
   maxBlockSize = newMaxBlockSize;
}

inline int tnlCuda::getGPUTransferBufferSize()
{
   return 1 << 20;
}


#endif /* TNLCUDA_IMPL_H_ */
