/***************************************************************************
                          tnlCuda.cpp  -  description
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

#include <core/tnlCuda.h>
#include <core/mfuncs.h>
#include <tnlConfig.h>
 
tnlString tnlCuda :: getDeviceType()
{
   return tnlString( "tnlCuda" );
}

int tnlCuda::getGPUTransferBufferSize()
{
   return 1 << 20;
}

int tnlCuda::getNumberOfBlocks( const int threads,
                                const int blockSize )
{
   return roundUpDivision( threads, blockSize );
}

int tnlCuda::getNumberOfGrids( const int blocks,
                               const int gridSize )
{
   return roundUpDivision( blocks, gridSize );
}

/*size_t tnlCuda::getFreeMemory()
{

}*/

