/***************************************************************************
                          tnlLongVectorCUDA.h  -  description
                             -------------------
    begin                : Feb 11, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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
 
#include <core/tnlLongVectorCUDA.cu.h> 

void tnlLongVectorCUDASetValue( int* data,
                                const int size,
                                const int& v )
{
   tnlLongVectorCUDASetValueKernelCaller( data, size, v );
}

void tnlLongVectorCUDASetValue( float* data,
                                const int size,
                                const float& v )
{
   tnlLongVectorCUDASetValueKernelCaller( data, size, v );
}

void tnlLongVectorCUDASetValue( double* data,
                                const int size,
                                const double& v )
{
   tnlLongVectorCUDASetValueKernelCaller( data, size, v );
}