/***************************************************************************
                          device-check.h -  description
                             -------------------
    begin                : Mar 20, 2013
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

#ifndef ERROR_CHECK_H_
#define ERROR_CHECK_H_

#include <iostream>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

using namespace std;

#define checkCudaDevice __checkCudaDevice( __FILE__, __LINE__ )

bool __checkCudaDevice( const char* file_name, int line );

#endif /* ERROR_CHECK_H_ */
