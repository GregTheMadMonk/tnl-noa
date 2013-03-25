/***************************************************************************
                          memory_operations_impl.cu  -  description
                             -------------------
    begin                : Mar 24, 2013
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

#include <implementation/core/memory-operations.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template bool copyMemoryCudaToHost( char* destination,
                                    const char* source,
                                    const int size );
                                    
template bool copyMemoryCudaToHost( int* destination,
                                    const int* source,
                                    const int size );

template bool copyMemoryCudaToHost( long int* destination,
                                    const long int* source,
                                    const int size );

template bool copyMemoryCudaToHost( float* destination,
                                    const float* source,
                                    const int size );

template bool copyMemoryCudaToHost( double* destination,
                                    const double* source,
                                    const int size );

template bool copyMemoryCudaToHost( char* destination,
                                    const char* source,
                                    const long int size );
                                    
template bool copyMemoryCudaToHost( int* destination,
                                    const int* source,
                                    const long int size );

template bool copyMemoryCudaToHost( long int* destination,
                                    const long int* source,
                                    const long int size );

template bool copyMemoryCudaToHost( float* destination,
                                    const float* source,
                                    const long int size );

template bool copyMemoryCudaToHost( double* destination,
                                    const double* source,
                                    const long int size );
                                                                        
#endif                                    