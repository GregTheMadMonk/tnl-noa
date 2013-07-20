/***************************************************************************
                          tnlVectorOperations_impl.cpp  -  description
                             -------------------
    begin                : Jul 20, 2013
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

#include <core/vectors/tnlVectorOperations.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template float       tnlVectorOperations< tnlHost >::getVectorMax( const float* v,       const int size );
template double      tnlVectorOperations< tnlHost >::getVectorMax( const double* v,      const int size );
template long double tnlVectorOperations< tnlHost >::getVectorMax( const long double* v, const int size );
template float       tnlVectorOperations< tnlHost >::getVectorMax( const float* v,       const long int size );
template double      tnlVectorOperations< tnlHost >::getVectorMax( const double* v,      const long int size );
template long double tnlVectorOperations< tnlHost >::getVectorMax( const long double* v, const long int size );

#endif


