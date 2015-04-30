/***************************************************************************
                          tnlStaticArray_impl.cu  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#include <core/arrays/tnlStaticArray.h>

#ifndef HAVE_CUDA
#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlStaticArray< 1, char >;
template class tnlStaticArray< 1, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlStaticArray< 1, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class tnlStaticArray< 1, float >;
#endif
template class tnlStaticArray< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticArray< 1, long double >;
#endif

template class tnlStaticArray< 2, char >;
template class tnlStaticArray< 2, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlStaticArray< 2, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class tnlStaticArray< 2, float >;
#endif
template class tnlStaticArray< 2, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticArray< 2, long double >;
#endif

template class tnlStaticArray< 3, char >;
template class tnlStaticArray< 3, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlStaticArray< 3, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class tnlStaticArray< 3, float >;
#endif
template class tnlStaticArray< 3, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticArray< 3, long double >;
#endif

template class tnlStaticArray< 4, char >;
template class tnlStaticArray< 4, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlStaticArray< 4, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class tnlStaticArray< 4, float >;
#endif
template class tnlStaticArray< 4, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticArray< 4, long double >;
#endif

#endif
#endif