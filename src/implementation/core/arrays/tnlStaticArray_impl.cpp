/***************************************************************************
                          tnlStaticArray_impl.cpp  -  description
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
template class tnlStaticArray< 1, long int >;
template class tnlStaticArray< 1, float >;
template class tnlStaticArray< 1, double >;
template class tnlStaticArray< 1, long double >;

template class tnlStaticArray< 2, char >;
template class tnlStaticArray< 2, int >;
template class tnlStaticArray< 2, long int >;
template class tnlStaticArray< 2, float >;
template class tnlStaticArray< 2, double >;
template class tnlStaticArray< 2, long double >;

template class tnlStaticArray< 3, char >;
template class tnlStaticArray< 3, int >;
template class tnlStaticArray< 3, long int >;
template class tnlStaticArray< 3, float >;
template class tnlStaticArray< 3, double >;
template class tnlStaticArray< 3, long double >;

template class tnlStaticArray< 4, char >;
template class tnlStaticArray< 4, int >;
template class tnlStaticArray< 4, long int >;
template class tnlStaticArray< 4, float >;
template class tnlStaticArray< 4, double >;
template class tnlStaticArray< 4, long double >;

#endif
#endif

