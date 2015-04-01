/***************************************************************************
                          tnlStaticVector_impl.cu  -  description
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

#include <core/vectors/tnlStaticVector.h>

#ifdef HAVE_CUDA
#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlStaticVector< 1, float >;
template class tnlStaticVector< 1, double >;
//template class tnlStaticVector< 1, long double >;

template class tnlStaticVector< 2, float >;
template class tnlStaticVector< 2, double >;
//template class tnlStaticVector< 2, long double >;

template class tnlStaticVector< 3, float >;
template class tnlStaticVector< 3, double >;
//template class tnlStaticVector< 3, long double >;

template class tnlStaticVector< 4, float >;
template class tnlStaticVector< 4, double >;
//template class tnlStaticVector< 4, long double >;

#endif
#endif
