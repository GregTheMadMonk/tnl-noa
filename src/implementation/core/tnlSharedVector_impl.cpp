/***************************************************************************
                          tnlSharedVector_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
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

#include <core/tnlSharedVector.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlSharedVector< float, tnlHost, int >;
template class tnlSharedVector< double, tnlHost, int >;
template class tnlSharedVector< float, tnlHost, long int >;
template class tnlSharedVector< double, tnlHost, long int >;

#ifdef HAVE_CUDA
template class tnlSharedVector< float, tnlCuda, int >;
template class tnlSharedVector< double, tnlCuda, int >;
template class tnlSharedVector< float, tnlCuda, long int >;
template class tnlSharedVector< double, tnlCuda, long int >;
#endif

#endif



