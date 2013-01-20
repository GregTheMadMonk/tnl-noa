/***************************************************************************
                          tnlVector_impl.cpp  -  description
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

#include <core/tnlVector.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlVector< float, tnlHost, int >;
template class tnlVector< double, tnlHost, int >;
template class tnlVector< float, tnlHost, long int >;
template class tnlVector< double, tnlHost, long int >;

#ifdef HAVE_CUDA
template class tnlVector< float, tnlCuda, int >;
template class tnlVector< double, tnlCuda, int >;
template class tnlVector< float, tnlCuda, long int >;
template class tnlVector< double, tnlCuda, long int >;
#endif

#endif


