/***************************************************************************
                          tnlSharedArray_impl.cu  -  description
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

#include <core/arrays/tnlSharedArray.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef HAVE_CUDA
template class tnlSharedArray< float, tnlCuda, int >;
template class tnlSharedArray< double, tnlCuda, int >;
template class tnlSharedArray< float, tnlCuda, long int >;
template class tnlSharedArray< double, tnlCuda, long int >;
#endif

#endif