/***************************************************************************
                          tnlMultiArray_impl.cu  -  description
                             -------------------
    begin                : Feb 4, 2014
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

#include <core/arrays/tnlMultiArray.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef HAVE_CUDA
// TODO: There are problems with nvlink - it maght be better in later versions
/*template class tnlMultiArray< 1, float,  tnlCuda, int >;
template class tnlMultiArray< 1, double, tnlCuda, int >;
template class tnlMultiArray< 1, float,  tnlCuda, long int >;
template class tnlMultiArray< 1, double, tnlCuda, long int >;
template class tnlMultiArray< 2, float,  tnlCuda, int >;
template class tnlMultiArray< 2, double, tnlCuda, int >;
template class tnlMultiArray< 2, float,  tnlCuda, long int >;
template class tnlMultiArray< 2, double, tnlCuda, long int >;
template class tnlMultiArray< 3, float,  tnlCuda, int >;
template class tnlMultiArray< 3, double, tnlCuda, int >;
template class tnlMultiArray< 3, float,  tnlCuda, long int >;
template class tnlMultiArray< 3, double, tnlCuda, long int >;
template class tnlMultiArray< 4, float,  tnlCuda, int >;
template class tnlMultiArray< 4, double, tnlCuda, int >;
template class tnlMultiArray< 4, float,  tnlCuda, long int >;
template class tnlMultiArray< 4, double, tnlCuda, long int >;*/

#endif

#endif