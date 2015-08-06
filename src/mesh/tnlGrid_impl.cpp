/***************************************************************************
                          tnlGrid_impl.cpp  -  description
                             -------------------
    begin                : Jan 21, 2013
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

#include <mesh/tnlGrid.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlGrid< 1, float,  tnlHost, int >;
template class tnlGrid< 1, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 1, float,  tnlHost, long int >;
template class tnlGrid< 1, double, tnlHost, long int >;
#endif

template class tnlGrid< 2, float,  tnlHost, int >;
template class tnlGrid< 2, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 2, float,  tnlHost, long int >;
template class tnlGrid< 2, double, tnlHost, long int >;
#endif

template class tnlGrid< 3, float,  tnlHost, int >;
template class tnlGrid< 3, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 3, float,  tnlHost, long int >;
template class tnlGrid< 3, double, tnlHost, long int >;
#endif

#ifdef HAVE_CUDA
#endif

template class tnlGrid< 1, float,  tnlCuda, int >;
template class tnlGrid< 1, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 1, float,  tnlCuda, long int >;
template class tnlGrid< 1, double, tnlCuda, long int >;
#endif

template class tnlGrid< 2, float,  tnlCuda, int >;
template class tnlGrid< 2, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 2, float,  tnlCuda, long int >;
template class tnlGrid< 2, double, tnlCuda, long int >;
#endif

template class tnlGrid< 3, float,  tnlCuda, int >;
template class tnlGrid< 3, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 3, float,  tnlCuda, long int >;
template class tnlGrid< 3, double, tnlCuda, long int >;
#endif

#endif





