/***************************************************************************
                          tnlArray_impl.cu  -  description
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

#include <core/arrays/tnlArray.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef HAVE_CUDA
template class tnlArray< float, tnlCuda, int >;
template class tnlArray< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlArray< long double, tnlCuda, int >;
#endif


#ifdef INSTANTIATE_LONG_INT
template class tnlArray< float, tnlCuda, long int >;
template class tnlArray< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlArray< long double, tnlCuda, long int >;
#endif
#endif

#endif

#endif
