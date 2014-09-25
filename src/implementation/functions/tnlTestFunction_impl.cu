/***************************************************************************
                          tnlTestFunction_impl.cu  -  description
                             -------------------
    begin                : Sep 21, 2014
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

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
#ifdef HAVE_CUDA

#include <functions/tnlTestFunction.h>

template class tnlTestFunction< 1, float, tnlCuda >;
template class tnlTestFunction< 2, float, tnlCuda >;
template class tnlTestFunction< 3, float, tnlCuda >;

template class tnlTestFunction< 1, double, tnlCuda >;
template class tnlTestFunction< 2, double, tnlCuda >;
template class tnlTestFunction< 3, double, tnlCuda >;

template class tnlTestFunction< 1, long double, tnlCuda >;
template class tnlTestFunction< 2, long double, tnlCuda >;
template class tnlTestFunction< 3, long double, tnlCuda >;

#endif
#endif
