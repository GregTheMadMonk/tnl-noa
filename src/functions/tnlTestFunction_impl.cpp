/***************************************************************************
                          tnlTestFunction_impl.cpp  -  description
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

#include <functions/tnlTestFunction.h>

#ifdef INSTANTIATE_FLOAT
template class tnlTestFunction< 1, float, tnlHost >;
template class tnlTestFunction< 2, float, tnlHost >;
template class tnlTestFunction< 3, float, tnlHost >;
#endif

template class tnlTestFunction< 1, double, tnlHost >;
template class tnlTestFunction< 2, double, tnlHost >;
template class tnlTestFunction< 3, double, tnlHost >;

#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlTestFunction< 1, long double, tnlHost >;
template class tnlTestFunction< 2, long double, tnlHost >;
template class tnlTestFunction< 3, long double, tnlHost >;
#endif

#endif

