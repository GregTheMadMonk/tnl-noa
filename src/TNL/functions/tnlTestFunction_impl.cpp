/***************************************************************************
                          tnlTestFunction_impl.cpp  -  description
                             -------------------
    begin                : Sep 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#include <TNL/functions/tnlTestFunction.h>


namespace TNL {

#ifdef INSTANTIATE_FLOAT
template class tnlTestFunction< 1, float, Devices::Host >;
template class tnlTestFunction< 2, float, Devices::Host >;
template class tnlTestFunction< 3, float, Devices::Host >;
#endif

template class tnlTestFunction< 1, double, Devices::Host >;
template class tnlTestFunction< 2, double, Devices::Host >;
template class tnlTestFunction< 3, double, Devices::Host >;

#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlTestFunction< 1, long double, Devices::Host >;
template class tnlTestFunction< 2, long double, Devices::Host >;
template class tnlTestFunction< 3, long double, Devices::Host >;
#endif

} // namespace TNL


#endif

