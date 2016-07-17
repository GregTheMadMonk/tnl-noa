/***************************************************************************
                          tnlTestFunction_impl.cu  -  description
                             -------------------
    begin                : Sep 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
#ifdef HAVE_CUDA

#include <functions/tnlTestFunction.h>

namespace TNL {

#ifdef INSTANTIATE_FLOAT
template class tnlTestFunction< 1, float, tnlCuda >;
template class tnlTestFunction< 2, float, tnlCuda >;
template class tnlTestFunction< 3, float, tnlCuda >;
#endif

template class tnlTestFunction< 1, double, tnlCuda >;
template class tnlTestFunction< 2, double, tnlCuda >;
template class tnlTestFunction< 3, double, tnlCuda >;

#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlTestFunction< 1, long double, tnlCuda >;
template class tnlTestFunction< 2, long double, tnlCuda >;
template class tnlTestFunction< 3, long double, tnlCuda >;
#endif

} // namespace TNL

#endif
#endif
