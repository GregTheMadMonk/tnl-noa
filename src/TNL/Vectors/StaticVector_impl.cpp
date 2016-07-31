/***************************************************************************
                          StaticVector_impl.cpp  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Vectors/StaticVector.h>

namespace TNL {
namespace Vectors {    

#ifndef HAVE_CUDA
#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class StaticVector< 1, float >;
#endif
template class StaticVector< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticVector< 1, long double >;
#endif

#ifdef INSTANTIATE_FLOAT
template class StaticVector< 2, float >;
#endif
template class StaticVector< 2, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticVector< 2, long double >;
#endif

#ifdef INSTANTIATE_FLOAT
template class StaticVector< 3, float >;
#endif
template class StaticVector< 3, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticVector< 3, long double >;
#endif

#ifdef INSTANTIATE_FLOAT
template class StaticVector< 4, float >;
#endif
template class StaticVector< 4, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticVector< 4, long double >;
#endif

#endif
#endif

} // namespace Vectors
} // namespace TNL
