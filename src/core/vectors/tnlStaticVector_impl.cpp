/***************************************************************************
                          tnlStaticVector_impl.cpp  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <core/vectors/tnlStaticVector.h>

#ifndef HAVE_CUDA
#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class tnlStaticVector< 1, float >;
#endif
template class tnlStaticVector< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticVector< 1, long double >;
#endif

#ifdef INSTANTIATE_FLOAT
template class tnlStaticVector< 2, float >;
#endif
template class tnlStaticVector< 2, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticVector< 2, long double >;
#endif

#ifdef INSTANTIATE_FLOAT
template class tnlStaticVector< 3, float >;
#endif
template class tnlStaticVector< 3, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticVector< 3, long double >;
#endif

#ifdef INSTANTIATE_FLOAT
template class tnlStaticVector< 4, float >;
#endif
template class tnlStaticVector< 4, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlStaticVector< 4, long double >;
#endif

#endif
#endif


