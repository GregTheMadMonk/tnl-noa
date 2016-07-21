/***************************************************************************
                          Array_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Arrays/Array.h>

namespace TNL {
namespace Arrays {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class Array< float, tnlHost, int >;
#endif
template class Array< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, tnlHost, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class Array< float, tnlHost, long int >;
#endif
template class Array< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, tnlHost, long int >;
#endif
#endif

#ifndef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class Array< float, tnlCuda, int >;
#endif
template class Array< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, tnlCuda, int >;
#endif


#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class Array< float, tnlCuda, long int >;
#endif
template class Array< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, tnlCuda, long int >;
#endif
#endif

#endif

#endif

} // namespace Arrays
} // namespace TNL