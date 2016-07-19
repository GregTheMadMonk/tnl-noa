/***************************************************************************
                          tnlArray_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/core/arrays/tnlArray.h>

namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class tnlArray< float, tnlHost, int >;
#endif
template class tnlArray< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlArray< long double, tnlHost, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlArray< float, tnlHost, long int >;
#endif
template class tnlArray< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlArray< long double, tnlHost, long int >;
#endif
#endif

#ifndef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class tnlArray< float, tnlCuda, int >;
#endif
template class tnlArray< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlArray< long double, tnlCuda, int >;
#endif


#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlArray< float, tnlCuda, long int >;
#endif
template class tnlArray< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlArray< long double, tnlCuda, long int >;
#endif
#endif

#endif

#endif

} // namespace TNL