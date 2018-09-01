/***************************************************************************
                          Array_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#include <TNL/Containers/Array.h>

namespace TNL {
namespace Containers {

#ifdef INSTANTIATE_FLOAT
template class Array< float, Devices::Host, int >;
#endif
template class Array< double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, Devices::Host, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class Array< float, Devices::Host, long int >;
#endif
template class Array< double, Devices::Host, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, Devices::Host, long int >;
#endif
#endif

#ifndef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class Array< float, Devices::Cuda, int >;
#endif
template class Array< double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, Devices::Cuda, int >;
#endif


#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class Array< float, Devices::Cuda, long int >;
#endif
template class Array< double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Array< long double, Devices::Cuda, long int >;
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL

#endif // #ifdef TEMPLATE_EXPLICIT_INSTANTIATION
