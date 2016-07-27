/***************************************************************************
                          SharedVector_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Vectors/SharedVector.h>

namespace TNL {
namespace Vectors {    

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class SharedVector< float, Devices::Host, int >;
#endif
template class SharedVector< double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, Devices::Host, int >;
#endif
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class SharedVector< float, Devices::Host, long int >;
#endif
template class SharedVector< double, Devices::Host, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, Devices::Host, long int >;
#endif
#endif

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class SharedVector< float, Devices::Cuda, int >;
#endif
template class SharedVector< double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, Devices::Cuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class SharedVector< float, Devices::Cuda, long int >;
#endif
template class SharedVector< double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, Devices::Cuda, long int >;
#endif
#endif

#endif

#endif

} // namespace Vectors
} // namespace TNL

