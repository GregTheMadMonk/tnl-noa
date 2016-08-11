/***************************************************************************
                          SharedArray_impl.cu  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/SharedArray.h>

namespace TNL {
namespace Containers {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class SharedArray< float, Devices::Cuda, int >;
#endif
template class SharedArray< double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class SharedArray< long double, Devices::Cuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class SharedArray< float, Devices::Cuda, long int >;
#endif
template class SharedArray< double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class SharedArray< long double, Devices::Cuda, long int >;
#endif
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
