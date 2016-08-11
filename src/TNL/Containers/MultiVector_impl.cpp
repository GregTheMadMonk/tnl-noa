/***************************************************************************
                          MultiVector_impl.cpp  -  description
                             -------------------
    begin                : Jan 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/MultiVector.h>

namespace TNL {
namespace Containers {    

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class MultiVector< 1, float,  Devices::Host, int >;
#endif
template class MultiVector< 1, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 1, float,  Devices::Host, long int >;
#endif
template class MultiVector< 1, double, Devices::Host, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
template class MultiVector< 2, float,  Devices::Host, int >;
#endif
template class MultiVector< 2, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 2, float,  Devices::Host, long int >;
#endif
template class MultiVector< 2, double, Devices::Host, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
template class MultiVector< 3, float,  Devices::Host, int >;
#endif
template class MultiVector< 3, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 3, float,  Devices::Host, long int >;
#endif
template class MultiVector< 3, double, Devices::Host, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
template class MultiVector< 4, float,  Devices::Host, int >;
#endif
template class MultiVector< 4, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 4, float,  Devices::Host, long int >;
#endif
template class MultiVector< 4, double, Devices::Host, long int >;
#endif

#ifdef HAVE_CUDA

#ifdef INSTANTIATE_FLOAT
template class MultiVector< 1, float,  Devices::Cuda, int >;
#endif
template class MultiVector< 1, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 1, float,  Devices::Cuda, long int >;
#endif
template class MultiVector< 1, double, Devices::Cuda, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 2, float,  Devices::Cuda, int >;
#endif
template class MultiVector< 2, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 2, float,  Devices::Cuda, long int >;
#endif
template class MultiVector< 2, double, Devices::Cuda, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 3, float,  Devices::Cuda, int >;
#endif
template class MultiVector< 3, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 3, float,  Devices::Cuda, long int >;
#endif
template class MultiVector< 3, double, Devices::Cuda, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 4, float,  Devices::Cuda, int >;
#endif
template class MultiVector< 4, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class MultiVector< 4, float,  Devices::Cuda, long int >;
#endif
template class MultiVector< 4, double, Devices::Cuda, long int >;
#endif

#endif

#endif
} // namespace Containers
} // namespace TNL


