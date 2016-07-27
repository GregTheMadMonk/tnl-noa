/***************************************************************************
                          tnlGrid_impl.cpp  -  description
                             -------------------
    begin                : Jan 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/mesh/tnlGrid.h>

namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlGrid< 1, float,  Devices::Host, int >;
template class tnlGrid< 1, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 1, float,  Devices::Host, long int >;
template class tnlGrid< 1, double, Devices::Host, long int >;
#endif

template class tnlGrid< 2, float,  Devices::Host, int >;
template class tnlGrid< 2, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 2, float,  Devices::Host, long int >;
template class tnlGrid< 2, double, Devices::Host, long int >;
#endif

template class tnlGrid< 3, float,  Devices::Host, int >;
template class tnlGrid< 3, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 3, float,  Devices::Host, long int >;
template class tnlGrid< 3, double, Devices::Host, long int >;
#endif

#ifdef HAVE_CUDA
#endif

template class tnlGrid< 1, float,  Devices::Cuda, int >;
template class tnlGrid< 1, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 1, float,  Devices::Cuda, long int >;
template class tnlGrid< 1, double, Devices::Cuda, long int >;
#endif

template class tnlGrid< 2, float,  Devices::Cuda, int >;
template class tnlGrid< 2, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 2, float,  Devices::Cuda, long int >;
template class tnlGrid< 2, double, Devices::Cuda, long int >;
#endif

template class tnlGrid< 3, float,  Devices::Cuda, int >;
template class tnlGrid< 3, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
template class tnlGrid< 3, float,  Devices::Cuda, long int >;
template class tnlGrid< 3, double, Devices::Cuda, long int >;
#endif

#endif

} // namespace TNL



