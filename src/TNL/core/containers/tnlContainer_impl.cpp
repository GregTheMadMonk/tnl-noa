/***************************************************************************
                          tnlContainer_impl.cpp  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/core/containers/tnlContainer.h>

namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlContainer< float, Devices::Host, int >;
template class tnlContainer< double, Devices::Host, int >;
template class tnlContainer< float, Devices::Host, long int >;
template class tnlContainer< double, Devices::Host, long int >;

#ifndef HAVE_CUDA
template class tnlContainer< float, Devices::Cuda, int >;
template class tnlContainer< double, Devices::Cuda, int >;
template class tnlContainer< float, Devices::Cuda, long int >;
template class tnlContainer< double, Devices::Cuda, long int >;
#endif

#endif

} // namespace TNL


