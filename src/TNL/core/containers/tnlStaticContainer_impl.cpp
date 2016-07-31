/***************************************************************************
                          tnlStaticContainer_impl.cpp  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/core/containers/tnlStaticContainer.h>

namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA

template class tnlStaticContainer< 1, char >;
template class tnlStaticContainer< 1, int >;
template class tnlStaticContainer< 1, float >;
template class tnlStaticContainer< 1, double >;

template class tnlStaticContainer< 2, char >;
template class tnlStaticContainer< 2, int >;
template class tnlStaticContainer< 2, float >;
template class tnlStaticContainer< 2, double >;

template class tnlStaticContainer< 3, char >;
template class tnlStaticContainer< 3, int >;
template class tnlStaticContainer< 3, float >;
template class tnlStaticContainer< 3, double >;

template class tnlStaticContainer< 4, char >;
template class tnlStaticContainer< 4, int >;
template class tnlStaticContainer< 4, float >;
template class tnlStaticContainer< 4, double >;

#endif /* ! HAVE_CUDA */

#endif /* TEMPLATE_EXPLICIT_INSTANTIATION */

} // namespace TNL


