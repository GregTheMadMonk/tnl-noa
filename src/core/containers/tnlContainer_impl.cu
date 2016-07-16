/***************************************************************************
                          tnlContainer_impl.cu  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <core/containers/tnlContainer.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef HAVE_CUDA
template class tnlContainer< float, tnlCuda, int >;
template class tnlContainer< double, tnlCuda, int >;
template class tnlContainer< float, tnlCuda, long int >;
template class tnlContainer< double, tnlCuda, long int >;
#endif

#endif
