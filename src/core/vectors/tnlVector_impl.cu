/***************************************************************************
                          tnlVector_impl.cu  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <core/vectors/tnlVector.h>

namespace TNL {

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class tnlVector< float, tnlCuda, int >;
#endif
template class tnlVector< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlVector< long double, tnlCuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlVector< float, tnlCuda, long int >;
#endif
template class tnlVector< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlVector< long double, tnlCuda, long int >;
#endif
#endif
#endif

#endif

} // namespace TNL
