/***************************************************************************
                          tnlSharedArray_impl.cu  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Arrays/SharedArray.h>

namespace TNL {
namespace Arrays {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class tnlSharedArray< float, tnlCuda, int >;
#endif
template class tnlSharedArray< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedArray< long double, tnlCuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlSharedArray< float, tnlCuda, long int >;
#endif
template class tnlSharedArray< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedArray< long double, tnlCuda, long int >;
#endif
#endif
#endif

#endif

} // namespace Arrays
} // namespace TNL
