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
template class SharedVector< float, tnlHost, int >;
#endif
template class SharedVector< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, tnlHost, int >;
#endif
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class SharedVector< float, tnlHost, long int >;
#endif
template class SharedVector< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, tnlHost, long int >;
#endif
#endif

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
template class SharedVector< float, tnlCuda, int >;
#endif
template class SharedVector< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, tnlCuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class SharedVector< float, tnlCuda, long int >;
#endif
template class SharedVector< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class SharedVector< long double, tnlCuda, long int >;
#endif
#endif

#endif

#endif

} // namespace Vectors
} // namespace TNL

