/***************************************************************************
                          tnlMultiVector_impl.cpp  -  description
                             -------------------
    begin                : Jan 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <core/vectors/tnlMultiVector.h>

namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 1, float,  tnlHost, int >;
#endif
template class tnlMultiVector< 1, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 1, float,  tnlHost, long int >;
#endif
template class tnlMultiVector< 1, double, tnlHost, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 2, float,  tnlHost, int >;
#endif
template class tnlMultiVector< 2, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 2, float,  tnlHost, long int >;
#endif
template class tnlMultiVector< 2, double, tnlHost, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 3, float,  tnlHost, int >;
#endif
template class tnlMultiVector< 3, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 3, float,  tnlHost, long int >;
#endif
template class tnlMultiVector< 3, double, tnlHost, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 4, float,  tnlHost, int >;
#endif
template class tnlMultiVector< 4, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 4, float,  tnlHost, long int >;
#endif
template class tnlMultiVector< 4, double, tnlHost, long int >;
#endif

#ifdef HAVE_CUDA

#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 1, float,  tnlCuda, int >;
#endif
template class tnlMultiVector< 1, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 1, float,  tnlCuda, long int >;
#endif
template class tnlMultiVector< 1, double, tnlCuda, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 2, float,  tnlCuda, int >;
#endif
template class tnlMultiVector< 2, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 2, float,  tnlCuda, long int >;
#endif
template class tnlMultiVector< 2, double, tnlCuda, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 3, float,  tnlCuda, int >;
#endif
template class tnlMultiVector< 3, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 3, float,  tnlCuda, long int >;
#endif
template class tnlMultiVector< 3, double, tnlCuda, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 4, float,  tnlCuda, int >;
#endif
template class tnlMultiVector< 4, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlMultiVector< 4, float,  tnlCuda, long int >;
#endif
template class tnlMultiVector< 4, double, tnlCuda, long int >;
#endif

#endif

#endif

} // namespace TNL


