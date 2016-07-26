/***************************************************************************
                          Vector_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Vectors/Vector.h>

namespace TNL {
namespace Vectors {    

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class Vector< float, tnlHost, int >;
template Vector< float, tnlHost, int >& Vector< float, tnlHost, int >:: operator = ( const Vector< double, tnlHost, int >& vector );
#endif


template class Vector< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Vector< long double, tnlHost, int >;
#endif
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class Vector< float, tnlHost, long int >;
#endif
template class Vector< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Vector< long double, tnlHost, long int >;
#endif
#endif

#endif

} // namespace Vectors
} // namespace TNL

