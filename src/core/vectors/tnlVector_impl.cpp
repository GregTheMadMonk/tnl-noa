/***************************************************************************
                          tnlVector_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <core/vectors/tnlVector.h>

namespace TNL {

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class tnlVector< float, tnlHost, int >;
template tnlVector< float, tnlHost, int >& tnlVector< float, tnlHost, int >:: operator = ( const tnlVector< double, tnlHost, int >& vector );
#endif


template class tnlVector< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlVector< long double, tnlHost, int >;
#endif
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class tnlVector< float, tnlHost, long int >;
#endif
template class tnlVector< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class tnlVector< long double, tnlHost, long int >;
#endif
#endif

#endif

} // namespace TNL

