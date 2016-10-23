/***************************************************************************
                          Vector_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Containers {    

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
template class Vector< float, Devices::Host, int >;
template Vector< float, Devices::Host, int >& Vector< float, Devices::Host, int >:: operator = ( const Vector< double, Devices::Host, int >& vector );
#endif


template class Vector< double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Vector< long double, Devices::Host, int >;
#endif
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
template class Vector< float, Devices::Host, long int >;
#endif
template class Vector< double, Devices::Host, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class Vector< long double, Devices::Host, long int >;
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL

