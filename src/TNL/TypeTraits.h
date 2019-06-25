/***************************************************************************
                          TypeTraits.h  -  description
                             -------------------
    begin                : Jun 25, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

namespace TNL {

template< typename T >
struct ViewType
{
   using Type = T;
};

} //namespace TNL