/***************************************************************************
                          LinearResidueGetter.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Solvers {
namespace Linear {

struct LinearResidueGetter
{
   template< typename Matrix, typename Vector1, typename Vector2 >
   static typename Matrix::RealType
   getResidue( const Matrix& matrix,
               const Vector1& x,
               const Vector2& b,
               typename Matrix::RealType bNorm = 0 );
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "LinearResidueGetter.hpp"
