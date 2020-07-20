/***************************************************************************
                          LinearResidueGetter_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/Linear/LinearResidueGetter.h>
#include <TNL/Solvers/Linear/Traits.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix, typename Vector1, typename Vector2 >
typename Matrix::RealType
LinearResidueGetter::
getResidue( const Matrix& matrix,
            const Vector1& x,
            const Vector2& b,
            typename Matrix::RealType bNorm )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = typename Traits< Matrix >::VectorType;

   if( bNorm == 0.0 )
      bNorm = lpNorm( b, 2.0 );
   VectorType v;
   v.setLike( b );
   matrix.vectorProduct( x, v );
   return l2Norm( v - b ) / bNorm;
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
