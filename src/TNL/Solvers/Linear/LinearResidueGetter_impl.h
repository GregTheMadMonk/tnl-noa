/***************************************************************************
                          LinearResidueGetter_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cmath>

#include <TNL/Solvers/Linear/LinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {   

template< typename Matrix, typename Vector>
typename LinearResidueGetter< Matrix, Vector >::RealType
LinearResidueGetter< Matrix, Vector >::
getResidue( const Matrix& matrix,
            const Vector& x,
            const Vector& b,
            RealType bNorm )
{
   const IndexType size = matrix.getRows();   
   RealType res( 0.0 );
   if( bNorm == 0.0 )
      bNorm = b.lpNorm( 2.0 );
   for( IndexType i = 0; i < size; i ++ )
   {
      RealType err = abs( matrix.rowVectorProduct( i, x ) - b[ i ] );
      res += err * err;
   }
   return std::sqrt( res ) / bNorm;
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
