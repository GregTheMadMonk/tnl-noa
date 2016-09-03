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

template< typename Matrix, typename Vector >
class LinearResidueGetter
{
   public:

      typedef Matrix MatrixType;
      typedef typename MatrixType::RealType RealType;
      typedef typename MatrixType::DeviceType DeviceType;
      typedef typename MatrixType::IndexType IndexType;

   static RealType getResidue( const Matrix& matrix,
                               const Vector& x,
                               const Vector& b,
                               RealType bNorm = 0 );
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/LinearResidueGetter_impl.h>
