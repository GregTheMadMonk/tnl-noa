/***************************************************************************
                          Diagonal.h  -  description
                             -------------------
    begin                : Dec 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Preconditioner.h"

#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix >
class Diagonal
: public Preconditioner< Matrix >
{
public:
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   virtual void update( const Matrix& matrix ) override;

   virtual bool solve( ConstVectorViewType b, VectorViewType x ) const override;

   String getType() const
   {
      return String( "Diagonal" );
   }

protected:
   VectorType diagonal;
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Preconditioners/Diagonal_impl.h>
