/***************************************************************************
                          TFQMR.h  -  description
                             -------------------
    begin                : Dec 8, 2012
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "LinearSolver.h"

#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class TFQMR
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   String getType() const;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void setSize( IndexType size );

   Containers::Vector< RealType, DeviceType, IndexType > d, r, w, u, v, r_ast, Au, M_tmp;

   IndexType size = 0;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/TFQMR_impl.h>
