/***************************************************************************
                          CG.h  -  description
                             -------------------
    begin                : 2007/07/31
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
class CG
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void setSize( IndexType size );

   Containers::Vector< RealType, DeviceType, IndexType >  r, p, Ap, z;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/CG_impl.h>
