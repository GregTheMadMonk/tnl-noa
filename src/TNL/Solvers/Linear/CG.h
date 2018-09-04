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
#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class CG
: public LinearSolver< Matrix >,
  public IterativeSolver< typename Matrix::RealType,
                          typename Matrix::IndexType >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   String getType() const;

   // to avoid ambiguity
   using Base::configSetup;

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void setSize( IndexType size );

   Containers::Vector< RealType, DeviceType, IndexType >  r, new_r, p, Ap;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/CG_impl.h>
